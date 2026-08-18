"""Production pipeline builder (Commit 6) — the ONE real graph.

``build_real_pipeline(config, on_decision=None, **injectables)`` assembles
the approved recovery stack:

    RealSegmenter(SileroVad) -> EngineManager(SeamlessM4TEngine) ->
    BalancedQualityGate -> [primary PASS | FallbackHandler:
    conservative M4T retry (same model, beam decoding) ->
    FixedSplitRechunker composition -> Whisper+M4T ASR/MT -> flagged] ->
    RealPostProcessor

Both the CLI (``sawti transcribe --engine m4t``) and the evaluator
(``make_pipeline_transcriber``) consume THIS builder — the runtime path
and the evaluation path are the same graph by construction.

All model-loading collaborators are injectable for hermetic tests; the
defaults are the real loaders. If ``fallback_to_asr_mt`` is enabled but
the provider is explicitly disabled, the build FAILS loudly rather than
silently constructing a weaker stack than the config promises.
"""
from __future__ import annotations

from sawti.config import SawtiConfig
from sawti.engine import EngineManager
from sawti.fallback import FallbackHandler
from sawti.pipeline import Pipeline


def _load_m4t(device: str):
    import torch
    from transformers import (AutoProcessor, SeamlessM4Tv2ForSpeechToText)

    from sawti.engine_m4t import SeamlessM4TEngine

    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
        "facebook/seamless-m4t-v2-large")
    engine = SeamlessM4TEngine(processor=processor, model=model, device=device)
    return engine


def make_conservative_retry(engine):
    """Bind the ConservativeRetry seam to the SAME loaded M4T engine with
    conservative generation (beam search, deterministic)."""

    def conservative(chunk, target_lang):
        return engine.translate(chunk, target_lang, conservative=True)

    return conservative


def build_real_pipeline(
    config: SawtiConfig | None = None,
    on_decision=None,
    *,
    device: str | None = None,
    m4t_engine=None,           # injectable: SeamlessM4TEngine-compatible
    provider="real",           # "real" | None | AsrMtProvider instance
    gate=None,
    postprocessor=None,
    segmenter=None,
    rechunker=None,
) -> Pipeline:
    cfg = config or SawtiConfig()
    dev = device or cfg.s2tt.device

    from sawti.engine_m4t import SeamlessM4TEngine  # noqa: F401 (type hint)
    from sawti.postprocess_real import RealPostProcessor
    from sawti.quality_gate_balanced import BalancedQualityGate
    from sawti.rechunk import FixedSplitRechunker
    from sawti.segmenter_silero import RealSegmenter
    from sawti.vad import SileroVad

    engine = m4t_engine if m4t_engine is not None else _load_m4t(dev)
    # Resident construction (spec §3.3 default): the built engine is
    # adopted directly. True lazy/idle_unload production would pass an
    # engine_factory performing the load — the lifecycle is real either
    # way (see tests/test_engine_manager.py).
    engine_mgr = EngineManager(engine=engine, config=cfg.s2tt)

    # Provider policy: the config promises ASR+MT escalation or it doesn't.
    if cfg.quality_gate.fallback_to_asr_mt:
        if provider is None:
            raise ValueError(
                "quality_gate.fallback_to_asr_mt is enabled but the ASR+MT "
                "provider was explicitly disabled — refusing to build a "
                "weaker stack than the configuration promises"
            )
        if provider == "real":
            from sawti.providers import WhisperM4TProvider

            provider = WhisperM4TProvider(device=dev)
    else:
        provider = None

    fallback = FallbackHandler(
        engine=engine_mgr,
        gate=None,  # evaluated per-call below; handler shares the gate
        asr_mt=provider,
        rechunker=rechunker if rechunker is not None else FixedSplitRechunker(),
        conservative=make_conservative_retry(engine),
        config=cfg.quality_gate,
    )

    gate = gate if gate is not None else BalancedQualityGate(cfg.quality_gate)
    fallback.gate = gate  # the handler and the pipeline share one gate

    return Pipeline(
        segmenter=(
            segmenter if segmenter is not None
            else RealSegmenter(vad=SileroVad(), config=cfg.segmentation)
        ),
        engine=engine_mgr,
        gate=gate,
        postprocessor=(
            postprocessor if postprocessor is not None
            else RealPostProcessor(cfg.postprocess)
        ),
        fallback=fallback,
        on_decision=on_decision,
    )
