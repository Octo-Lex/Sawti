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


def make_conservative_retry(engine_mgr):
    """Bind the ConservativeRetry seam to the EngineManager's CURRENT
    engine instance (same loaded model, conservative generation: beam
    search, deterministic). Resolved at CALL time so lazy/idle_unload
    rebuilds are honored."""

    def conservative(chunk, target_lang):
        return engine_mgr.engine.translate(
            chunk, target_lang, conservative=True
        )

    return conservative


def build_real_pipeline(
    config: SawtiConfig | None = None,
    on_decision=None,
    *,
    device: str | None = None,
    m4t_engine=None,           # injectable: adopts a BUILT engine (resident only)
    m4t_factory=None,          # injectable: engine factory (honors load_policy)
    provider="real",           # "real" | None | AsrMtProvider instance
    gate=None,
    postprocessor=None,
    segmenter=None,
    rechunker=None,
    clock=None,                # injectable monotonic clock (idle_unload tests)
) -> Pipeline:
    cfg = config or SawtiConfig()
    dev = device or cfg.s2tt.device

    # Config truthfulness: only the balanced gate policy exists in M1.
    # Reject anything else loudly rather than silently ignoring the knob.
    if cfg.quality_gate.policy != "balanced":
        raise ValueError(
            f"unsupported quality_gate.policy: {cfg.quality_gate.policy!r} "
            f"(only 'balanced' is implemented in M1)"
        )

    from sawti.engine_m4t import SeamlessM4TEngine  # noqa: F401 (type hint)
    from sawti.postprocess_real import RealPostProcessor
    from sawti.quality_gate_balanced import BalancedQualityGate
    from sawti.rechunk import FixedSplitRechunker
    from sawti.segmenter_silero import RealSegmenter
    from sawti.vad import SileroVad

    # Lifecycle is REAL on the production path: the default builder hands
    # EngineManager an M4T factory, so load_policy controls model loading
    # (resident=eager, lazy=first translate, idle_unload=release+rebuild).
    # Adopting a prebuilt engine is constrained to resident semantics.
    if m4t_engine is not None:
        if cfg.s2tt.load_policy != "resident":
            raise ValueError(
                "m4t_engine injection adopts a BUILT engine — only valid "
                "with load_policy='resident'; use m4t_factory for "
                "lazy/idle_unload semantics"
            )
        engine_mgr = EngineManager(engine=m4t_engine, config=cfg.s2tt,
                                   clock=clock)
    else:
        factory = m4t_factory if m4t_factory is not None else (
            lambda: _load_m4t(dev)
        )
        engine_mgr = EngineManager(engine_factory=factory, config=cfg.s2tt,
                                   clock=clock)

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
        conservative=make_conservative_retry(engine_mgr),
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
