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


def make_restart_redecode(engine_mgr):
    """Bind the degenerate-restart seam (2026-10-09) to the EngineManager's
    CURRENT engine instance: greedy re-decode with n-gram repetition
    blocked (engine_m4t.RESTART_NO_REPEAT_NGRAM). Call-time resolution,
    same engine, no second model."""

    def restart(chunk, target_lang):
        return engine_mgr.engine.translate(chunk, target_lang, restart=True)

    return restart


def _check_vram_budget(budget_gb: float) -> None:
    """Fail BEFORE the session (release-rule pattern) when torch has
    reserved more VRAM than the configured budget. CPU-only hosts and
    CUDA-less introspection skip the check — it guards GPU thrash, not
    correctness."""
    try:
        import torch

        if not torch.cuda.is_available():
            return
        reserved = torch.cuda.memory_reserved() / 1e9
    except Exception:
        return
    if reserved > budget_gb:
        raise ValueError(
            f"VRAM budget exceeded: {reserved:.1f} GB reserved > "
            f"{budget_gb} GB budget — the configured stack does not fit; "
            f"refusing to start a session that would thrash"
        )


def build_real_pipeline(
    config: SawtiConfig | None = None,
    on_decision=None,
    *,
    engine: str = "m4t",       # primary engine kind: "m4t" | "sawti-sa"
    device: str | None = None,
    m4t_engine=None,           # injectable: adopts a BUILT engine (resident only)
    m4t_factory=None,          # injectable: engine factory (honors load_policy)
    sa_engine=None,            # injectable: a BUILT SaudiWhisperEngine (resident only)
    provider="real",           # "real" | "sa" | None | AsrMtProvider instance
    gate=None,
    postprocessor=None,
    segmenter=None,
    rechunker=None,
    clock=None,                # injectable monotonic clock (idle_unload tests)
) -> Pipeline:
    cfg = config or SawtiConfig()
    dev = device or cfg.s2tt.device

    if engine not in ("m4t", "sawti-sa"):
        raise ValueError(f"unsupported engine kind {engine!r} "
                         f"(expected 'm4t' or 'sawti-sa')")
    if engine == "sawti-sa" and m4t_engine is not None:
        raise ValueError("m4t_engine injection is only valid with "
                         "engine='m4t'; use sa_engine for 'sawti-sa'")
    if engine == "m4t" and sa_engine is not None:
        raise ValueError("sa_engine injection is only valid with "
                         "engine='sawti-sa'")

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
    # EngineManager the engine factory, so load_policy controls model loading
    # (resident=eager, lazy=first translate, idle_unload=release+rebuild).
    # Adopting a prebuilt engine is constrained to resident semantics.
    if engine == "sawti-sa":
        # SA primary mode (spec SA 4.2): the merged Saudi Whisper export
        # in the PRIMARY seat; the fallback lane stays a DIFFERENT model
        # (stock whisper-medium provider) so escalation is a genuine
        # second opinion, never a re-run of the same decoder.
        if sa_engine is not None:
            if cfg.s2tt.load_policy != "resident":
                raise ValueError(
                    "sa_engine injection adopts a BUILT engine — only valid "
                    "with load_policy='resident'; use a factory for "
                    "lazy/idle_unload semantics"
                )
            engine_mgr = EngineManager(engine=sa_engine, config=cfg.s2tt,
                                       clock=clock)
        else:
            def _factory():
                from sawti.engine_sa import load_sa_engine

                return load_sa_engine(dev)

            engine_mgr = EngineManager(engine_factory=_factory,
                                       config=cfg.s2tt, clock=clock)
    elif m4t_engine is not None:
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
        elif provider == "sa":
            # SA integration (locked shape): merged Saudi Whisper export in
            # the ASR seat + the existing M4T MT lane. Lazy-loading — no
            # model weights are touched at build time.
            from sawti.providers_sa import SaudiWhisperM4TProvider

            provider = SaudiWhisperM4TProvider(device=dev)
    else:
        provider = None

    fallback = FallbackHandler(
        engine=engine_mgr,
        gate=None,  # evaluated per-call below; handler shares the gate
        asr_mt=provider,
        rechunker=rechunker if rechunker is not None else FixedSplitRechunker(),
        conservative=make_conservative_retry(engine_mgr),
        restart=make_restart_redecode(engine_mgr),
        config=cfg.quality_gate,
    )

    gate = gate if gate is not None else BalancedQualityGate(cfg.quality_gate)
    fallback.gate = gate  # the handler and the pipeline share one gate

    # Live-session hardening (2026-10-09): optionally bring the fallback
    # lane's models up at BUILD time (no mid-session disk->GPU stall) and
    # enforce the VRAM budget before the session starts. Guarded by
    # hasattr so injected fakes are never forced through real loaders.
    if cfg.s2tt.prewarm_fallback and provider is not None:
        if hasattr(provider, "_ensure_asr"):
            provider._ensure_asr()
        if hasattr(provider, "_ensure_mt"):
            provider._ensure_mt()
    if cfg.s2tt.vram_budget_gb is not None:
        _check_vram_budget(cfg.s2tt.vram_budget_gb)

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
