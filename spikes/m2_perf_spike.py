"""M2 performance spike (2026-10-09): can this graph run at RTF < 1?

Matrix driven externally, one combination per process (clean VRAM):

    python spikes/m2_perf_spike.py --dtype fp16 --threshold 0.35 \
        --wav outputs/mic_recording.wav

Measures, on REAL mic audio through the UNCHANGED production graph:
- wall RTF and per-chunk engine latency (greedy + any beam retries)
- which fallback paths fired (retry / rechunk / asr_mt) and their cost
- the primary-pass confidence distribution feeding the threshold decision

Only knobs touched, both injection seams that already exist:
- M4T load dtype (fp16 vs fp32) via build_real_pipeline(m4t_factory=...)
- quality_gate.confidence_threshold via config

Emits ONE JSON summary line on stdout for aggregation.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path


def load_m4t_dtype(device: str, dtype: str):
    import torch
    from transformers import (AutoProcessor,
                              SeamlessM4Tv2ForSpeechToText)

    from sawti.engine_m4t import SeamlessM4TEngine

    model_id = "facebook/seamless-m4t-v2-large"
    processor = AutoProcessor.from_pretrained(model_id)
    kwargs = {"dtype": torch.float16} if dtype == "fp16" else {}
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(model_id, **kwargs)
    return SeamlessM4TEngine(processor=processor, model=model, device=device)


def main() -> None:
    import soundfile as sf

    from sawti.audio_io import FileSource
    from sawti.build import build_real_pipeline
    from sawti.config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["fp32", "fp16"], required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--wav", default="outputs/mic_recording.wav")
    ap.add_argument("--target", default="eng")
    args = ap.parse_args()

    cfg = load_config("config/default.yaml")
    cfg.quality_gate.confidence_threshold = args.threshold

    decisions = []

    def on_decision(d):
        decisions.append({
            "chunk": d.chunk_id,
            "path": d.fallback_path,
            "accepted": d.accepted,
            "conf": round(d.result.confidence, 3),
            "engine_ms": round(d.result.timing_ms.get("engine", 0.0), 1),
            "asr_mt_ms": round(d.result.timing_ms.get("asr_mt_ms", 0.0), 1),
        })

    t0 = time.perf_counter()
    pipe = build_real_pipeline(
        cfg, on_decision=on_decision,
        m4t_factory=lambda: load_m4t_dtype(cfg.s2tt.device, args.dtype),
    )
    load_s = time.perf_counter() - t0

    audio_s = sf.info(args.wav).duration
    src = FileSource(Path(args.wav), frame_samples=16000)

    t1 = time.perf_counter()
    texts = [seg.text for seg in pipe.run(src, target_lang=args.target)]
    wall_s = time.perf_counter() - t1

    n_retry = sum(1 for d in decisions if d["path"] == "retry")
    n_rechunk = sum(1 for d in decisions if d["path"] == "rechunk")
    n_asr = sum(1 for d in decisions if d["path"] == "asr_mt")
    n_exhausted = sum(1 for d in decisions if d["path"] == "exhausted")
    n_primary = sum(1 for d in decisions if d["path"] is None)

    print(json.dumps({
        "dtype": args.dtype,
        "threshold": args.threshold,
        "wav": args.wav,
        "audio_s": round(audio_s, 2),
        "load_s": round(load_s, 1),
        "wall_s": round(wall_s, 2),
        "rtf": round(wall_s / audio_s, 3),
        "chunks": len(decisions),
        "primary_pass": n_primary,
        "retry": n_retry,
        "rechunk": n_rechunk,
        "asr_mt": n_asr,
        "exhausted": n_exhausted,
        "engine_ms_sum": round(sum(d["engine_ms"] for d in decisions), 1),
        "asr_mt_ms_sum": round(sum(d["asr_mt_ms"] for d in decisions), 1),
        "decisions": decisions,
        "texts": texts,
    }, ensure_ascii=False))
    gc.collect()


if __name__ == "__main__":
    main()
