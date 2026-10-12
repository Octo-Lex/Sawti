"""Isolate the live-session slowdown: replay tee'd audio through the
EXACT pre-smoke state — task7smoke config + pre-warmed fallback lane.

If this reproduces the ~4.7x slowdown vs the cold replay (RTF 0.39),
the culprit is the pre-warmed VRAM residency, not live capture.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from sawti.audio_io import FileSource
from sawti.build import build_real_pipeline
from sawti.config import load_config


def main() -> None:
    cfg = load_config("config/task7smoke.yaml")

    t0 = time.perf_counter()
    pipe = build_real_pipeline(cfg)
    built_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    provider = pipe.fallback.asr_mt
    provider._ensure_asr()
    warm_asr_s = time.perf_counter() - t1
    t2 = time.perf_counter()
    provider._ensure_mt()
    warm_mt_s = time.perf_counter() - t2

    import torch

    vram_gb = torch.cuda.memory_reserved() / 1e9
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[warmup] built {built_s:.1f}s | whisper {warm_asr_s:.1f}s | "
          f"t2tt {warm_mt_s:.1f}s | torch reserved {vram_gb:.1f}/"
          f"{total_gb:.1f} GB", flush=True)

    src = FileSource(Path("outputs/live_session2.wav"), frame_samples=16000)
    t3 = time.perf_counter()
    n = 0
    for seg in pipe.run(src, target_lang="eng"):
        n += 1
        now = time.perf_counter()
        print(f"[{seg.start_time:6.2f}-{seg.end_time:6.2f}] "
              f"chunk {(seg.end_time - seg.start_time):4.1f}s "
              f"| seg#{n} t+{now - t3:6.1f}s | {seg.text[:60]!r}", flush=True)
    wall = time.perf_counter() - t3
    print(f"== {n} segments | 124.0s audio in {wall:.2f}s wall "
          f"(RTF {wall / 124.0:.2f}) ==", flush=True)


if __name__ == "__main__":
    main()
