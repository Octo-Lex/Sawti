"""Long-horizon degradation probe: loop the tee'd live audio ~4x through
the pre-warmed task7smoke graph (same state as the failed live session),
printing per-segment wall latency so progressive slowdown becomes
visible. Run nvidia-smi sampling alongside to correlate.

Usage: python spikes/m2_degradation_probe.py [repeats]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from sawti.build import build_real_pipeline
from sawti.config import load_config
from sawti.sources import AudioFrame

WAV = Path("outputs/live_session2.wav")
SR = 16000


class LoopSource:
    """Repeats the wav's frames `repeats` times back-to-back."""

    def __init__(self, wav: Path, repeats: int) -> None:
        audio, _ = sf.read(str(wav), dtype="float32")
        self.audio = audio
        self.repeats = repeats

    def iter_frames(self):
        chunk = SR  # 1s frames, like FileSource
        for _ in range(self.repeats):
            for i in range(0, len(self.audio), chunk):
                block = self.audio[i:i + chunk]
                yield AudioFrame(
                    audio=np.ascontiguousarray(block, dtype=np.float32),
                    sample_rate=SR,
                    timestamp_s=i / SR,
                )


def main() -> None:
    repeats = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    cfg = load_config("config/task7smoke.yaml")

    t0 = time.perf_counter()
    pipe = build_real_pipeline(cfg)
    provider = pipe.fallback.asr_mt
    provider._ensure_asr()
    provider._ensure_mt()
    import torch

    print(f"[warmup] {time.perf_counter() - t0:.1f}s | reserved "
          f"{torch.cuda.memory_reserved() / 1e9:.1f} GB", flush=True)

    src = LoopSource(WAV, repeats)
    t1 = time.perf_counter()
    seg_i = 0
    prev_wall = t1
    for seg in pipe.run(src, target_lang="eng"):
        seg_i += 1
        now = time.perf_counter()
        dur = seg.end_time - seg.start_time
        proc = now - prev_wall
        prev_wall = now
        if seg_i % 5 == 0 or dur > 8:
            print(f"seg#{seg_i:3d} t+{now - t1:7.1f}s wall | chunk "
                  f"{dur:4.1f}s proc {proc:5.2f}s (x{proc / max(dur, 0.1):4.2f})"
                  f" | {seg.text[:40]!r}", flush=True)
    wall = time.perf_counter() - t1
    total_audio = repeats * sf.info(str(WAV)).duration
    print(f"== {seg_i} segments | {total_audio:.0f}s audio in {wall:.1f}s "
          f"(RTF {wall / total_audio:.2f}) ==", flush=True)


if __name__ == "__main__":
    main()
