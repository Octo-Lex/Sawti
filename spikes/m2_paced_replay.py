"""Final isolation: real-time PACED replay of tee'd live audio.

Delivers the tee'd frames as 100 ms blocks at real-time pace (exactly
MicSource's delivery pattern) through the pre-warmed task7smoke graph,
printing per-chunk STAGE wall times via on_decision. If per-chunk cost
inflates ~5x under pacing alone, the live slowdown is a pacing/interplay
effect inside the pipeline, not the mic. Runs unattended.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import soundfile as sf

from sawti.build import build_real_pipeline
from sawti.config import load_config
from sawti.sources import AudioFrame

WAV = Path("outputs/live_session2.wav")
SR = 16000
BLOCK = 1600  # 100 ms — MicSource's frame size
LIMIT_S = 100.0  # first 100 s of the session audio


class PacedSource:
    """Yields 100 ms frames at real-time pace, like MicSource."""

    def __init__(self, wav: Path, limit_s: float) -> None:
        audio, _ = sf.read(str(wav), dtype="float32")
        self.audio = audio[: int(limit_s * SR)]
        self.t0 = None

    def iter_frames(self):
        emitted = 0.0
        for i in range(0, len(self.audio), BLOCK):
            if self.t0 is None:
                self.t0 = time.perf_counter()
            target = emitted + 0.1
            # sleep until real-time pace (skip catch-up sleeps)
            while time.perf_counter() - self.t0 < target - 0.005:
                time.sleep(0.002)
            block = self.audio[i:i + BLOCK]
            yield AudioFrame(
                audio=np.ascontiguousarray(block, dtype=np.float32),
                sample_rate=SR,
                timestamp_s=emitted,
            )
            emitted += len(block) / SR


def main() -> None:
    cfg = load_config("config/task7smoke.yaml")
    t0 = time.perf_counter()
    pipe = build_real_pipeline(cfg)
    provider = pipe.fallback.asr_mt
    provider._ensure_asr()
    provider._ensure_mt()
    print(f"[warmup] {time.perf_counter() - t0:.1f}s", flush=True)

    def on_decision(d):
        now = time.perf_counter()
        print(f"  chunk {d.chunk_id}: stage-path={d.fallback_path} "
              f"conf={d.result.confidence:.3f} "
              f"engine_ms={d.result.timing_ms.get('engine', 0):.0f} "
              f"asr_ms={d.result.timing_ms.get('asr_mt_ms', 0):.0f} "
              f"wall_end={now - t0:.1f}s", flush=True)

    pipe.on_decision = on_decision

    src = PacedSource(WAV, LIMIT_S)
    t1 = time.perf_counter()
    n = 0
    for seg in pipe.run(src, target_lang="eng"):
        n += 1
        print(f"OUT #{n} [{seg.start_time:6.2f}-{seg.end_time:6.2f}] "
              f"wall={time.perf_counter() - t1:6.1f}s | {seg.text[:50]!r}",
              flush=True)
    wall = time.perf_counter() - t1
    print(f"== {n} segments | {LIMIT_S:.0f}s paced audio | wall {wall:.1f}s "
          f"(overhead beyond real time: {wall - LIMIT_S:.1f}s) ==", flush=True)


if __name__ == "__main__":
    main()
