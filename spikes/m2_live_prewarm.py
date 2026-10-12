"""Task 7 re-smoke #3: pre-warmed fallback lane + audio tee.

Hypothesis under test (2026-10-09, after re-smoke #2 died at 19.4s): the
live killer is the ~20s asr_mt cliff, dominated by LAZY model loading
(disk -> GPU, GIL-heavy) starving the PortAudio callback. Pre-warming
whisper-medium + M4T T2TT before the stream opens should turn an
exhaustion into a pure-inference stall the 30s queue can absorb.

No repo code is touched: the lazy provider caches are warmed through
their own _ensure_* seams, and the audio tee wraps the frame iterator.

Also tees all mic audio to outputs/live_session2.wav (rewritten every 50
frames, so even a hard kill loses <5s) for offline replay/diagnosis.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from sawti.audio_io import FileSource  # noqa: F401  (type parity)
from sawti.build import build_real_pipeline
from sawti.config import load_config
from sawti.mic_source import MicSource, TARGET_SR
from sawti.sources import AudioFrame

OUT_WAV = Path("outputs/live_session2.wav")


class TeeSource:
    """AudioSource wrapper: records every frame, passes it through."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.frames: list[np.ndarray] = []

    def iter_frames(self):
        n = 0
        for frame in self.inner.iter_frames():
            self.frames.append(frame.audio)
            n += 1
            if n % 50 == 0:
                self._flush()
            yield frame

    def _flush(self) -> None:
        if not self.frames:
            return
        OUT_WAV.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(OUT_WAV), np.concatenate(self.frames), TARGET_SR,
                 subtype="PCM_16")


def main() -> None:
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1
                      else "config/task7smoke.yaml")
    target = sys.argv[2] if len(sys.argv) > 2 else "eng"

    t0 = time.perf_counter()
    pipe = build_real_pipeline(cfg)
    built_s = time.perf_counter() - t0

    # Pre-warm the fallback lane through its own lazy seams.
    t1 = time.perf_counter()
    provider = pipe.fallback.asr_mt
    provider._ensure_asr()
    warm_asr_s = time.perf_counter() - t1
    t2 = time.perf_counter()
    provider._ensure_mt()
    warm_mt_s = time.perf_counter() - t2
    print(f"[warmup] graph built {built_s:.1f}s | whisper {warm_asr_s:.1f}s | "
          f"t2tt {warm_mt_s:.1f}s", flush=True)

    src = MicSource(device=None)
    tee = TeeSource(src)
    t3 = time.perf_counter()
    try:
        for seg in pipe.run(tee, target_lang=target):
            latency = time.perf_counter() - t3 - seg.end_time
            print(f"[{seg.start_time:6.2f}-{seg.end_time:6.2f}] "
                  f"(lag {latency:+5.1f}s) {seg.text}", flush=True)
            t3 = time.perf_counter()  # rough per-segment latency base
    except Exception as e:
        tee._flush()
        print(f"SESSION ERROR: {type(e).__name__}: {e}", flush=True)
        raise
    finally:
        src.close()
        tee._flush()
        s = src.stats()
        print(f"[capture] {s['captured_seconds']}s captured / "
              f"{s['emitted_seconds']}s emitted | queue high-water "
              f"{s['queue_high_water']} | input overflows "
              f"{s['input_overflow_count']} | queue overflows "
              f"{s['queue_overflow_count']}", flush=True)


if __name__ == "__main__":
    main()
