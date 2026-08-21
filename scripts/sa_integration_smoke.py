"""Production-path smoke (reviewer-released): load models/sa_whisper_v1
through the ACTUAL Sawti builder path — build_real_pipeline(provider="sa")
with the full recovery stack (Silero VAD segmenter -> M4T engine ->
balanced gate -> fallback) — and prove real speech traverses it.

Feeds a SADA VALIDATION clip (never test-split audio) through both an
Arabic and an English target session. Success = segments emitted with
non-empty target-language text via the real graph; no stubs anywhere.

OPERATOR:
  uv run python scripts/sa_integration_smoke.py data/sada_training/val/<clip>.wav
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf


class _FileFrames:
    """Minimal AudioSource: one AudioFrame per 30s-or-less block of the
    file (the pipeline's segmenter consumes frames and re-segments)."""

    def __init__(self, wav: str, block_s: float = 10.0) -> None:
        audio, sr = sf.read(wav, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        assert sr == 16000, f"expected 16 kHz, got {sr}"
        self.audio = audio
        self.block = int(block_s * sr)

    def iter_frames(self):
        from sawti.sources import AudioFrame

        for lo in range(0, len(self.audio), self.block):
            chunk = self.audio[lo:lo + self.block]
            yield AudioFrame(audio=chunk, sample_rate=16000,
                             timestamp_s=lo / 16000.0)


def main() -> None:
    from sawti.env import load_env
    load_env(override=True)

    wav = sys.argv[1] if len(sys.argv) > 1 else None
    if not wav or not Path(wav).exists():
        raise SystemExit(f"usage: {sys.argv[0]} <validation-clip.wav>")

    from sawti.build import build_real_pipeline
    from sawti.config import SawtiConfig
    from sawti.types import AudioChunk

    decisions: list = []
    pipe = build_real_pipeline(
        SawtiConfig(),
        on_decision=lambda d: decisions.append(d),
        provider="sa",
    )
    for target in ("ara", "eng"):
        segs = list(pipe.run(_FileFrames(wav), target))
        print(f"--- target {target}: {len(segs)} segment(s) ---")
        for s in segs:
            print(f"  [{s.start_time:7.2f}-{s.end_time:7.2f}] "
                  f"{'LOW-CONF ' if s.low_confidence else ''}{s.text}")
        assert segs, f"no segments emitted for target {target}"
        assert any(s.text.strip() for s in segs), "all segments empty"
    # on_decision receives GateDecision objects; fallback_path is
    # None | 'retry' | 'rechunk' | 'asr_mt'.
    fell_back = sum(1 for d in decisions if d.fallback_path is not None)
    print(f"--- gate decisions: {len(decisions)} ({fell_back} fell back) ---")

    # The fallback seat is lazy: if no decision fell back on this clip, the
    # SA provider's ASR lane never fired. Exercise the BUILDER-CONSTRUCTED
    # provider instance directly on the same real clip so the SA lane
    # itself is proven on the production path, whatever the gate did.
    provider = pipe.fallback.asr_mt
    audio, sr = sf.read(wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    chunk = AudioChunk(id="smoke", audio=audio, sample_rate=sr,
                       start_time=0.0, end_time=len(audio) / sr)
    for target in ("ara", "eng"):
        res = provider.asr_mt(chunk, target)
        print(f"--- SA lane target {target}: {res.raw_text!r} "
              f"({res.timing_ms['path']}) ---")
        assert res.raw_text.strip(), f"SA lane empty for {target}"
        assert res.source_lang_guess == "ara"
    print("SMOKE PASS: real speech traversed the full production graph "
          "with the SA provider.")


if __name__ == "__main__":
    main()
