"""Real segmenter implementing the close-decision policy (spec §2.4) —
Commit 4: wall-clock-true chunk semantics.

Contract (invariant-tested):

- ``AudioChunk.audio`` and ``[start_time, end_time]`` describe the SAME
  audio interval: sample counts are derived from the same offsets as the
  timestamps, so ``len(audio) == (end_time - start_time) * sample_rate``
  exactly. VAD decides boundaries; it NEVER destructively edits the
  waveform — internal silence inside an open chunk is preserved in the
  emitted audio. The pause-threshold silence that triggers a close is
  lookahead (consumed, not emitted): chunks span first-speech-start ..
  last-speech-end.
- ``sample_rate`` propagates from the input frames.
- ``overlap_ms``: the chunk AFTER an emitted one physically begins with
  that chunk's tail audio (``overlap_from_prev_s`` = seconds actually
  carried; a shorter previous chunk carries less). Note the splice
  semantics: the leading overlap re-covers the prior chunk's closing
  words for downstream dedup — the chunk's OWN span (excluding overlap)
  remains strictly wall-clock-true.
- ``min_speech_ms``: total SPEECH content required for any emission —
  sub-threshold blips are dropped even when min_chunk_duration_ms is 0.
- ``min_gap_ms``: inter-chunk silence must reach this before a
  pause-close may split; silence shorter than this stays internal to the
  open chunk (observable when configured above pause_threshold_ms).
- Close policy (spec §2.4): pause-close requires silence >=
  pause_threshold_ms AND silence >= min_gap_ms AND span >=
  min_chunk_duration_ms; force-close fires at max_chunk_duration_s
  mid-speech; EOF flushes the open chunk. Emission always additionally
  requires speech >= min_speech_ms. Ordering/ids deterministic.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from sawti.config import SegmentationConfig
from sawti.sources import AudioFrame
from sawti.types import AudioChunk
from sawti.vad import VAD


class RealSegmenter:
    """VAD + pause + max-duration + min-duration segmenter (wall-clock)."""

    def __init__(
        self,
        vad: VAD,
        config: SegmentationConfig | None = None,
    ) -> None:
        self.vad = vad
        self.config = config or SegmentationConfig()
        self._counter = 0

    def process(self, frames: Iterable[AudioFrame]) -> Iterable[AudioChunk]:
        cfg = self.config
        sr: int | None = None
        window = 512  # Silero contract: 512 @ 16 kHz, 256 @ 8 kHz
        pending_tail: np.ndarray | None = None  # overlap carry for next open

        opened = False
        own_start = 0.0            # wall-clock first-speech start
        last_speech_end = 0.0
        parts: list[np.ndarray] = []   # contiguous wall-clock since open
        buffered = 0                # samples buffered since open (position)
        speech_end_pos = 0          # buffer position of last speech END
        speech_samples = 0          # total speech samples (min_speech gate)
        silence_ms = 0.0

        def reset() -> None:
            nonlocal opened, parts, buffered, speech_end_pos
            nonlocal speech_samples, silence_ms
            opened = False
            parts = []
            buffered = 0
            speech_end_pos = 0
            speech_samples = 0
            silence_ms = 0.0

        def emit(emit_end: float) -> AudioChunk | None:
            """Build the chunk [own_start, emit_end]; None if gates fail.
            On success, stage this chunk's tail as the next overlap carry."""
            nonlocal pending_tail
            speech_ms = speech_samples / sr * 1000.0
            if speech_ms < cfg.min_speech_ms:
                return None  # blip: dropped, pending_tail untouched
            own_audio = (
                np.concatenate(parts)[:speech_end_pos]
                if parts else np.zeros(0, dtype=np.float32)
            )
            tail = None
            if cfg.overlap_ms > 0 and len(own_audio) > 0:
                carry = min(int(cfg.overlap_ms / 1000.0 * sr), len(own_audio))
                if carry > 0:
                    tail = own_audio[-carry:].copy()
            audio = (
                np.concatenate([pending_tail, own_audio])
                if pending_tail is not None and len(pending_tail) > 0
                else own_audio
            )
            overlap_s = (len(pending_tail) / sr) if pending_tail is not None else 0.0
            chunk = AudioChunk(
                id=f"c{self._counter}",
                audio=np.ascontiguousarray(audio, dtype=np.float32),
                sample_rate=sr,
                start_time=own_start - overlap_s,
                end_time=emit_end,
                overlap_from_prev_s=overlap_s,
                meta={},
            )
            self._counter += 1
            pending_tail = tail
            return chunk

        for frame in frames:
            if sr is None:
                sr = frame.sample_rate
                window = 512 if sr == 16000 else 256
            n = len(frame.audio)
            if n == 0:
                continue
            n_full = (n + window - 1) // window
            for wi in range(n_full):
                a = wi * window
                b = min(a + window, n)
                sub = frame.audio[a:b]
                if len(sub) < window:
                    sub = np.pad(sub, (0, window - len(sub)))
                sub_start = frame.timestamp_s + a / sr
                sub_end = frame.timestamp_s + b / sr
                sub_ms = (b - a) / sr * 1000.0

                vr = self.vad.prob(sub, sr)
                if vr.is_speech:
                    if not opened:
                        opened = True
                        own_start = sub_start
                        parts = []
                        buffered = 0
                        speech_end_pos = 0
                        speech_samples = 0
                        silence_ms = 0.0
                        # pending_tail (overlap carry) deliberately survives
                        # opens/closes — it belongs to the next EMITTED chunk.
                    parts.append(sub[: b - a] if b - a < window else sub)
                    buffered += b - a
                    speech_end_pos = buffered       # buffer POSITION, not a
                    speech_samples += b - a         # speech-sample count
                    last_speech_end = sub_end
                    silence_ms = 0.0

                    span_ms = (sub_end - own_start) * 1000.0
                    if span_ms >= cfg.max_chunk_duration_s * 1000.0:
                        if (sub_end - own_start) * 1000.0 >= cfg.min_chunk_duration_ms:
                            chunk = emit(sub_end)
                            if chunk is not None:
                                yield chunk
                        reset()
                else:
                    if opened:
                        parts.append(sub[: b - a] if b - a < window else sub)
                        buffered += b - a
                        silence_ms += sub_ms
                        content_ms = (last_speech_end - own_start) * 1000.0
                        if (
                            silence_ms >= cfg.pause_threshold_ms
                            and silence_ms >= cfg.min_gap_ms
                            and content_ms >= cfg.min_chunk_duration_ms
                        ):
                            chunk = emit(last_speech_end)
                            if chunk is not None:
                                yield chunk
                            reset()

        if opened:
            content_ms = (last_speech_end - own_start) * 1000.0
            if content_ms >= cfg.min_chunk_duration_ms:
                chunk = emit(last_speech_end)
                if chunk is not None:
                    yield chunk
            reset()
