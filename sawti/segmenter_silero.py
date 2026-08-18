"""Real segmenter implementing the close-decision policy (spec §2.4) —
Commit 4 corrective pass: source-time-true chunk semantics.

STRONG invariant: every emitted chunk's audio is the LITERAL source
waveform for [start_time, end_time] — assertable as
``np.array_equal(chunk.audio, source[int(start*sr):int(end*sr)])``.

- An open chunk buffers the contiguous wall-clock stream; VAD decides
  boundaries and never destructively edits audio. Internal silence is
  retained; the pause-threshold silence is lookahead (consumed, not
  emitted). Emitted span: first-speech-start .. last-speech-end.
- Overlap carries the previous EMITTED chunk's tail ACROSS the real gap:
  the next chunk's audio is tail | actual intervening silence | new
  speech, and its start_time is the tail's true source time.
  ``overlap_from_prev_s`` counts ONLY the duplicated tail, never the
  bridge silence. Carry budget: if carried + gap would exceed
  max_chunk_duration_s, the carry is DROPPED (fresh open at speech
  start) — overlap must never force a chunk beyond max span, and a
  dropped carry beats falsified timestamps.
- max_chunk_duration_s is enforced on the full wall-clock span INCLUDING
  retained silence: a max-span close fires while the current window is
  silent, not only on speech.
- sample_rate propagates from input frames.
- min_speech_ms gates every emission on total SPEECH content (the tail's
  speech belongs to the previous chunk and does not count).
- min_gap_ms: inter-chunk silence must reach it before a pause-close may
  split; shorter silence stays internal.
- Close policy: pause-close requires silence >= pause_threshold_ms AND
  silence >= min_gap_ms AND speech-span >= min_chunk_duration_ms;
  max-span close (speech or silent window) at max_chunk_duration_s; EOF
  flush. Discarded blips merge their buffered audio into the gap stream
  so bridge accounting stays source-exact. Ordering/ids deterministic.
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

        # Overlap-carry machinery (between emitted chunks).
        pending_tail: np.ndarray | None = None   # prev chunk's tail samples
        prev_end: float | None = None            # prev chunk's end_time
        gap_parts: list[np.ndarray] = []         # source samples since prev_end
        gap_samples = 0

        # Open-chunk state.
        opened = False
        timeline_start = 0.0      # true source start of the chunk's audio
        overlap_s = 0.0           # duplicated previous-tail seconds (this chunk)
        own_start = 0.0           # first-speech start (content gates)
        last_speech_end = 0.0
        parts: list[np.ndarray] = []
        buffered = 0              # samples buffered since timeline_start
        speech_end_pos = 0        # buffer position of last speech END
        speech_samples = 0        # total speech samples (min_speech gate)
        silence_ms = 0.0

        def _merge_into_gap() -> None:
            """A discarded chunk's buffered audio is intervening SOURCE
            audio — merge it so bridge accounting stays exact."""
            nonlocal gap_parts, gap_samples
            if parts:
                gap_parts.append(np.concatenate(parts))
                gap_samples += buffered

        def _drop_carry_if_over_budget() -> None:
            nonlocal pending_tail, gap_parts, gap_samples
            if pending_tail is not None and (
                len(pending_tail) + gap_samples
            ) / sr > cfg.max_chunk_duration_s:
                pending_tail = None
                gap_parts = []
                gap_samples = 0

        def reset_open() -> None:
            nonlocal opened, parts, buffered, speech_end_pos
            nonlocal speech_samples, silence_ms, overlap_s
            opened = False
            parts = []
            buffered = 0
            speech_end_pos = 0
            speech_samples = 0
            silence_ms = 0.0
            overlap_s = 0.0

        def emit(emit_end: float) -> AudioChunk | None:
            """Build the chunk [timeline_start, emit_end]; None if gates
            fail (blip — buffered audio merges into the gap stream)."""
            nonlocal pending_tail, prev_end, gap_parts, gap_samples
            if speech_samples / sr * 1000.0 < cfg.min_speech_ms:
                _merge_into_gap()
                _drop_carry_if_over_budget()
                reset_open()
                return None
            full = np.concatenate(parts) if parts else np.zeros(0, np.float32)
            own_audio = full[:speech_end_pos]
            residue = full[speech_end_pos:]  # lookahead after last speech
            chunk = AudioChunk(
                id=f"c{self._counter}",
                audio=np.ascontiguousarray(own_audio, dtype=np.float32),
                sample_rate=sr,
                start_time=timeline_start,
                end_time=emit_end,
                overlap_from_prev_s=overlap_s,
                meta={},
            )
            self._counter += 1
            # Stage the tail for the next chunk; seed the gap stream with
            # the residue (source samples immediately after emit_end).
            if cfg.overlap_ms > 0 and len(own_audio) > 0:
                carry = min(int(cfg.overlap_ms / 1000.0 * sr), len(own_audio))
                pending_tail = own_audio[-carry:].copy() if carry > 0 else None
            else:
                pending_tail = None
            prev_end = emit_end
            gap_parts = [residue] if len(residue) else []
            gap_samples = len(residue)
            _drop_carry_if_over_budget()
            reset_open()
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
                        # Open, optionally carrying the previous tail across
                        # the REAL bridged gap (tail | gap | this speech).
                        if pending_tail is not None:
                            bridge = (
                                np.concatenate(gap_parts)
                                if gap_parts else np.zeros(0, np.float32)
                            )
                            parts = [pending_tail, bridge]
                            buffered = len(pending_tail) + gap_samples
                            timeline_start = prev_end - len(pending_tail) / sr
                            overlap_s = len(pending_tail) / sr
                        else:
                            parts = []
                            buffered = 0
                            timeline_start = sub_start
                            overlap_s = 0.0
                        pending_tail = None
                        gap_parts = []
                        gap_samples = 0
                        opened = True
                        own_start = sub_start
                        last_speech_end = sub_start
                        speech_end_pos = 0
                        speech_samples = 0
                        silence_ms = 0.0
                    parts.append(sub[: b - a] if b - a < window else sub)
                    buffered += b - a
                    speech_end_pos = buffered
                    speech_samples += b - a
                    last_speech_end = sub_end
                    silence_ms = 0.0

                    span_ms = (sub_end - timeline_start) * 1000.0
                    if span_ms >= cfg.max_chunk_duration_s * 1000.0:
                        if (last_speech_end - own_start) * 1000.0 >= cfg.min_chunk_duration_ms:
                            chunk = emit(sub_end)
                            if chunk is not None:
                                yield chunk
                        else:
                            _merge_into_gap()
                            _drop_carry_if_over_budget()
                            reset_open()
                else:
                    if opened:
                        parts.append(sub[: b - a] if b - a < window else sub)
                        buffered += b - a
                        silence_ms += sub_ms
                        span_ms = (sub_end - timeline_start) * 1000.0
                        content_ms = (last_speech_end - own_start) * 1000.0
                        if span_ms >= cfg.max_chunk_duration_s * 1000.0:
                            # Max span enforced during retained silence too.
                            if content_ms >= cfg.min_chunk_duration_ms:
                                chunk = emit(last_speech_end)
                                if chunk is not None:
                                    yield chunk
                            else:
                                _merge_into_gap()
                                _drop_carry_if_over_budget()
                                reset_open()
                        elif (
                            silence_ms >= cfg.pause_threshold_ms
                            and silence_ms >= cfg.min_gap_ms
                            and content_ms >= cfg.min_chunk_duration_ms
                        ):
                            chunk = emit(last_speech_end)
                            if chunk is not None:
                                yield chunk
                    elif pending_tail is not None:
                        # Gap growing between chunks: buffer the source
                        # silence while a carry remains budgeted.
                        gap_parts.append(
                            sub[: b - a] if b - a < window else sub
                        )
                        gap_samples += b - a
                        _drop_carry_if_over_budget()

        if opened:
            if (last_speech_end - own_start) * 1000.0 >= cfg.min_chunk_duration_ms:
                chunk = emit(last_speech_end)
                if chunk is not None:
                    yield chunk
            else:
                _merge_into_gap()
                _drop_carry_if_over_budget()
                reset_open()
