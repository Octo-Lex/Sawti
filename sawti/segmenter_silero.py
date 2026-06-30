"""Real segmenter implementing the close-decision policy (spec §2.4).

Takes an injectable VAD so the policy is unit-testable with FakeVad. Uses
the SegmentationConfig (frozen) for thresholds. Emits AudioChunk (frozen type).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from sawti.config import SegmentationConfig
from sawti.sources import AudioFrame
from sawti.types import AudioChunk
from sawti.vad import VAD


class RealSegmenter:
    """VAD + pause + max-duration + min-duration segmenter."""

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
        frame_dur_ms = 0.0  # learned from first frame
        buf_audio: list[np.ndarray] = []
        buf_start: float | None = None
        buf_end: float = 0.0
        last_speech_end: float = 0.0  # content boundary (excludes trailing silence)
        silence_ms = 0.0
        open_chunk = False

        for frame in frames:
            if frame_dur_ms == 0.0 and len(frame.audio) > 0:
                frame_dur_ms = len(frame.audio) / frame.sample_rate * 1000.0
            frame_end = frame.timestamp_s + len(frame.audio) / frame.sample_rate

            vr = self._frame_speech_prob(frame.audio, frame.sample_rate)

            if vr.is_speech:
                if not open_chunk:
                    buf_start = frame.timestamp_s
                    open_chunk = True
                    silence_ms = 0.0
                buf_audio.append(frame.audio)
                buf_end = frame_end
                last_speech_end = frame_end
                silence_ms = 0.0

                # Force-close on max duration fires even mid-speech.
                content_ms = (last_speech_end - buf_start) * 1000.0
                if content_ms >= cfg.max_chunk_duration_s * 1000.0:
                    if content_ms >= cfg.min_chunk_duration_ms:
                        yield self._emit(buf_audio, buf_start, last_speech_end)
                    buf_audio, open_chunk, buf_start = [], False, None
                    silence_ms = 0.0
            else:
                if open_chunk:
                    silence_ms += frame_dur_ms
                    buf_audio.append(frame.audio)  # include trailing silence in buffer
                    buf_end = frame_end
                    content_ms = (last_speech_end - buf_start) * 1000.0
                    total_ms = (buf_end - buf_start) * 1000.0

                    # Force-close on max duration regardless of silence.
                    if total_ms >= cfg.max_chunk_duration_s * 1000.0:
                        if content_ms >= cfg.min_chunk_duration_ms:
                            yield self._emit(buf_audio, buf_start, last_speech_end)
                        buf_audio, open_chunk, buf_start = [], False, None
                        silence_ms = 0.0
                    # Close on pause threshold once min (speech) duration met.
                    elif silence_ms >= cfg.pause_threshold_ms and \
                            content_ms >= cfg.min_chunk_duration_ms:
                        yield self._emit(buf_audio, buf_start, last_speech_end)
                        buf_audio, open_chunk, buf_start = [], False, None
                        silence_ms = 0.0

        # Flush any open buffer at end of stream (gated on speech content).
        if open_chunk and buf_audio and buf_start is not None:
            content_ms = (last_speech_end - buf_start) * 1000.0
            if content_ms >= cfg.min_chunk_duration_ms:
                yield self._emit(buf_audio, buf_start, last_speech_end)

    def _frame_speech_prob(self, audio: np.ndarray, sample_rate: int):
        """Compute a frame-level VAD verdict by aggregating over Silero's
        required sub-window size (512 samples at 16kHz, 256 at 8kHz).

        Silero VAD raises if given anything other than exactly 512/256
        samples, so a frame larger than that must be windowed. We take the
        max probability across sub-windows and treat the frame as speech if
        that max exceeds the VAD's threshold (any-speech semantics — a frame
        containing some speech counts as speech).
        """
        window = 512 if sample_rate == 16000 else 256
        n = len(audio)
        if n == 0:
            from sawti.vad import VadResult

            return VadResult(probability=0.0, is_speech=False)
        if n <= window:
            # Pad up to one window so Silero gets the size it expects.
            padded = np.pad(audio, (0, window - n))
            return self.vad.prob(padded, sample_rate)
        # Aggregate over full sub-windows (drop the trailing partial window).
        best_p = 0.0
        best_is_speech = False
        for i in range(0, n - window + 1, window):
            sub = audio[i : i + window]
            r = self.vad.prob(sub, sample_rate)
            if r.probability > best_p:
                best_p = r.probability
                best_is_speech = r.is_speech
        from sawti.vad import VadResult

        return VadResult(probability=best_p, is_speech=best_is_speech)

    def _emit(
        self, buf_audio: list[np.ndarray], start: float, end: float
    ) -> AudioChunk:
        chunk_id = f"c{self._counter}"
        self._counter += 1
        audio = np.concatenate(buf_audio).astype(np.float32) if buf_audio \
            else np.zeros(0, dtype=np.float32)
        return AudioChunk(
            id=chunk_id,
            audio=audio,
            sample_rate=16000,
            start_time=start,
            end_time=end,
            overlap_from_prev_s=0.0,
            meta={},
        )
