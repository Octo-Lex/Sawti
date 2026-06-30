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
        """Iterate at VAD sub-window (512-sample) granularity so chunk
        timestamps reflect actual speech boundaries to ~32ms precision,
        not the coarser outer-frame timestamp (review P2 #2).

        Each incoming frame is split into 512-sample sub-windows; each
        sub-window gets its own VAD verdict and its own timestamp, and the
        close-decision policy (pause / max-duration / min-duration) operates
        on those sub-windows. This avoids the previous bug where a single
        speech sub-window inside a 1s frame inflated the chunk's start/end
        to the whole frame.
        """
        cfg = self.config
        sample_rate = 16000
        window = 512  # Silero's required window at 16kHz
        window_s = window / sample_rate  # 0.032s
        window_ms = window_s * 1000.0

        buf_audio: list[np.ndarray] = []
        buf_start: float | None = None
        last_speech_end: float = 0.0
        silence_ms = 0.0
        open_chunk = False

        for frame in frames:
            audio = frame.audio
            base_ts = frame.timestamp_s
            sr = frame.sample_rate
            w = 512 if sr == 16000 else 256
            w_s = w / sr
            n = len(audio)
            if n == 0:
                continue
            # Slice the frame into full sub-windows; the trailing partial
            # sub-window is padded up to one window so Silero gets the size
            # it expects (its verdict applies to that last ~partial span).
            n_full = (n + w - 1) // w
            for wi in range(n_full):
                start_i = wi * w
                sub = audio[start_i : start_i + w]
                sub_ts = base_ts + start_i / sr
                sub_end = base_ts + min(start_i + w, n) / sr
                if len(sub) < w:
                    sub = np.pad(sub, (0, w - len(sub)))
                vr = self.vad.prob(sub, sr)

                if vr.is_speech:
                    if not open_chunk:
                        buf_start = sub_ts
                        open_chunk = True
                        silence_ms = 0.0
                    buf_audio.append(sub)
                    last_speech_end = sub_end
                    silence_ms = 0.0

                    # Force-close on max speech-content duration (mid-speech).
                    # buf_start is not None here (open_chunk is True).
                    content_ms = (last_speech_end - buf_start) * 1000.0
                    if content_ms >= cfg.max_chunk_duration_s * 1000.0:
                        if content_ms >= cfg.min_chunk_duration_ms:
                            yield self._emit(buf_audio, buf_start, last_speech_end)
                        buf_audio, open_chunk, buf_start = [], False, None
                        silence_ms = 0.0
                else:
                    if open_chunk:
                        silence_ms += window_ms
                        # Note: we do NOT append silent sub-windows to buf_audio
                        # (the emitted chunk's audio is speech content only).
                        content_ms = (last_speech_end - buf_start) * 1000.0
                        total_ms = (sub_end - buf_start) * 1000.0

                        # Force-close on max total span regardless of silence.
                        if total_ms >= cfg.max_chunk_duration_s * 1000.0:
                            if content_ms >= cfg.min_chunk_duration_ms:
                                yield self._emit(buf_audio, buf_start, last_speech_end)
                            buf_audio, open_chunk, buf_start = [], False, None
                            silence_ms = 0.0
                        # Close on pause threshold once min speech duration met.
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
