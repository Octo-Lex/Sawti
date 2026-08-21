"""Live microphone source (M2 Task 2) — AudioSource over sounddevice.

Architecture (docs/superpowers/plans/2026-08-21-m2-live-mic.md, frozen):

    PortAudio callback -> non-blocking copy into a bounded queue ->
    MicSource emits AudioFrame(float32 mono, 16 kHz, session timestamps)

The EXISTING Pipeline consumes these frames unchanged — M2 is capture-only
integration; no pipeline/segmenter/gate/model code is touched.

Fail-loud contract (plan Task 2, load-bearing): a full queue and a
PortAudio input-overflow are FATAL. MicSource never drops audio and never
falsifies the timeline; the error is raised on the next iteration of
iter_frames so the real-time callback itself stays non-blocking.

Timestamps come from a cumulative sample cursor
(``samples_emitted / 16000``), never wall-clock arrival: deterministic and
jitter-free while capture continuity holds; any discontinuity fails the
session instead of corrupting timestamps silently.

sounddevice is an OPTIONAL dependency (``uv sync --extra mic``); the
import is lazy so `import sawti`, tests, and non-mic commands never need
PortAudio.
"""
from __future__ import annotations

from queue import Full, Queue

import numpy as np

from sawti.sources import AudioFrame

TARGET_SR = 16000
CHANNELS = 1
DTYPE = "float32"
BLOCK_MS = 100
QUEUE_SECONDS = 30


class MicError(Exception):
    """Base: microphone capture problems."""


class MicUnavailableError(MicError):
    """The mic extra or the PortAudio backend is not usable."""


class MicCaptureError(MicError):
    """Capture failed (stream/device level)."""


class MicOverflowError(MicCaptureError):
    """Input overflow or bounded-queue overflow — audio would be lost."""


class MicSource:
    """AudioSource over a sounddevice.InputStream.

    device: None (backend default), a numeric PortAudio device id, or a
    case-insensitive name substring resolved against query_devices().
    """

    def __init__(self, device=None, block_ms: int = BLOCK_MS,
                 queue_seconds: float = QUEUE_SECONDS, backend=None) -> None:
        self.device = device
        self.block_ms = block_ms
        self._queue_blocks = max(1, int(queue_seconds * 1000 / block_ms))
        self._backend = backend          # injectable sounddevice-like module
        self._stream = None
        self._queue: Queue | None = None
        self._samples_emitted = 0
        self._capture_error: MicError | None = None
        self.selected_device = None

    # -- backend & device resolution ------------------------------------

    def _sd(self):
        if self._backend is None:
            try:
                import sounddevice  # lazy: optional 'mic' extra
            except Exception as e:  # ImportError, OSError (no PortAudio)
                raise MicUnavailableError(
                    "microphone backend unavailable — install it with "
                    "`uv sync --extra mic` (sounddevice + PortAudio)") from e
            self._backend = sounddevice
        return self._backend

    def _resolve_device(self, sd):
        if self.device is None or isinstance(self.device, int):
            return self.device
        needle = str(self.device).lower()
        for idx, info in enumerate(sd.query_devices()):
            if needle in str(info.get("name", "")).lower():
                return idx
        raise MicCaptureError(
            f"no input device matching {self.device!r} — see "
            f"`sawti listen --list-devices`")

    def _preflight(self, sd, dev) -> None:
        """Fail BEFORE the session (plan release rule): the device must
        open mono float32 at 16 kHz. No silent rate reinterpretation."""
        try:
            sd.check_input_settings(
                device=dev, channels=CHANNELS, dtype=DTYPE,
                samplerate=TARGET_SR)
        except Exception as e:
            raise MicCaptureError(
                f"device {dev!r} cannot capture {CHANNELS}ch/{DTYPE}/"
                f"{TARGET_SR}Hz (plan release rule: fail before session, "
                f"never reinterpret another rate): {e}") from e

    # -- real-time callback (non-blocking ONLY) --------------------------

    def _on_audio(self, indata, frames, time_info, status) -> None:
        """Runs under PortAudio real-time constraints: status check, one
        copy, one non-blocking enqueue. Everything else is deferred."""
        if indata is None:
            # End-of-stream sentinel — only injected backends/tests use it;
            # PortAudio never delivers a None block.
            self._queue.put_nowait(None)
            return
        if status:
            self._capture_error = MicOverflowError(
                f"PortAudio input status: {status}")
            return
        block = np.array(indata, dtype=np.float32, copy=True).reshape(-1)
        try:
            self._queue.put_nowait(block)
        except Full:
            self._capture_error = MicOverflowError(
                "capture queue full — downstream cannot keep up; refusing "
                "to drop audio (M2 fail-loud contract)")

    def _raise_deferred(self) -> None:
        if self._capture_error is not None:
            err, self._capture_error = self._capture_error, None
            raise err

    # -- AudioSource protocol ---------------------------------------------

    def iter_frames(self):
        self._open()
        try:
            while True:
                # Deferred fail-loud check FIRST: an error recorded while
                # the consumer was blocked in get() must surface even if a
                # sentinel/backlog sits in the queue ahead of it.
                self._raise_deferred()
                block = self._queue.get()
                if block is None:
                    # End-of-stream sentinel: only injected backends/tests
                    # use it; PortAudio never produces a None block.
                    break
                yield AudioFrame(
                    audio=np.ascontiguousarray(block, dtype=np.float32),
                    sample_rate=TARGET_SR,
                    timestamp_s=self._samples_emitted / TARGET_SR,
                )
                self._samples_emitted += int(block.shape[0])
        finally:
            self.close()

    def _open(self) -> None:
        sd = self._sd()
        dev = self._resolve_device(sd)
        self._preflight(sd, dev)
        self.selected_device = dev
        self._queue = Queue(maxsize=self._queue_blocks)
        self._samples_emitted = 0
        self._capture_error = None
        blocksize = int(TARGET_SR * self.block_ms / 1000)
        self._stream = sd.InputStream(
            samplerate=TARGET_SR, blocksize=blocksize, device=dev,
            channels=CHANNELS, dtype=DTYPE, callback=self._on_audio)
        self._stream.start()

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Idempotent teardown; safe from finally blocks and Ctrl+C."""
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.stop()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass

    def stop(self) -> None:
        self.close()
