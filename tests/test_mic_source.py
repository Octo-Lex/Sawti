"""M2 Task 3: MicSource hermetic contract tests.

No microphone, no sounddevice import, no PortAudio — the backend is an
injected fake. These pin the load-bearing capture/backpressure contract
from docs/superpowers/plans/2026-08-21-m2-live-mic.md.
"""
import sys
from queue import Full, Queue

import numpy as np
import pytest

import sawti.mic_source as ms
from sawti.mic_source import (
    CHANNELS,
    DTYPE,
    TARGET_SR,
    MicCaptureError,
    MicOverflowError,
    MicSource,
    MicUnavailableError,
)


class FakeStream:
    def __init__(self, callback, blocks):
        self.callback = callback
        self.blocks = blocks
        self.stopped = 0
        self.closed = 0

    def start(self):
        # Replay scripted blocks through the REAL callback (exercising the
        # copy/enqueue path), then signal end-of-stream via the sentinel.
        for b in self.blocks:
            self.callback(np.asarray(b, np.float32).reshape(-1, 1),
                          len(b), None, None)
        self.callback(None, 0, None, None)

    def stop(self):
        self.stopped += 1

    def close(self):
        self.closed += 1


class FakeSD:
    """sounddevice-shaped fake. blocks: list of 1-D waveforms to replay."""

    def __init__(self, blocks=(), devices=None, settings_error=False):
        self.blocks = list(blocks)
        self.devices = devices or [{"name": "USB Conference Mic",
                                    "max_input_channels": 2}]
        self.settings_error = settings_error
        self.check_calls = []
        self.stream: FakeStream | None = None
        self.stream_kwargs = None

    def query_devices(self):
        return self.devices

    def check_input_settings(self, device=None, channels=None, dtype=None,
                             samplerate=None):
        self.check_calls.append({"device": device, "channels": channels,
                                 "dtype": dtype, "samplerate": samplerate})
        if self.settings_error:
            raise ValueError("invalid input settings")

    def InputStream(self, samplerate, blocksize, device, channels, dtype,
                    callback):
        self.stream_kwargs = {"samplerate": samplerate, "blocksize": blocksize,
                              "device": device, "channels": channels,
                              "dtype": dtype}
        self.stream = FakeStream(callback, self.blocks)
        return self.stream


def _src(backend, device=None, **kw):
    return MicSource(device=device, backend=backend, **kw)


def test_module_import_is_hermetic():
    """No sounddevice anywhere in the import graph (CPU-only CI pin)."""
    assert "sounddevice" not in sys.modules
    assert ms.TARGET_SR == 16000 and ms.CHANNELS == 1
    assert ms.DTYPE == "float32" and ms.BLOCK_MS == 100


def test_preflight_called_with_frozen_capture_contract():
    sd = FakeSD(blocks=[])
    list(_src(sd).iter_frames())
    assert sd.check_calls == [{"device": None, "channels": CHANNELS,
                               "dtype": DTYPE, "samplerate": TARGET_SR}]
    assert sd.stream_kwargs["samplerate"] == 16000
    assert sd.stream_kwargs["channels"] == 1
    assert sd.stream_kwargs["dtype"] == "float32"
    assert sd.stream_kwargs["blocksize"] == 1600  # 100 ms at 16 kHz


def test_preflight_failure_fails_before_session():
    sd = FakeSD(blocks=[], settings_error=True)
    with pytest.raises(MicCaptureError, match="fail before session"):
        list(_src(sd).iter_frames())
    assert sd.stream is None  # no stream was ever opened


def test_emitted_frames_are_contiguous_1d_float32():
    sd = FakeSD(blocks=[np.zeros(1600), np.zeros(1600)])
    frames = list(_src(sd).iter_frames())
    assert len(frames) == 2
    for f in frames:
        assert f.audio.dtype == np.float32 and f.audio.ndim == 1
        assert f.audio.flags["C_CONTIGUOUS"]
        assert f.sample_rate == TARGET_SR


def test_cumulative_timestamps_exact_and_monotonic():
    blocks = [np.zeros(1600), np.zeros(1600), np.zeros(800)]  # variable tail
    frames = list(_src(FakeSD(blocks=blocks)).iter_frames())
    assert [f.timestamp_s for f in frames] == [0.0, 0.1, 0.2]
    ts = [f.timestamp_s for f in frames]
    assert ts == sorted(ts)


def test_variable_final_block_preserves_cursor_arithmetic():
    # 100 ms + 50 ms + 30 ms: timestamps follow the SAMPLE cursor exactly.
    blocks = [np.zeros(1600), np.zeros(800), np.zeros(480)]
    frames = list(_src(FakeSD(blocks=blocks)).iter_frames())
    assert [f.timestamp_s for f in frames] == [0.0, 0.1, 0.15]


def test_callback_order_and_samples_preserved():
    """Source-contract regression: a known synthetic waveform crosses the
    callback/queue boundary unaltered and in order."""
    wave = np.arange(3 * 1600, dtype=np.int64) % 97  # deterministic ramp
    blocks = [wave[:1600].astype(np.float32),
              wave[1600:3200].astype(np.float32),
              wave[3200:].astype(np.float32)]
    frames = list(_src(FakeSD(blocks=blocks)).iter_frames())
    got = np.concatenate([f.audio for f in frames])
    assert np.array_equal(got, wave.astype(np.float32))  # bit-exact


def test_device_resolution_none_int_and_substring():
    sd = FakeSD(blocks=[])
    list(_src(sd, device=None).iter_frames())
    assert sd.check_calls[0]["device"] is None

    sd = FakeSD(blocks=[])
    list(_src(sd, device=3).iter_frames())
    assert sd.check_calls[0]["device"] == 3  # numeric id passes through

    sd = FakeSD(blocks=[])
    list(_src(sd, device="usb conference").iter_frames())
    assert sd.check_calls[0]["device"] == 0  # substring, case-insensitive

    with pytest.raises(MicCaptureError, match="no input device matching"):
        list(_src(FakeSD(blocks=[]), device="ghost").iter_frames())


def test_queue_saturation_fails_loudly_never_drops():
    src = MicSource(block_ms=100, queue_seconds=0.1)  # 1-block queue
    src._queue = Queue(maxsize=1)
    first = np.full(1600, 0.5, np.float32)
    src._on_audio(first.reshape(-1, 1), 1600, None, None)
    assert src._queue.qsize() == 1
    second = np.full(1600, -0.5, np.float32)
    src._on_audio(second.reshape(-1, 1), 1600, None, None)  # queue FULL
    # Nothing dropped, nothing overwritten:
    assert src._queue.qsize() == 1
    assert np.array_equal(src._queue.get_nowait(), first)
    # The failure surfaces on the next consumer check:
    with pytest.raises(MicOverflowError, match="queue full"):
        src._raise_deferred()


def test_input_overflow_status_fails_loudly():
    src = MicSource()
    src._queue = Queue(maxsize=4)
    src._on_audio(np.zeros((1600, 1), np.float32), 1600, None,
                  status="input overflow")
    assert src._queue.qsize() == 0  # overflowed block NOT enqueued
    with pytest.raises(MicOverflowError, match="input overflow"):
        src._raise_deferred()


def test_deferred_overflow_surfaces_through_iter_frames():
    sd = FakeSD(blocks=[np.zeros(1600)])
    src = _src(sd)
    gen = src.iter_frames()
    next(gen)                       # first frame fine
    # Overflow recorded while the consumer waits on the queue:
    src._capture_error = MicOverflowError("boom")
    with pytest.raises(MicOverflowError, match="boom"):
        next(gen)


def test_close_is_idempotent():
    sd = FakeSD(blocks=[np.zeros(1600)])
    src = _src(sd)
    gen = src.iter_frames()
    next(gen)
    src.close()
    src.close()
    src.stop()
    assert sd.stream.stopped == 1 and sd.stream.closed == 1


def test_stream_closes_on_generator_exit():
    sd = FakeSD(blocks=[np.zeros(1600)] * 5)
    src = _src(sd)
    gen = src.iter_frames()
    next(gen)
    gen.close()                     # GeneratorExit -> finally -> close()
    assert sd.stream.closed == 1


def test_stream_closes_on_consumer_exception():
    sd = FakeSD(blocks=[np.zeros(1600)] * 5)
    src = _src(sd)
    with pytest.raises(RuntimeError, match="consumer died"):
        for i, _ in enumerate(src.iter_frames()):
            if i == 1:
                raise RuntimeError("consumer died")
    assert sd.stream.closed == 1


def test_unavailable_backend_gives_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)  # import fails
    src = MicSource()  # no backend injected -> tries the real import
    with pytest.raises(MicUnavailableError, match="uv sync --extra mic"):
        list(src.iter_frames())
