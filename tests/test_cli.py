"""CLI contract tests: transcribe, eval, and listen (M2 Task 5).

Hermetic: listen tests inject a finite fake MicSource and reuse the M1
stub graph; --list-devices runs against a fake sounddevice module; no
microphone, no ML models, no network.
"""
import sys
import types

import numpy as np
from typer.testing import CliRunner

from sawti.cli import app
from sawti.sources import AudioFrame

runner = CliRunner()


def test_transcribe_runs_on_stubs():
    result = runner.invoke(app, ["transcribe", "--target", "eng"])
    assert result.exit_code == 0
    assert "hello" in result.stdout


def test_eval_runs_skeleton():
    result = runner.invoke(app, ["eval", "tests/fixtures", "--target", "eng"])
    assert result.exit_code == 0
    assert "report" in result.stdout.lower()


def test_transcribe_file_uses_real_pipeline(tmp_path):
    """transcribe <file> wires the real pipeline (silero vad stubbed via
    a config flag to keep it hermetic)."""
    import soundfile as sf
    wav = tmp_path / "clip.wav"
    sf.write(wav, np.zeros(16000, np.float32), 16000)
    result = runner.invoke(
        app, ["transcribe", str(wav), "--target", "eng", "--engine", "stub"]
    )
    assert result.exit_code == 0


def test_eval_engine_typo_rejected():
    result = runner.invoke(app, ["eval", "tests/fixtures", "--target", "eng",
                                 "--engine", "rael"])
    assert result.exit_code != 0
    assert "unsupported engine" in result.output


def test_transcribe_engine_typo_rejected(tmp_path):
    import soundfile as sf
    wav = tmp_path / "c.wav"
    sf.write(wav, np.zeros(16000, np.float32), 16000)
    result = runner.invoke(app, ["transcribe", str(wav), "--target", "eng",
                                 "--engine", "rael"])
    assert result.exit_code != 0
    assert "unsupported engine" in result.output


# ---- M2: sawti listen ----

class FakeMicSource:
    """Finite, controllable MicSource fake (plan Task 5)."""

    instances: list["FakeMicSource"] = []
    fail_with: Exception | None = None
    interrupt_after: int | None = None

    def __init__(self, device=None, **kw):
        self.device = device
        self.closed = 0
        self.frames = [
            AudioFrame(audio=np.full(1600, 0.1, np.float32),
                       sample_rate=16000, timestamp_s=0.1 * i)
            for i in range(4)
        ]
        FakeMicSource.instances.append(self)

    def iter_frames(self):
        for i, f in enumerate(self.frames):
            if FakeMicSource.interrupt_after is not None and \
                    i >= FakeMicSource.interrupt_after:
                raise KeyboardInterrupt  # simulate Ctrl+C mid-stream
            if FakeMicSource.fail_with is not None:
                raise FakeMicSource.fail_with
            yield f

    def close(self):
        self.closed += 1

    def stats(self):
        return {"captured_samples": 6400, "captured_seconds": 0.4,
                "emitted_samples": 6400, "emitted_seconds": 0.4,
                "queue_depth": 0, "queue_high_water": 1,
                "input_overflow_count": 0, "queue_overflow_count": 0,
                "selected_device": self.device, "sample_rate": 16000,
                "block_ms": 100, "blocksize_samples": 1600}


def _reset_fake(monkeypatch, **cls_attrs):
    FakeMicSource.instances = []
    FakeMicSource.fail_with = None
    FakeMicSource.interrupt_after = None
    for k, v in cls_attrs.items():
        setattr(FakeMicSource, k, v)
    monkeypatch.setattr("sawti.mic_source.MicSource", FakeMicSource)


def _with_stub_graph(monkeypatch):
    """listen must use the SAME production-graph seam as M1: the M1 stub
    graph proves the wiring without loading any model."""
    from sawti.cli import _stub_pipeline

    seen = {}

    class RecordingPipe:
        def run(self, source, target_lang):
            seen["target"] = target_lang
            seen["source"] = source
            yield from _stub_pipeline().run(source, target_lang=target_lang)

    monkeypatch.setattr("sawti.cli._real_pipeline",
                        lambda config, on_decision=None: RecordingPipe())
    return seen


def test_listen_forwards_target_and_normalizes_digit_device(monkeypatch):
    _reset_fake(monkeypatch)
    seen = _with_stub_graph(monkeypatch)
    result = runner.invoke(app, ["listen", "--target", "ara", "--device", "3"])
    assert result.exit_code == 0, result.output
    assert FakeMicSource.instances[0].device == 3      # int id, not "3"
    assert seen["target"] == "ara"
    assert "[0.00-" in result.stdout                    # [start-end] text
    assert FakeMicSource.instances[0].closed == 1


def test_listen_device_name_passes_through_as_substring(monkeypatch):
    _reset_fake(monkeypatch)
    _with_stub_graph(monkeypatch)
    result = runner.invoke(app, ["listen", "--device", "usb mic"])
    assert result.exit_code == 0
    assert FakeMicSource.instances[0].device == "usb mic"


def test_listen_rejects_bad_target(monkeypatch):
    result = runner.invoke(app, ["listen", "--target", "klingon"])
    assert result.exit_code != 0


def test_listen_list_devices_never_builds_the_graph(monkeypatch):
    def _forbidden(*a, **k):
        raise AssertionError("models loaded during --list-devices")

    monkeypatch.setattr("sawti.cli._real_pipeline", _forbidden)
    fake_sd = types.SimpleNamespace(query_devices=lambda: [
        {"name": "USB Conference Mic", "max_input_channels": 1,
         "default_samplerate": 48000},
        {"name": "HDMI Out", "max_input_channels": 0,
         "default_samplerate": 48000},
    ])
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    result = runner.invoke(app, ["listen", "--list-devices"])
    assert result.exit_code == 0, result.output
    assert "USB Conference Mic" in result.output
    assert "HDMI Out" not in result.output  # input devices only


def test_listen_mic_error_is_clean_cli_failure(monkeypatch):
    from sawti.mic_source import MicCaptureError

    _reset_fake(monkeypatch,
                fail_with=MicCaptureError("no input device matching 'ghost'"))
    _with_stub_graph(monkeypatch)
    result = runner.invoke(app, ["listen", "--device", "ghost"])
    assert result.exit_code == 1
    assert "listen:" in result.output and "ghost" in result.output
    assert "Traceback" not in result.output
    assert FakeMicSource.instances[0].closed == 1       # finally ran


def test_listen_ctrl_c_is_clean_shutdown(monkeypatch):
    _reset_fake(monkeypatch, interrupt_after=1)
    _with_stub_graph(monkeypatch)
    result = runner.invoke(app, ["listen"])
    assert result.exit_code == 0
    assert "Traceback" not in result.output
    assert FakeMicSource.instances[0].closed == 1
    assert "queue high-water" in result.stderr       # stats to stderr only


def test_listen_finite_source_exits_and_closes_capture(monkeypatch):
    _reset_fake(monkeypatch)
    _with_stub_graph(monkeypatch)
    result = runner.invoke(app, ["listen", "--target", "fra"])
    assert result.exit_code == 0
    assert FakeMicSource.instances[0].closed == 1


def test_transcribe_and_eval_unchanged_by_listen():
    # No injection: the M1 commands still run their original paths.
    r1 = runner.invoke(app, ["transcribe", "--target", "eng"])
    r2 = runner.invoke(app, ["eval", "tests/fixtures", "--target", "eng"])
    assert r1.exit_code == 0 and "hello" in r1.stdout
    assert r2.exit_code == 0 and "report" in r2.stdout.lower()


# ---- hermetic replay: MicSource -> the M1 graph (plan Task 5) ----

# ---- teardown reporting must never crash shutdown (Task 7 finding) ----

class _StatsSrc:
    def stats(self):
        return {"captured_seconds": 6.6, "emitted_seconds": 6.6,
                "queue_depth": 0, "queue_high_water": 1,
                "input_overflow_count": 0, "queue_overflow_count": 0,
                "selected_device": 15, "sample_rate": 16000,
                "block_ms": 100, "blocksize_samples": 1600,
                "captured_samples": 105600, "emitted_samples": 105600}


def test_capture_stats_survive_broken_console_handle(monkeypatch):
    """Task 7 real-session finding: on Windows Ctrl+C, the colorama/click
    wrapped stderr can raise OSError 6 (invalid handle) mid-teardown —
    the stats line was lost and a traceback replaced the clean exit. The
    print must fall back to the raw interpreter stderr and NEVER raise."""
    import io
    import sys

    import sawti.cli as cli

    def _broken_echo(msg=None, err=False, **kw):
        raise OSError(6, "Windows error: 6 (invalid console handle)")

    raw = io.StringIO()                       # the raw-interpreter stderr
    monkeypatch.setattr(cli.typer, "echo", _broken_echo)
    monkeypatch.setattr(sys, "__stderr__", raw)
    cli._print_capture_stats(_StatsSrc())      # must not raise
    assert "[capture] 6.6s captured" in raw.getvalue()  # fallback delivered
    assert "input overflows 0" in raw.getvalue()


def test_capture_stats_double_failure_exits_silently(monkeypatch):
    """Even with BOTH the wrapped echo and the raw interpreter stderr
    broken, teardown stays clean — a lost summary beats a crashed exit."""
    import sys

    import sawti.cli as cli

    def _broken_echo(msg=None, err=False, **kw):
        raise OSError(6, "Windows error: 6")

    class _BrokenRaw:
        def write(self, *a):
            raise OSError(6)

        def flush(self):
            raise OSError(6)

    monkeypatch.setattr(cli.typer, "echo", _broken_echo)
    monkeypatch.setattr(sys, "__stderr__", _BrokenRaw())
    cli._print_capture_stats(_StatsSrc())      # must not raise


def test_replay_real_mic_source_through_m1_stub_graph():
    """The load-bearing M2 proof: the REAL MicSource (fake backend
    replaying known waveform blocks through the REAL callback) feeds the
    existing M1 graph stubs, and source order/timestamps survive the live
    edge — M2 uses the M1 graph, not a parallel implementation."""
    from sawti.cli import _stub_pipeline
    from sawti.mic_source import MicSource
    from tests.test_mic_source import FakeSD

    blocks = [np.full(1600, v, np.float32) for v in (0.1, 0.2, 0.3, 0.4)]
    src = MicSource(device=None, backend=FakeSD(blocks=blocks))
    pipe = _stub_pipeline()  # StubSegmenter(chunk_frames=2) -> 2 chunks
    segs = list(pipe.run(src, target_lang="eng"))
    assert len(segs) == 2
    # Sample-cursor timestamps survive the graph: chunk 1 packs frames at
    # 0.0-0.1s, chunk 2 at 0.1-0.2s; each end covers both packed frames.
    assert segs[0].start_time == 0.0 and segs[0].end_time == 0.2
    assert segs[1].start_time == 0.2 and segs[1].end_time == 0.4
    assert all(s.text.strip() for s in segs)
