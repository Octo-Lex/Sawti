from typer.testing import CliRunner

from sawti.cli import app

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
    import numpy as np
    import soundfile as sf
    wav = tmp_path / "clip.wav"
    sf.write(wav, np.zeros(16000, np.float32), 16000)
    result = runner.invoke(
        app, ["transcribe", str(wav), "--target", "eng", "--engine", "stub"]
    )
    assert result.exit_code == 0
