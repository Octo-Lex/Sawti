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
