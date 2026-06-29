import json
from pathlib import Path

from eval.harness import run_eval
from eval.metrics import compute_chrf_stub


def test_chrf_stub_returns_score_for_match():
    score = compute_chrf_stub("hello world", "hello world")
    assert 0.0 <= score <= 100.0


def test_chrf_stub_lower_for_mismatch():
    match = compute_chrf_stub("hello world", "hello world")
    miss = compute_chrf_stub("hello world", "completely different")
    assert miss < match


def test_run_eval_writes_report(tmp_path: Path):
    # empty eval set; harness skeleton should still produce a report file
    report_path = run_eval(tmp_path, target_lang="eng")
    assert Path(report_path).exists()
    data = json.loads(Path(report_path).read_text(encoding="utf-8"))
    assert data["target_lang"] == "eng"
    assert "clips" in data
    assert "metrics" in data
