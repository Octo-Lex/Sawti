"""Commit 5: the real evaluator — injected seam, structured traces, honest
aggregates. No stub hypotheses anywhere (the CLI's stub-COMPONENT pipeline
is a real execution, not a stubbed hypothesis)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from eval.harness import run_eval
from eval.metrics import compute_chrf, norm_for_eval, wer_counts
from eval.transcribers import Transcription, make_pipeline_transcriber


def _fake_transcriber(results: dict[str, Transcription]):
    def fn(wav_path: str) -> Transcription:
        return results[Path(wav_path).name]
    return fn


def _mk_clip(tmp_path: Path, name: str, ref: str | None, dur_s: float = 0.5):
    sf.write(tmp_path / f"{name}.wav", np.zeros(int(16000 * dur_s), np.float32), 16000)
    if ref is not None:
        (tmp_path / f"{name}.txt").write_text(ref, encoding="utf-8")


# --- metric primitives (kept from before) ---

def test_real_chrf_perfect_match_scores_high():
    assert compute_chrf("hello world", "hello world") > 90.0


def test_real_chrf_mismatch_scores_lower_than_match():
    match = compute_chrf("hello world", "hello world")
    miss = compute_chrf("hello world", "completely different text here")
    assert miss < match


def test_wer_counts_exact_math():
    wer, edits, n_ref = wer_counts("a b", "a x y")     # 1 sub + 1 ins / 2
    assert wer == 1.0 and edits == 2.0 and n_ref == 2
    assert wer_counts("", "anything") == (None, 0.0, 0)


def test_norm_for_eval_matches_spike_recipe():
    # Note: the spike recipe does NOT lowercase (matches the Saudi
    # harness numbers verbatim); casing survives normalization.
    assert norm_for_eval("مَرْحَباً ، World!") == "مرحبا ، World"


# --- harness contract ---

def test_missing_transcriber_is_an_error():
    with pytest.raises(ValueError, match="transcriber"):
        run_eval("whatever", target_lang="eng", transcriber=None)


def test_report_schema_and_aggregates(tmp_path: Path):
    _mk_clip(tmp_path, "b_second", "a b c", 0.5)
    _mk_clip(tmp_path, "a_first", "a b", 0.5)
    tr = _fake_transcriber(
        {
            "a_first.wav": Transcription(hypothesis="a x y", low_confidence=True,
                                         fallback_paths=["asr_mt"]),
            "b_second.wav": Transcription(hypothesis="a b c"),
        }
    )
    out = run_eval(tmp_path, target_lang="eng", transcriber=tr, output_dir=tmp_path)
    report = json.loads(Path(out).read_text(encoding="utf-8"))

    # Deterministic ordering: sorted clip discovery.
    assert [c["clip"] for c in report["clips"]] == ["a_first.wav", "b_second.wav"]
    assert report["n_clips"] == 2 and report["n_referenced"] == 2

    m = report["metrics"]
    # a_first: wer 1.0 (2 edits/2 ref); b_second: 0. macro = 0.5.
    assert m["macro_wer"] == pytest.approx(50.0)
    # corpus: (2 + 0) / (2 + 3) = 0.4.
    assert m["corpus_wer"] == pytest.approx(40.0)
    assert m["mean_chrf"] is not None
    assert m["loop_rate"] == 0.0

    clip = report["clips"][0]
    assert clip["has_reference"] is True
    assert clip["low_confidence"] is True and clip["fallback_paths"] == ["asr_mt"]
    for k in ("clip", "reference", "hypothesis", "chrf", "wer", "loop",
              "segments", "trace", "trace_text"):
        assert k in clip


def test_missing_reference_evaluable_but_excluded(tmp_path: Path):
    _mk_clip(tmp_path, "no_ref", None)
    _mk_clip(tmp_path, "with_ref", "hello world")
    tr = _fake_transcriber(
        {
            "no_ref.wav": Transcription(hypothesis="guessed text"),
            "with_ref.wav": Transcription(hypothesis="hello world"),
        }
    )
    out = run_eval(tmp_path, target_lang="eng", transcriber=tr, output_dir=tmp_path)
    report = json.loads(Path(out).read_text(encoding="utf-8"))

    assert report["n_clips"] == 2 and report["n_referenced"] == 1
    no_ref = report["clips"][0]
    assert no_ref["clip"] == "no_ref.wav"
    assert no_ref["has_reference"] is False
    assert no_ref["hypothesis"] == "guessed text"   # produced anyway
    assert no_ref["chrf"] is None and no_ref["wer"] is None
    assert report["metrics"]["macro_wer"] == pytest.approx(0.0)  # only ref'd clip


def test_loop_rate_uses_shared_production_detector(tmp_path: Path):
    _mk_clip(tmp_path, "loop", "x")
    _mk_clip(tmp_path, "clean", "y")
    tr = _fake_transcriber(
        {
            "loop.wav": Transcription(hypothesis="اشتركوا في القناه " * 3),
            "clean.wav": Transcription(hypothesis="no no wait no no stop"),
        }
    )
    out = run_eval(tmp_path, target_lang="eng", transcriber=tr, output_dir=tmp_path)
    report = json.loads(Path(out).read_text(encoding="utf-8"))
    assert report["metrics"]["loop_rate"] == 50.0
    assert report["metrics"]["n_loops"] == 1
    # Sorted discovery: clean.wav sorts before loop.wav.
    assert report["clips"][0]["clip"] == "clean.wav"
    assert report["clips"][0]["loop"] is False   # frequency alone never gates
    assert report["clips"][1]["loop"] is True


def test_empty_set_is_valid(tmp_path: Path):
    tr = _fake_transcriber({})
    out = run_eval(tmp_path, target_lang="eng", transcriber=tr, output_dir=tmp_path)
    report = json.loads(Path(out).read_text(encoding="utf-8"))
    assert report["n_clips"] == 0
    assert report["metrics"]["macro_wer"] is None


def test_deterministic_report_content(tmp_path: Path):
    _mk_clip(tmp_path, "c1", "a b")
    tr = _fake_transcriber({"c1.wav": Transcription(hypothesis="a b")})
    a = run_eval(tmp_path, target_lang="eng", transcriber=tr, output_dir=tmp_path / "r1")
    b = run_eval(tmp_path, target_lang="eng", transcriber=tr, output_dir=tmp_path / "r2")
    assert Path(a).read_text(encoding="utf-8") == Path(b).read_text(encoding="utf-8")


# --- req 10 acceptance: full fallback traversal preserved in the report ---

def test_fallback_stage_sequence_preserved_in_report(tmp_path: Path):
    from sawti.engine import EngineManager
    from sawti.fallback import FallbackHandler
    from sawti.pipeline import Pipeline
    from sawti.postprocess import StubPostProcessor
    from sawti.segmenter import StubSegmenter
    from sawti.types import EngineResult
    from tests.test_fallback_contract import (ConservativeFake, FakeAsrMt,
                                              FakeRechunker, ScriptedGate)

    _mk_clip(tmp_path, "bad", "anything", dur_s=1.0)   # 1 chunk under stubs

    class OneShotEngine:
        def translate(self, chunk, target):
            return EngineResult(chunk.id, "weak", 0.3, "ara", {}, target)

    # Gate verdicts: primary(fail) -> retry(fail) -> r0(fail) -> r1(fail)
    # -> asr_mt(accept).
    gate = ScriptedGate([
        {"accepted": False, "checks": {"repetition_loop": True}},
        {"accepted": False, "checks": {"low_confidence": True}},
        {"accepted": False, "checks": {"empty_output": True}},
        {"accepted": False, "checks": {"low_confidence": True}},
        {"accepted": True, "checks": {}},
    ])
    engine = OneShotEngine()
    fallback = FallbackHandler(
        engine=engine, gate=gate, asr_mt=FakeAsrMt(),
        rechunker=FakeRechunker(), conservative=ConservativeFake())

    def factory(on_decision=None):
        return Pipeline(
            segmenter=StubSegmenter(chunk_frames=2, sample_rate=16000),
            engine=EngineManager(engine=engine),
            gate=gate,
            postprocessor=StubPostProcessor(),
            fallback=fallback,
            on_decision=on_decision,
        )

    transcriber = make_pipeline_transcriber(factory, "eng", frame_samples=16000)
    out = run_eval(tmp_path, target_lang="eng", transcriber=transcriber,
                   output_dir=tmp_path)
    report = json.loads(Path(out).read_text(encoding="utf-8"))

    clip = report["clips"][0]
    stages = [e["stage"] for e in clip["trace"] if isinstance(e, dict) and "stage" in e]
    assert stages == ["primary", "retry", "rechunk[0]", "rechunk[1]", "asr_mt"]
    assert clip["fallback_paths"] == ["asr_mt"]
    assert clip["hypothesis"] == "asr_mt:recovered"
    assert clip["low_confidence"] is False
    # The human-readable rendering exists but metrics consumed structure.
    assert "asr_mt        -> accepted" in clip["trace_text"]


# --- corrective pass: zero-word references + chunk attribution ---

def test_punctuation_only_reference_pins_zero_words():
    # Non-empty TEXTUAL references that normalize to zero words (ASCII
    # punctuation is stripped by the spike recipe).
    assert wer_counts("...", "anything") == (None, 0.0, 0)
    assert wer_counts("!? ... ---", "anything") == (None, 0.0, 0)
    # Documented recipe consequence: Arabic-script punctuation (، ؛)
    # SURVIVES normalization (spike-consistent) and therefore counts as
    # word-bearing — it enters WER denominators as garbage words rather
    # than being zero-word-excluded.
    assert wer_counts("، ؛", "anything")[2] == 2


def test_zero_word_reference_no_crash_no_contamination(tmp_path: Path):
    _mk_clip(tmp_path, "punct", "...")            # textual ref, zero words
    _mk_clip(tmp_path, "valid", "a b")
    tr = _fake_transcriber(
        {
            "punct.wav": Transcription(hypothesis="some guess"),
            "valid.wav": Transcription(hypothesis="a b"),
        }
    )
    out = run_eval(tmp_path, target_lang="eng", transcriber=tr, output_dir=tmp_path)
    report = json.loads(Path(out).read_text(encoding="utf-8"))   # no crash

    # Presence vs WER-eligibility denominators are now explicit.
    assert report["n_referenced"] == 2
    assert report["n_wer_referenced"] == 1
    punct = report["clips"][0]
    assert punct["has_reference"] is True and punct["wer"] is None
    # The valid clip's aggregates are uncontaminated.
    assert report["metrics"]["macro_wer"] == pytest.approx(0.0)
    assert report["metrics"]["corpus_wer"] == pytest.approx(0.0)
    assert report["metrics"]["mean_chrf"] is not None   # chrF still uses ref_rows


def test_all_zero_word_references_wer_none_not_zero(tmp_path: Path):
    _mk_clip(tmp_path, "only", "...")
    tr = _fake_transcriber({"only.wav": Transcription(hypothesis="x")})
    out = run_eval(tmp_path, target_lang="eng", transcriber=tr, output_dir=tmp_path)
    report = json.loads(Path(out).read_text(encoding="utf-8"))
    assert report["n_referenced"] == 1
    assert report["n_wer_referenced"] == 0
    assert report["metrics"]["macro_wer"] is None    # None, not 0.0
    assert report["metrics"]["corpus_wer"] is None


def test_every_trace_entry_is_chunk_attributed(tmp_path: Path):
    """Happy-path multi-chunk clip: gate entries without stage keys still
    leave the adapter with chunk attribution (filled in copies — no stage
    synthesis)."""
    from sawti.engine import EngineManager, StubEngine
    from sawti.pipeline import Pipeline
    from sawti.postprocess import StubPostProcessor
    from sawti.quality_gate import StubQualityGate
    from sawti.segmenter import StubSegmenter

    _mk_clip(tmp_path, "multi", "hello hello", dur_s=4.0)  # 2 chunks under stubs

    def factory(on_decision=None):
        return Pipeline(
            segmenter=StubSegmenter(chunk_frames=2, sample_rate=16000),
            engine=EngineManager(engine=StubEngine("hello", 0.9)),
            gate=StubQualityGate(),
            postprocessor=StubPostProcessor(),
            on_decision=on_decision,
        )

    transcriber = make_pipeline_transcriber(factory, "eng", frame_samples=16000)
    out = run_eval(tmp_path, target_lang="eng", transcriber=transcriber,
                   output_dir=tmp_path)
    report = json.loads(Path(out).read_text(encoding="utf-8"))
    clip = report["clips"][0]
    assert clip["hypothesis"] == "hello hello"

    entries = [e for e in clip["trace"] if isinstance(e, dict)]
    assert entries, "expected trace entries on the happy path"
    assert all("chunk_id" in e for e in entries)     # full attribution
    ids = {e["chunk_id"] for e in entries}
    assert ids == {"c0", "c1"}                        # both chunks covered
    # Primary-success gate records stay unstaged (no synthesis).
    assert all("stage" not in e for e in entries)
