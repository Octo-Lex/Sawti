"""Real pipeline evaluator (spec §7.6) — Commit 5.

Contract:
- NO stub hypotheses: every evaluated WAV flows through the injected
  ``transcriber`` seam (``(wav_path) -> eval.transcribers.Transcription``).
  The CLI injects a real stub-component pipeline until Commit 6 binds the
  production builder.
- Structured traces: the per-clip ``trace`` is the final GateDecision's
  stage log (stage / attempt / round / checks / accepted /
  low_confidence), persisted as dictionaries — metrics never parse text.
  ``trace_text`` is an optional human-readable rendering.
- WAV + sibling TXT contract only. No clean/dialect labels exist here;
  that distinction belongs to the Saudi harness where it is defined.
- Missing references stay evaluable (hypothesis + trace produced) but are
  excluded from every reference-based denominator.
- Deterministic: sorted clip discovery, fixed report field order.
- Hermetic: importing this module loads no models.
"""
from __future__ import annotations

from pathlib import Path

from eval.metrics import compute_chrf, wer_counts
from eval.report import write_report
from sawti.loop_detect import is_loop


def run_eval(
    eval_set: Path | str,
    target_lang: str,
    transcriber=None,
    output_dir: Path | str | None = None,
) -> str:
    """Evaluate every WAV in eval_set through the transcriber seam.

    Args:
        eval_set: directory of *.wav clips (sibling *.txt = reference).
        target_lang: target language code (eng|ara|fra).
        transcriber: REQUIRED callable (wav_path) -> Transcription. The
            stub-hypothesis era is over; a missing transcriber is an error.
        output_dir: report destination (default ``outputs/``).
    """
    if transcriber is None:
        raise ValueError(
            "run_eval requires a transcriber (see eval.transcribers."
            "make_pipeline_transcriber); stub hypotheses are gone"
        )
    eval_set = Path(eval_set)
    wavs = sorted(eval_set.glob("*.wav"))

    rows = []
    for wav in wavs:
        ref_path = wav.with_suffix(".txt")
        ref = ref_path.read_text(encoding="utf-8").strip() if ref_path.exists() else ""
        t = transcriber(str(wav))
        has_ref = bool(ref)
        wer = edits = None
        n_ref = 0
        chrf = None
        if has_ref:
            wer, edits, n_ref = wer_counts(ref, t.hypothesis)
            chrf = compute_chrf(t.hypothesis, ref)
        rows.append(
            {
                "clip": wav.name,
                "reference": ref,
                "has_reference": has_ref,
                "hypothesis": t.hypothesis,
                "low_confidence": t.low_confidence,
                "fallback_paths": t.fallback_paths,
                "loop": is_loop(t.hypothesis) if t.hypothesis else False,
                "chrf": chrf,
                "wer": wer,
                "segments": t.segments,
                "trace": t.trace,
                "trace_text": t.trace_text,
                "_edits": edits if edits is not None else 0.0,
                "_n_ref": n_ref,
            }
        )

    ref_rows = [r for r in rows if r["has_reference"]]
    metrics = {
        "macro_wer": (
            100.0 * sum(r["wer"] for r in ref_rows) / len(ref_rows) if ref_rows else None
        ),
        "corpus_wer": (
            100.0 * sum(r["_edits"] for r in ref_rows)
            / max(1, sum(r["_n_ref"] for r in ref_rows))
            if ref_rows else None
        ),
        "mean_chrf": (
            sum(r["chrf"] for r in ref_rows) / len(ref_rows) if ref_rows else None
        ),
        "loop_rate": 100.0 * sum(1 for r in rows if r["loop"]) / max(1, len(rows)),
        "n_loops": sum(1 for r in rows if r["loop"]),
    }

    clips = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    report = {
        "target_lang": target_lang,
        "n_clips": len(rows),
        "n_referenced": len(ref_rows),
        "metrics": metrics,
        "clips": clips,
    }
    out_dir = Path(output_dir) if output_dir is not None else Path("outputs")
    out_path = out_dir / f"eval-{target_lang}.json"
    return write_report(out_path, report)
