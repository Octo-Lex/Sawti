"""Eval harness skeleton (spec §7.6). M0: discovers clips, scores with the
chrF stub against .txt references, writes a JSON report. M1 swaps the stub
pipeline + metric for real components.
"""
from __future__ import annotations

from pathlib import Path

from eval.metrics import compute_chrf_stub
from eval.report import write_report


def run_eval(eval_set: Path, target_lang: str) -> str:
    clips = sorted(eval_set.glob("*.wav"))
    scored = []
    for wav in clips:
        ref_path = wav.with_suffix(".txt")
        ref = ref_path.read_text(encoding="utf-8").strip() if ref_path.exists() else ""
        hyp = "[stub hypothesis]"  # M1: run real pipeline on `wav`
        chrf = compute_chrf_stub(hyp, ref) if ref else None
        scored.append({"clip": wav.name, "chrf": chrf, "has_reference": bool(ref)})

    report = {
        "target_lang": target_lang,
        "n_clips": len(scored),
        "clips": scored,
        "metrics": {
            "mean_chrf": (
                sum(c["chrf"] for c in scored if c["chrf"] is not None)
                / max(1, sum(1 for c in scored if c["chrf"] is not None))
            ) if scored else 0.0,
        },
    }
    out_path = Path("outputs") / f"eval-{target_lang}.json"
    return write_report(out_path, report)
