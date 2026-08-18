"""Spike: zero-shot Whisper-large-v3 WER on the SADA Saudi-dialect sample.

Reference: manifest cleaned_text (fallback text), normalized with Sawti's
normalize_arabic_for_match (alef unification, diacritic/tatweel removal) on
both hypothesis and reference; punctuation stripped. Research use only.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import sawti.env  # noqa: F401
from sawti.text_normalize import normalize_arabic_for_match

DATA = Path("data/sada_spike")
MODEL_ID = "openai/whisper-large-v3"


def norm(text: str) -> str:
    """Matching-grade Arabic normalization + punctuation strip for WER."""
    import re

    t = normalize_arabic_for_match(text)
    t = re.sub(r"[^\w\s\u0600-\u06FF]", " ", t)  # drop punctuation/symbols
    return " ".join(t.split())


def main() -> None:
    import jiwer
    import torch
    from transformers import pipeline

    manifest = [
        json.loads(line)
        for line in (DATA / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | model: {MODEL_ID}")
    asr = pipeline(
        "automatic-speech-recognition",
        model=MODEL_ID,
        torch_dtype=torch.float16,
        device=0 if device == "cuda" else -1,
        chunk_length_s=30,  # SADA clips run up to ~96s
    )

    results = []
    for m in manifest:
        wav = str(DATA / f"{m['clip_id']}.wav")
        hyp = asr(wav, generate_kwargs={"language": "arabic", "task": "transcribe"})
        hyp_text = hyp["text"].strip()
        ref_text = (m.get("cleaned_text") or m.get("text") or "").strip()
        n_hyp, n_ref = norm(hyp_text), norm(ref_text)
        wer = jiwer.wer(n_ref, n_hyp) if n_ref else float("nan")
        results.append(
            {**m, "hyp": hyp_text, "wer": round(wer, 4) if wer == wer else None}
        )

    # Aggregate: overall + per dialect.
    def agg(rows):
        w = [r["wer"] for r in rows if r["wer"] is not None]
        return (100 * sum(w) / len(w), len(w)) if w else (float("nan"), 0)

    overall, n = agg(results)
    print(f"\n=== OVERALL normalized WER: {overall:.1f}%  (n={n}) ===")
    by_d = defaultdict(list)
    for r in results:
        by_d[r["dialect"]].append(r)
    for d, rows in sorted(by_d.items()):
        w, cnt = agg(rows)
        print(f"  {d:10s} WER {w:5.1f}%  (n={cnt})")

    # Duration/quality stats.
    durs = [m["duration_s"] for m in results]
    print(
        f"\ndurations: min={min(durs):.1f}s median={sorted(durs)[len(durs)//2]:.1f}s "
        f"max={max(durs):.1f}s"
    )
    empty_ref = sum(1 for r in results if not norm(r.get("cleaned_text") or r.get("text") or ""))
    print(f"empty-after-norm references: {empty_ref}")

    # Worst / best examples for qualitative inspection.
    print("\n=== 3 worst ===")
    for r in sorted(results, key=lambda r: -(r["wer"] or 0))[:3]:
        print(f"[{r['clip_id']} {r['dialect']} wer={r['wer']}]")
        print(f"  ref: {(r.get('cleaned_text') or r.get('text'))[:150]}")
        print(f"  hyp: {r['hyp'][:150]}")
    print("=== 3 best ===")
    for r in sorted(results, key=lambda r: (r["wer"] if r["wer"] is not None else 9))[:3]:
        print(f"[{r['clip_id']} {r['dialect']} wer={r['wer']}] {r['hyp'][:100]}")

    (DATA / "eval_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"\nfull results -> {DATA / 'eval_results.json'}")


if __name__ == "__main__":
    main()
