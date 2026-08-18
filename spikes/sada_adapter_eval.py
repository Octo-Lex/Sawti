"""Spike: evaluate dev-ahmedhany/whisper-large-v3-arabic-ft-v3-lora (best ckpt
revision per model card) on our held-out SADA Saudi sample.

Same clips, normalization, and degenerate-case logic as the zero-shot baseline
(spikes/sada_whisper_eval.py + sada_reanalyze.py), so numbers are directly
comparable. Prints a side-by-side table vs the stored zero-shot results.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

import sawti.env  # noqa: F401
from sawti.text_normalize import normalize_arabic_for_match

DATA = Path("data/sada_spike")
ADAPTER_ID = "dev-ahmedhany/whisper-large-v3-arabic-ft-v3-lora"
ADAPTER_REV = "7923fe7bc9b7"  # best ckpt-4750 per model card
BASE_ID = "openai/whisper-large-v3"


def norm(text: str) -> str:
    import re

    t = normalize_arabic_for_match(text)
    t = re.sub(r"[^\w\s\u0600-\u06FF]", " ", t)
    return " ".join(t.split())


def is_loop(hyp: str) -> bool:
    toks = hyp.split()
    if len(toks) < 6:
        return False
    uniq = len(set(toks)) / len(toks)
    most = Counter(toks).most_common(1)[0][1] / len(toks)
    return uniq < 0.25 or most > 0.6


def main() -> None:
    import jiwer
    import torch
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline

    manifest = [
        json.loads(line)
        for line in (DATA / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    zero = {
        r["clip_id"]: r
        for r in json.loads((DATA / "eval_results.json").read_text(encoding="utf-8"))
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | base: {BASE_ID} | adapter: {ADAPTER_ID}@{ADAPTER_REV}")

    base = WhisperForConditionalGeneration.from_pretrained(
        BASE_ID, dtype=torch.float16
    )
    model = PeftModel.from_pretrained(base, ADAPTER_ID, revision=ADAPTER_REV)
    processor = WhisperProcessor.from_pretrained(BASE_ID)
    model = model.merge_and_unload()  # fold adapter into base for pipeline use
    model.to(device)
    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.float16,
        device=0 if device == "cuda" else -1,
        chunk_length_s=30,
    )

    results = []
    for m in manifest:
        wav = str(DATA / f"{m['clip_id']}.wav")
        hyp_text = asr(
            wav, generate_kwargs={"language": "arabic", "task": "transcribe"}
        )["text"].strip()
        ref_text = (m.get("cleaned_text") or m.get("text") or "").strip()
        wer = jiwer.wer(norm(ref_text), norm(hyp_text)) if norm(ref_text) else None
        degenerate = m["duration_s"] < 1.0 or is_loop(hyp_text) or wer is None
        results.append({**m, "hyp": hyp_text, "wer": wer, "degenerate": degenerate})

    (DATA / "adapter_eval_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    # Side-by-side per dialect: zero-shot (stored) vs adapter (this run).
    print("\n=== Saudi-dialect clean-subset WER: zero-shot vs Arabic-ft adapter ===")
    by_d = defaultdict(list)
    for r in results:
        by_d[r["dialect"]].append(r)
    z_all_c, a_all_c = [], []
    for d, rows in sorted(by_d.items()):
        z_clean = [
            zero[r["clip_id"]]["wer"]
            for r in rows
            if not (
                r["duration_s"] < 1.0
                or is_loop(zero[r["clip_id"]]["hyp"])
                or zero[r["clip_id"]]["wer"] is None
            )
        ]
        a_clean = [r["wer"] for r in rows if not r["degenerate"]]
        z_all_c.extend(z_clean)
        a_all_c.extend(a_clean)
        zw = 100 * np.mean(z_clean) if z_clean else float("nan")
        aw = 100 * np.mean(a_clean) if a_clean else float("nan")
        a_deg = sum(r["degenerate"] for r in rows)
        print(
            f"  {d:10s} n={len(rows):2d}  zero {zw:5.1f}% -> adapter {aw:5.1f}%  "
            f"({aw - zw:+.1f} pp)  | adapter-degenerate {a_deg}"
        )
    zw, aw = 100 * np.mean(z_all_c), 100 * np.mean(a_all_c)
    print(
        f"  {'ALL':10s} n={len(results):2d}  zero {zw:5.1f}% -> adapter {aw:5.1f}%  "
        f"({aw - zw:+.1f} pp)  | adapter-degenerate "
        f"{sum(r['degenerate'] for r in results)}"
    )

    print("\n=== 3 biggest improvements ===")
    def delta(r):
        z = zero[r["clip_id"]].get("wer")
        return (r["wer"] - z) if (z is not None and r["wer"] is not None) else 0
    for r in sorted(results, key=delta)[:3]:
        print(f"[{r['clip_id']}] zero {zero[r['clip_id']].get('wer')} -> {r['wer']}")
        print(f"  hyp: {r['hyp'][:120]}")
    print("=== 3 biggest regressions ===")
    for r in sorted(results, key=delta, reverse=True)[:3]:
        print(f"[{r['clip_id']}] zero {zero[r['clip_id']].get('wer')} -> {r['wer']}")
        print(f"  hyp: {r['hyp'][:120]}")


if __name__ == "__main__":
    main()
