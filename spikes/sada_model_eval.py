"""Spike: generalized Saudi-sample ASR eval.

Evaluates any of: a full HF model repo, or a PEFT adapter on a base model,
on the 75-clip SADA Saudi sample with identical normalization/loop logic to
the original baseline. Prints side-by-side vs the stored zero-shot
large-v3 baseline.

Usage:
  uv run python spikes/sada_model_eval.py full  <model_id> [revision]
  uv run python spikes/sada_model_eval.py adapter <base_id> <adapter_id> [rev]
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import sawti.env  # noqa: F401
from sawti.text_normalize import normalize_arabic_for_match

DATA = Path("data/sada_spike")


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
    mode = sys.argv[1]
    import jiwer
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline

    label = ""
    if mode == "full":
        model_id, revision = sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else None)
        # Optional 4th arg: processor source, for repos with broken tokenizer
        # configs (e.g. oddadmix's extra_special_tokens list-vs-dict quirk).
        # A full Whisper fine-tune keeps the base tokenizer.
        proc_id = sys.argv[4] if len(sys.argv) > 4 else model_id
        label = model_id.split("/")[-1] + (f"@{revision[:7]}" if revision else "")
        model = WhisperForConditionalGeneration.from_pretrained(
            model_id, revision=revision, dtype=torch.float16
        )
        processor = WhisperProcessor.from_pretrained(proc_id)
    elif mode == "adapter":
        base_id, adapter_id = sys.argv[2], sys.argv[3]
        revision = sys.argv[4] if len(sys.argv) > 4 else None
        from peft import PeftModel

        label = adapter_id.split("/")[-1] + (f"@{revision[:7]}" if revision else "")
        base = WhisperForConditionalGeneration.from_pretrained(
            base_id, dtype=torch.float16
        )
        model = PeftModel.from_pretrained(base, adapter_id, revision=revision)
        model = model.merge_and_unload()
        processor = WhisperProcessor.from_pretrained(base_id)
    else:
        raise SystemExit(f"unknown mode {mode}")

    manifest = [
        json.loads(line)
        for line in (DATA / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    zero = {
        r["clip_id"]: r
        for r in json.loads((DATA / "eval_results.json").read_text(encoding="utf-8"))
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"eval: {label} | device: {device}")
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

    slug = label.replace("/", "_").replace("@", "_")
    (DATA / f"eval_{slug}.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    print(f"\n=== {label}: clean-subset WER vs zero-shot large-v3 ===")
    by_d = defaultdict(list)
    for r in results:
        by_d[r["dialect"]].append(r)
    z_all, a_all = [], []
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
        z_all.extend(z_clean)
        a_all.extend(a_clean)
        zw = 100 * np.mean(z_clean) if z_clean else float("nan")
        aw = 100 * np.mean(a_clean) if a_clean else float("nan")
        print(
            f"  {d:10s} n={len(rows):2d}  zero(l-v3) {zw:5.1f}% -> {aw:5.1f}% "
            f"({aw - zw:+.1f} pp) | degenerate {sum(r['degenerate'] for r in rows)}"
        )
    zw, aw = 100 * np.mean(z_all), 100 * np.mean(a_all)
    print(
        f"  {'ALL':10s} n={len(results):2d}  zero(l-v3) {zw:5.1f}% -> {aw:5.1f}% "
        f"({aw - zw:+.1f} pp) | degenerate {sum(r['degenerate'] for r in results)}"
    )


if __name__ == "__main__":
    main()
