# SA — Saudi ASR Milestone Implementation Plan

> **RECONCILED 2026-08-18 against the merged M1 baseline (PR #3):**
> M1 is the immutable runtime baseline. SA does NOT modify Pipeline,
> fallback ordering, segmentation, or the generic gate. Task 1 imports
> the SHARED production loop detector (no dominance-heuristic fork).
> Task 3 derives augmentation RNG per-sample (deterministic). Tasks 8-10
> specialize the ASR component via the existing provider/builder
> architecture instead of creating a parallel orchestration path.

> **For agentic workers:** Execute this plan task-by-task with fresh subagents per task, two-stage review on Tasks 9–11. Steps use checkbox (`- [ ]`) syntax. Tasks marked **OPERATOR** are run by the human/controller directly (long GPU jobs or large downloads), not by implementation subagents.

**Goal:** Train a Saudi-dialect Whisper-large-v3 QLoRA on SADA (Saudi-filtered), export a merged model, and integrate it into Sawti as the `AsrMtProvider` fallback component plus an `--engine sawti-sa` mode.

**Architecture:** Training lives in `sawti/training/` (never imported by runtime). Integration adds two injectable components (`asr_whisper_sa.py`, `mt_m4t.py`) behind existing Protocols. AMENDED 2026-08-18 (re-review): the spec-gap repair milestone wires `FallbackHandler` into `Pipeline` FIRST (`pipeline.py` is un-frozen there per the approved architectural repair); SA targets the repaired orchestrator. Checkpoint selection on held-out Saudi dev — clean macro WER gated by n-gram loop-rate; all-valid macro AND corpus WER reported alongside (Addendum 4 metric set). Training includes required audio augmentation (Task 3).

**Tech Stack:** transformers 4.57.x (pinned <5), peft, torch+cu126 (local overlay), datasets (streaming), jiwer, soundfile.

**Spec:** `docs/superpowers/specs/2026-08-17-saudi-asr-milestone-design.md`
**Evidence:** `docs/superpowers/research/2026-08-17-sada-whisper-baseline.md` (+2 addenda)

---

## Critical constraints (read first)

1. **Frozen (amended):** `sawti/types.py`, `sawti/config.py` are never modified. `pipeline.py` is modified by the spec-gap fallback repair which lands BEFORE this plan's integration tasks; SA itself adds no orchestrator changes. Integration adds files and edits `cli.py`.
2. **Hermetic unit tests:** no model loads, downloads, CUDA, or network. Real-model tests carry `@pytest.mark.integration` (auto-skipped unless `SAWTI_RUN_INTEGRATION=1`).
3. **GPU overlay stays unstaged:** the local cu126 block at the end of `pyproject.toml` must never be committed. Before any commit that stages `pyproject.toml`/`uv.lock`: strip the overlay, `uv lock`, stage, commit, re-append overlay locally (the established pattern from PR #2).
4. **Branch/PR:** all work on `feat/sa-milestone` → PR → squash-merge. No direct pushes to `main` (protected).
5. **Data license:** SADA is CC BY-NC-SA — research use only; merged model inherits this (stated in its model card).
6. **Training deps are dev-group only.**

## File structure

```
sawti/
├── asr_whisper_sa.py          # Task 9: SaudiWhisperAsr (AsrMtProvider impl)
├── mt_m4t.py                  # Task 8: M4TTtTranslator (T2TT wrapper)
├── cli.py                     # Task 10: + --engine sawti-sa (modify)
├── training/
│   ├── __init__.py
│   ├── eval_utils.py          # Task 1: norm/is_loop/run_eval/aggregate (shared)
│   ├── data_prep.py           # Task 2: census + filter + materialize
│   ├── dataset.py             # Task 3: SadaDataset + WhisperCollator
│   ├── train_qlora.py         # Tasks 4-5: config + Trainer + DevEvalCallback
│   └── export_merge.py        # Task 6: merge best adapter -> model dir
├── build_sa.py                # Task 10: build_sawti_sa_pipeline (injectable)
tests/
├── test_eval_utils.py         # Task 1
├── test_data_prep.py          # Task 2
├── test_dataset.py            # Task 3
├── test_train_qlora.py        # Tasks 4-5
├── test_export_merge.py       # Task 6
├── test_mt_m4t.py             # Task 8
├── test_asr_whisper_sa.py     # Task 9
├── test_build_sa.py           # Task 10
└── test_sa_integration.py     # Task 11 (opt-in)
data/sada_training/            # gitignored: train/, val/, census.json
data/sada_spike/               # exists (75-clip dev set + eval_results.json)
```

---

## Task 0: Training dependencies + bitsandbytes Windows probe

**Files:** Modify `pyproject.toml` (dev group), `uv.lock`.

- [ ] **Step 1:** Add dev deps: `uv add --dev bitsandbytes` (peft/datasets/jiwer already added in the spike PR).
- [ ] **Step 2:** Probe bnb on this machine:
  ```bash
  uv run python -c "
import torch, bitsandbytes as bnb
from torch import nn
lin = nn.Linear(64, 64).cuda()
q = bnb.nn.Linear4bit(64, 64).cuda()
opt = bnb.optim.PagedAdamW8bit(q.parameters(), lr=1e-4)
print('bnb OK:', bnb.__version__)
"
  ```
- [ ] **Step 3 — decision (encode result in `sawti/training/__init__.py` docstring):**
  - **If probe passes:** training uses QLoRA NF4 + `optim="paged_adamw_8bit"`, per-device batch 8 × accum 2.
  - **If probe fails (Windows wheel/functional issue):** training uses plain LoRA fp16 (no 4-bit), `optim="adamw_torch_fused"`, per-device batch 4 × accum 4, `gradient_checkpointing=True`. Both branches are fully supported by Tasks 4–5; the flag is `TRAIN_FLAVOR` in `train_qlora.py`.
- [ ] **Step 4:** Commit (overlay-strip pattern from constraint 3):
  ```bash
  git checkout -b feat/sa-milestone
  git add pyproject.toml uv.lock && git commit -m "build(sa): add bitsandbytes dev dep; record TRAIN_FLAVOR probe result"
  git push -u origin feat/sa-milestone
  ```

---

## Task 1: `sawti/training/eval_utils.py` — shared eval logic

Extracted from the spike scripts so the training callback and final eval use identical logic.

**Files:** Create `sawti/training/__init__.py` (empty), `sawti/training/eval_utils.py`; Test `tests/test_eval_utils.py`.

- [ ] **Step 1: Failing test** — `tests/test_eval_utils.py`:
```python
import json
from pathlib import Path

import numpy as np
import pytest

from sawti.training.eval_utils import (aggregate, annotate_degenerate,
                                    is_loop, norm, wer_clean)
# is_loop is re-exported from sawti.loop_detect (shared production module)


def test_norm_unifies_arabic_and_strips_punct():
    assert norm("مَرْحَباً ، World!") == "مرحبا world"
    assert norm("أحمد  إبراهيم") == "احمد ابراهيم"


def test_shared_detector_pinned_through_eval_utils():
    # The shared M1 detector, imported via eval_utils: phrase loops catch,
    # frequency alone never gates (the dominance fork is gone).
    assert is_loop("اشتركوا في القناه " * 3) is True
    assert is_loop("لا " * 12) is True
    assert is_loop("no no wait no no stop no no listen") is False


def test_wer_clean_basic():
    assert wer_clean("احمد ذهب", "احمد ذهب") == 0.0
    # 1 insertion over 1 reference word = WER 1.0 (jiwer convention).
    assert wer_clean("احمد", "احمد ذهب") == pytest.approx(1.0)


def test_annotate_degenerate_sets_metric_fields():
    rows = [
        {"dialect": "Najdi", "duration_s": 5.0, "cleaned_text": "مرحبا بك",
         "hyp": "مرحبا بك", "wer": 0.0},
        {"dialect": "Najdi", "duration_s": 5.0, "cleaned_text": "كلمة",
         "hyp": "اشتركوا في القناه اشتركوا في القناه اشتركوا في القناه", "wer": 9.0},
        {"dialect": "Hijazi", "duration_s": 0.5, "cleaned_text": "نعم",
         "hyp": "نعم", "wer": 0.0},
    ]
    out = annotate_degenerate(rows)
    assert out[0]["loop"] is False and out[0]["valid_ref"] is True
    assert out[0]["n_ref_words"] == 2
    assert out[1]["loop"] is True and out[1]["degenerate"] is True  # phrase loop
    assert out[2]["degenerate"] is True  # short clip


def test_aggregate_exposes_full_metric_set():
    rows = [
        {"dialect": "Najdi", "duration_s": 5.0, "cleaned_text": "مرحبا بك",
         "hyp": "مرحبا بك", "wer": 0.0, "loop": False, "degenerate": False,
         "valid_ref": True, "n_ref_words": 2},
        {"dialect": "Najdi", "duration_s": 5.0, "cleaned_text": "كلمة واحدة",
         "hyp": "كلمة", "wer": 0.5, "loop": False, "degenerate": False,
         "valid_ref": True, "n_ref_words": 2},
        {"dialect": "Hijazi", "duration_s": 5.0, "cleaned_text": "كلمة",
         "hyp": "اشتركوا في القناه " * 3, "wer": 9.0, "loop": True,
         "degenerate": True, "valid_ref": True, "n_ref_words": 1},
    ]
    out = aggregate(rows)
    assert out["clean_macro_wer"] == pytest.approx(25.0)          # (0 + 0.5) / 2
    assert out["all_valid_macro_wer"] == pytest.approx(100 * (0 + 0.5 + 9) / 3)
    assert out["all_valid_corpus_wer"] == pytest.approx(100 * (0 * 2 + 0.5 * 2 + 9 * 1) / 5)
    assert out["loop_pct"] == pytest.approx(100 / 3)
    assert out["per_dialect"]["Najdi"]["clean_macro_wer"] == pytest.approx(25.0)
    assert out["per_dialect"]["Hijazi"]["n_clean"] == 0
```
- [ ] **Step 2:** `uv run pytest tests/test_eval_utils.py -v` → FAIL (ModuleNotFoundError).
- [ ] **Step 3: Implement** — `sawti/training/eval_utils.py`:
```python
"""Shared Saudi-eval logic (normalization, loop detection, aggregation).

Used by the training dev-eval callback and the final test-slice eval so both
produce identical numbers. Mirrors the spike scripts exactly.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from sawti.text_normalize import normalize_arabic_for_match


def norm(text: str) -> str:
    t = normalize_arabic_for_match(text)
    t = re.sub(r"[^\w\s\u0600-\u06FF]", " ", t)
    return " ".join(t.split())


# THE detector is the shared production implementation (M1 PR #3):
# pure consecutive-block semantics — lexical frequency/dominance NEVER
# gates. Importing it here guarantees training-time selection, the
# evaluator, and the runtime gate agree by construction.
from sawti.loop_detect import is_loop


def wer_clean(ref: str, hyp: str) -> float:
    import jiwer

    n_ref, n_hyp = norm(ref), norm(hyp)
    if not n_ref:
        return float("nan")
    return jiwer.wer(n_ref, n_hyp)


def load_manifest(data_dir: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (Path(data_dir) / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def annotate_degenerate(rows: list[dict]) -> list[dict]:
    """Sets every field aggregate() needs: loop flag (n-gram), degenerate,
    valid_ref, and reference word count (for corpus WER)."""
    for r in rows:
        ref = (r.get("cleaned_text") or r.get("text") or "").strip()
        n_ref = norm(ref)
        r["loop"] = is_loop(r.get("hyp", ""))
        r["degenerate"] = (
            r.get("duration_s", 99) < 1.0 or r["loop"] or r.get("wer") is None
        )
        r["valid_ref"] = bool(n_ref)
        r["n_ref_words"] = len(n_ref.split())
    return rows


def _macro(rows: list[dict]) -> float:
    w = [r["wer"] for r in rows if r.get("wer") is not None]
    return 100 * float(np.mean(w)) if w else float("nan")


def _corpus(rows: list[dict]) -> float:
    """Corpus WER over valid-reference rows: per-clip WER weighted by
    reference word count (exact from stored values)."""
    err = total = 0.0
    for r in rows:
        if r.get("wer") is None:
            continue
        err += r["wer"] * r["n_ref_words"]
        total += r["n_ref_words"]
    return 100 * err / total if total else float("nan")


def aggregate(rows: list[dict]) -> dict:
    """Full Addendum-4 metric set: clean macro WER (checkpoint gate),
    n-gram loop-rate (eligibility constraint), all-valid macro AND corpus
    (robustness views), per-dialect clean metrics."""
    by_d: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_d[r["dialect"]].append(r)
    per = {}
    for d, rs in by_d.items():
        clean = [r for r in rs if not r["degenerate"]]
        per[d] = {"clean_macro_wer": _macro(clean), "n_clean": len(clean)}
    valid = [r for r in rows if r.get("valid_ref")]
    clean_all = [r for r in rows if not r["degenerate"]]
    return {
        "clean_macro_wer": _macro(clean_all),
        "all_valid_macro_wer": _macro(valid),
        "all_valid_corpus_wer": _corpus(valid),
        "loop_pct": 100 * sum(bool(r.get("loop")) for r in rows) / max(1, len(rows)),
        "degenerate_rate": 100 * sum(bool(r["degenerate"]) for r in rows) / max(1, len(rows)),
        "per_dialect": per,
        "n": len(rows),
    }


def run_eval(asr_fn, data_dir: str | Path) -> list[dict]:
    """asr_fn(wav_path) -> str. Returns annotated rows (writes nothing)."""
    rows = []
    for m in load_manifest(data_dir):
        hyp = asr_fn(str(Path(data_dir) / f"{m['clip_id']}.wav"))
        ref = (m.get("cleaned_text") or m.get("text") or "").strip()
        rows.append({**m, "hyp": hyp, "wer": wer_clean(ref, hyp)})
    return annotate_degenerate(rows)

```
- [ ] **Step 4:** `uv run pytest tests/test_eval_utils.py -v` → 6 PASS (incl. phrase-loop regression + full metric-set contract). Full suite green.
- [ ] **Step 5:** Commit: `feat(training): shared Saudi eval utils (norm/loop/aggregate)`.

---

## Task 2: `sawti/training/data_prep.py` — census + filter + materialize

**Files:** Create `sawti/training/data_prep.py`; Test `tests/test_data_prep.py`; outputs `data/sada_training/` (gitignored).

- [ ] **Step 1: Failing test** — `tests/test_data_prep.py`:
```python
from sawti.training.data_prep import census_labels, keep_row, saudi_label_set


def test_census_labels_counts():
    rows = [{"speaker_dialect": "Najdi"}, {"speaker_dialect": "Najdi"},
            {"speaker_dialect": "MSA"}, {"speaker_dialect": None}]
    assert census_labels(rows) == {"Najdi": 2, "MSA": 1, "unknown": 1}


def test_saudi_label_set_from_census():
    census = {"Najdi": 10, "Hijazi": 5, "Khaliji": 5, "MSA": 99, "Yemeni": 3}
    # Saudi core confirmed by the spike; census adds any other Saudi-specific
    # labels it finds via the SUFFIX probe below.
    assert saudi_label_set(census, extra=[]) == {"Najdi", "Hijazi", "Khaliji"}
    assert saudi_label_set(census, extra=["Southern Saudi"]) == {
        "Najdi", "Hijazi", "Khaliji", "Southern Saudi"}


def test_keep_row_filters_duration_and_text():
    base = {"duration_s": 5.0, "cleaned_text": "مرحبا", "speaker_dialect": "Najdi"}
    assert keep_row(base, {"Najdi"}) is True
    assert keep_row({**base, "duration_s": 45.0}, {"Najdi"}) is False  # >30s
    assert keep_row({**base, "duration_s": 0.3}, {"Najdi"}) is False  # <0.5s
    assert keep_row({**base, "cleaned_text": "  "}, {"Najdi"}) is False
    assert keep_row(base, {"Hijazi"}) is False  # dialect not selected
```
- [ ] **Step 2:** FAIL check, then **implement** `sawti/training/data_prep.py`:
```python
"""SADA data prep: dialect census, Saudi filter, materialization.

Usage (OPERATOR):
  uv run python -m sawti.training.data_prep --split train --out data/sada_training/train
  uv run python -m sawti.training.data_prep --split validation --out data/sada_training/val

Streams the split (no full 50GB download), decodes audio bytes with
soundfile (torchcodec is Windows-hostile), writes wav + manifest.jsonl +
census.json. Saudi labels: spike-confirmed core + any census labels the
operator adds via --extra-label after inspecting census.json.
"""
from __future__ import annotations

import argparse
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

# (No env loading here: tests/conftest.py owns the test-runner edge.)

CORE = ["Najdi", "Hijazi", "Khaliji"]
MAX_S = 30.0   # Whisper native window; longer clips are dropped (logged)
MIN_S = 0.5


def census_labels(rows) -> dict[str, int]:
    c = Counter()
    for r in rows:
        c[r.get("speaker_dialect") or "unknown"] += 1
    return dict(c)


def saudi_label_set(census: dict[str, int], extra: list[str]) -> set[str]:
    return set(CORE) | set(extra)


def keep_row(row: dict, labels: set[str]) -> bool:
    if row.get("speaker_dialect") not in labels:
        return False
    if not (MIN_S <= float(row.get("duration_s", 0)) <= MAX_S):
        return False
    return bool((row.get("cleaned_text") or row.get("text") or "").strip())


def _decode(audio_entry: dict):
    arr, sr = sf.read(io.BytesIO(audio_entry["bytes"]), dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return arr, sr


def materialize(split: str, out: str, extra_labels: list[str], cap: int | None) -> dict:
    from datasets import Audio, load_dataset

    ds = load_dataset("MohamedRashad/SADA22", split=split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    out_p = Path(out)
    out_p.mkdir(parents=True, exist_ok=True)
    dropped = Counter()
    manifest, kept_hours, seen = [], 0.0, 0
    labels = saudi_label_set({}, extra_labels)
    for row in ds:
        seen += 1
        arr, sr = _decode(row["audio"])
        rec = {
            "speaker_dialect": row.get("speaker_dialect"),
            "cleaned_text": row.get("cleaned_text") or "",
            "text": row.get("text") or "",
            "duration_s": round(len(arr) / sr, 2),
        }
        if not keep_row(rec, labels):
            if rec["speaker_dialect"] in labels:
                dropped["duration_or_text"] += 1
            else:
                dropped["dialect"] += 1
            continue
        clip_id = f"{split}_{len(manifest):06d}"
        sf.write(out_p / f"{clip_id}.wav", arr, sr)
        manifest.append(
            {"clip_id": clip_id, "dialect": rec["speaker_dialect"],
             "cleaned_text": rec["cleaned_text"], "text": rec["text"],
             "duration_s": rec["duration_s"],
             "speaker_gender": row.get("speaker_gender", ""),
             "speaker_age": row.get("speaker_age", "")})
        kept_hours += rec["duration_s"] / 3600
        if cap and len(manifest) >= cap:
            break
    stats = {
        "split": split, "scanned": seen, "kept": len(manifest),
        "kept_hours": round(kept_hours, 2), "dropped": dict(dropped),
        "labels": sorted(labels),
        "dialect_counts": dict(Counter(m["dialect"] for m in manifest)),
    }
    (out_p / "manifest.jsonl").write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in manifest), encoding="utf-8")
    (out_p / "census.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    return stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, choices=["train", "validation", "test"])
    p.add_argument("--out", required=True)
    p.add_argument("--extra-label", action="append", default=[])
    p.add_argument("--cap", type=int, default=None)
    a = p.parse_args()
    stats = materialize(a.split, a.out, a.extra_label, a.cap)
    print(json.dumps(stats, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
```
- [ ] **Step 3:** `uv run pytest tests/test_data_prep.py -v` → 3 PASS.
- [ ] **Step 4 — OPERATOR (validation split first, small):**
  ```bash
  uv run python -m sawti.training.data_prep --split validation --out data/sada_training/val --cap 200
  ```
  Inspect `census.json` — note the label inventory and any additional
  Saudi-specific labels; if found, re-run with `--extra-label "<name>"`.
- [ ] **Step 5 — OPERATOR (train split, full Saudi filter; ~20–30GB, hours):**
  ```bash
  uv run python -m sawti.training.data_prep --split train --out data/sada_training/train
  ```
  Expected: `kept_hours` in the hundreds; `census.json` records dialect
  counts. If disk pressure appears, add `--cap 150000`.
- [ ] **Step 6:** Commit code only (data is gitignored): `feat(training): SADA data prep with census + Saudi filter`.

---

## Task 3: `sawti/training/dataset.py` — dataset + collator + augmentation (spec §2.6)

**Files:** Create `sawti/training/dataset.py`; Test `tests/test_dataset.py`.

- [ ] **Step 1: Failing test** — `tests/test_dataset.py`:
```python
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sawti.training.dataset import SadaDataset, WhisperCollator


class FakeTokenizer:
    """Callable (the collator invokes tok(...)), with bos_token_id."""

    bos_token_id = 1

    def __call__(self, texts, padding=True, truncation=True,
                 max_length=448, return_tensors="pt"):
        ids = torch.tensor([[1, 5, 6], [1, 5, 0]])
        am = torch.tensor([[1, 1, 1], [1, 1, 0]])

        class B:
            input_ids = ids
            attention_mask = am
        return B()


class FakeProcessor:
    tokenizer = FakeTokenizer()

    def __call__(self, audio, sampling_rate=16000, return_tensors="pt"):
        class F:
            input_features = torch.zeros(len(audio), 80, 3000)
            attention_mask = torch.ones(len(audio), 3000, dtype=torch.long)
        return F()


def _mk(tmp_path: Path, n=2):
    import soundfile as sf

    for i in range(n):
        sf.write(tmp_path / f"c{i}.wav", np.zeros(8000, np.float32), 16000)
    rows = [{"clip_id": f"c{i}", "cleaned_text": "مرحبا", "text": "مرحبا"}
            for i in range(n)]
    (tmp_path / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


def test_dataset_returns_audio_and_text(tmp_path):
    _mk(tmp_path)
    ds = SadaDataset(str(tmp_path))
    item = ds[0]
    assert item["audio"].dtype == np.float32
    assert item["text"] == "مرحبا"


def test_collator_shapes_and_masking(tmp_path):
    _mk(tmp_path)
    ds = SadaDataset(str(tmp_path))
    batch = WhisperCollator(FakeProcessor())([ds[0], ds[1]])
    assert batch["input_features"].shape[0] == 2
    assert batch["labels"].shape[0] == 2
    # bos stripped; padded positions are -100
    assert (batch["labels"] == -100).any()


def test_augment_deterministic_per_sample_and_epoch():
    from sawti.training.dataset import augment

    audio = np.ones(16000, np.float32) * 0.5
    a1 = augment(audio, np.random.default_rng([42, 0, 3]))
    a2 = augment(audio, np.random.default_rng([42, 0, 3]))
    a3 = augment(audio, np.random.default_rng([42, 1, 3]))  # next epoch
    assert np.array_equal(a1, a2)      # same (seed, epoch, index)
    assert not np.array_equal(a1, a3)  # epoch advances the stream
    assert a1.dtype == np.float32
    assert float(np.std(a1)) > 1e-4    # noise actually added
```
- [ ] **Step 2:** FAIL, then **implement**:
```python
"""Whisper training dataset + collator for materialized SADA shards."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def augment(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Deterministic-seeded augmentation (spec §2.6): random gain, additive
    gaussian noise at ~15-30 dB SNR, speed perturbation {0.9, 1.0, 1.1} via
    linear resampling. The augmentation class plausibly behind oddadmix's
    zero-loop robustness (Addendum 4). Music/reverb: deferred extension."""
    out = audio.copy()
    out *= float(rng.uniform(0.5, 1.0))
    p_signal = float(np.mean(out ** 2)) + 1e-12
    snr_db = float(rng.uniform(15.0, 30.0))
    p_noise = p_signal / (10 ** (snr_db / 10))
    out = out + rng.normal(0.0, p_noise ** 0.5, size=out.shape).astype(np.float32)
    speed = float(rng.choice([0.9, 1.0, 1.1]))
    if speed != 1.0:
        n_out = max(1, int(len(out) / speed))
        out = np.interp(
            np.linspace(0.0, len(out) - 1, n_out),
            np.arange(len(out)), out).astype(np.float32)
    return out.astype(np.float32)


class SadaDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str | Path, max_text_len: int = 448,
                 augment_enabled: bool = False, seed: int = 0,
                 epoch: int = 0) -> None:
        self.dir = Path(data_dir)
        self.rows = [
            json.loads(line)
            for line in (self.dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.max_text_len = max_text_len
        self.augment_enabled = augment_enabled
        self.seed = seed
        self.epoch = epoch

    def set_epoch(self, epoch: int) -> None:
        """Advance the augmentation stream deterministically per epoch."""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        audio, sr = sf.read(self.dir / f"{r['clip_id']}.wav", dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if self.augment_enabled:
            # Deterministic PER SAMPLE (and epoch): no shared mutable RNG,
            # so dataloader worker ordering cannot change augmentation.
            rng = np.random.default_rng([self.seed, self.epoch, i])
            audio = augment(audio, rng)
        text = (r.get("cleaned_text") or r.get("text") or "").strip()[: self.max_text_len]
        return {"audio": np.ascontiguousarray(audio, np.float32), "text": text}


class WhisperCollator:
    """Standard Whisper fine-tuning collator: 30s-padded features + masked labels."""

    def __init__(self, processor) -> None:
        self.processor = processor

    def __call__(self, features: list[dict]) -> dict:
        inputs = self.processor(
            [f["audio"] for f in features], sampling_rate=16000, return_tensors="pt"
        )
        tok = self.processor.tokenizer
        batch = tok(
            [f["text"] for f in features], padding=True,
            truncation=True, max_length=448, return_tensors="pt",
        )
        labels = batch["input_ids"].masked_fill(batch["attention_mask"].ne(1), -100)
        if (labels[:, 0] == tok.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        out = dict(input_features=inputs.input_features)
        if hasattr(inputs, "attention_mask"):
            out["attention_mask"] = inputs.attention_mask
        out["labels"] = labels
        return out
```
- [ ] **Step 3:** `uv run pytest tests/test_dataset.py -v` → 3 PASS (incl. augmentation). Commit `feat(training): SadaDataset + collator + required augmentation`.

---

## Task 4: `sawti/training/train_qlora.py` — config + Trainer factory

**Files:** Create `sawti/training/train_qlora.py`; Test `tests/test_train_qlora.py`.

- [ ] **Step 1: Failing test** — `tests/test_train_qlora.py`:
```python
from sawti.training.train_qlora import build_lora_config, build_training_args


def test_lora_config_matches_spec_recipe():
    cfg = build_lora_config()
    assert cfg.r == 8 and cfg.lora_alpha == 16 and cfg.lora_dropout == 0.05
    mods = set(cfg.target_modules)
    assert {"q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"} <= mods


def test_training_args_qlora_flavor():
    a = build_training_args(".", flavor="qlora")
    assert a.optim == "paged_adamw_8bit"
    assert a.per_device_train_batch_size * a.gradient_accumulation_steps == 16
    assert a.learning_rate == 1e-4 and a.warmup_ratio == 0.1


def test_training_args_lora_fallback_flavor():
    a = build_training_args(".", flavor="lora")
    assert a.optim == "adamw_torch_fused"
    assert a.per_device_train_batch_size * a.gradient_accumulation_steps == 16
```
- [ ] **Step 2:** FAIL, then **implement** `sawti/training/train_qlora.py` (factory half; the run half lands with Task 5):
```python
"""QLoRA training for Saudi Whisper (spec §2).

Run (OPERATOR):
  uv run python -m sawti.training.train_qlora \
    --train data/sada_training/train --val data/sada_training/val \
    --dev data/sada_spike --out checkpoints/sa_qlora

TRAIN_FLAVOR is set by the Task 0 bitsandbytes probe:
  qlora -> NF4 4-bit + paged_adamw_8bit, batch 8 x accum 2
  lora  -> fp16 LoRA + adamw_torch_fused, batch 4 x accum 4
"""
from __future__ import annotations

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]


def build_lora_config():
    from peft import LoraConfig

    return LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, target_modules=LORA_TARGETS,
    )


def build_training_args(out_dir: str, flavor: str = "qlora", max_steps: int = 10000):
    from transformers import TrainingArguments

    if flavor == "qlora":
        optim, bs, accum = "paged_adamw_8bit", 8, 2
    elif flavor == "lora":
        optim, bs, accum = "adamw_torch_fused", 4, 4
    else:
        raise ValueError(f"unknown flavor {flavor}")
    return TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=accum,
        learning_rate=1e-4,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        max_steps=max_steps,
        logging_steps=50,
        save_steps=500,
        save_total_limit=4,
        eval_strategy="no",
        optim=optim,
        gradient_checkpointing=True,
        per_device_eval_batch_size=8,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=[],
        save_safetensors=True,
    )
```
- [ ] **Step 3:** `uv run pytest tests/test_train_qlora.py -v` → 3 PASS. Commit `feat(training): QLoRA config + TrainingArguments factory (spec recipe)`.

---

## Task 5: dev-eval callback + `main()` training entry

**Files:** Modify `sawti/training/train_qlora.py`; Test `tests/test_train_qlora.py` (append).

- [ ] **Step 1: Failing tests (append)** — `tests/test_train_qlora.py`:
```python
class _Ctl:
    def __init__(self):
        self.should_training_stop = False


def _metrics(wer, loop=0.0):
    return {"clean_macro_wer": wer, "all_valid_macro_wer": wer * 2.0,
            "all_valid_corpus_wer": wer, "loop_pct": loop,
            "degenerate_rate": loop, "per_dialect": {}, "n": 59}


def test_dev_callback_best_tracking_and_stop_after_3_regressions(tmp_path):
    from sawti.training.train_qlora import DevEvalCallback

    seq = iter([_metrics(40.0), _metrics(30.0), _metrics(35.0),
                _metrics(36.0), _metrics(37.0)])
    cb = DevEvalCallback(eval_fn=lambda m: next(seq),
                         log_path=str(tmp_path / "dev_log.jsonl"), patience=3)
    for _ in range(5):
        cb.on_save(args=None, state=None, control=_Ctl(), model=None)
    log = [__import__("json").loads(l) for l in
           (tmp_path / "dev_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert log[1]["clean_macro_wer"] == 30.0 and log[1]["is_best"] is True
    assert log[1]["all_valid_macro_wer"] == 60.0  # all four metrics logged
    stops = [l for l in log if l.get("stop")]
    assert stops and stops[0]["eval_index"] == 5  # regressions at evals 3,4,5


def test_dev_callback_loop_constraint_blocks_ineligible_best(tmp_path):
    from sawti.training.train_qlora import DevEvalCallback

    seq = iter([_metrics(40.0, loop=0.0), _metrics(20.0, loop=9.0),
                _metrics(35.0, loop=0.0)])
    cb = DevEvalCallback(eval_fn=lambda m: next(seq),
                         log_path=str(tmp_path / "dev_log.jsonl"), patience=3)
    for _ in range(3):
        cb.on_save(args=None, state=None, control=_Ctl(), model=None)
    log = [__import__("json").loads(l) for l in
           (tmp_path / "dev_log.jsonl").read_text(encoding="utf-8").splitlines()]
    # eval 2: best raw WER (20.0) but loop 9% > limit -> ineligible, not best
    assert log[1]["eligible"] is False and log[1]["is_best"] is False
    # eval 3: eligible and better than the standing best (40.0) -> new best
    assert log[2]["is_best"] is True and log[2]["best_clean_macro_wer"] == 35.0
```
- [ ] **Step 2:** FAIL, then **append to** `sawti/training/train_qlora.py`:
```python
class SetEpochCallback:
    """Advances the dataset's augmentation stream per epoch so
    (seed, epoch, index) determinism actually varies across epochs in
    training — not merely in isolation tests."""

    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def on_epoch_begin(self, args, state, control, **kw) -> None:
        if state is not None and getattr(state, "epoch", None) is not None:
            self.dataset.set_epoch(int(state.epoch))


class DevEvalCallback:
    """Evaluates on the 75-clip Saudi dev set at each save point. Selection
    rule (spec §2.4, Addendum 4 metric set): a checkpoint is ELIGIBLE only
    when its n-gram loop-rate <= loop_limit_pct; among eligible checkpoints
    the lowest clean macro WER wins. All four first-class metrics are logged
    (clean macro, all-valid macro, all-valid corpus, loop-rate). An
    ineligible eval counts toward the regression/stop counter."""

    def __init__(self, eval_fn, log_path: str, patience: int = 3,
                 loop_limit_pct: float = 5.0) -> None:
        self.eval_fn = eval_fn
        self.log_path = log_path
        self.patience = patience
        self.loop_limit_pct = loop_limit_pct
        self.best = float("inf")
        self.regress = 0
        self.eval_index = 0

    def _log(self, record: dict) -> None:
        from pathlib import Path

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def on_save(self, args, state, control, model=None, **kw) -> None:
        self.eval_index += 1
        result = self.eval_fn(model)
        wer = result["clean_macro_wer"]
        loop = result["loop_pct"]
        eligible = loop <= self.loop_limit_pct
        is_best = eligible and wer < self.best
        if is_best:
            self.best, self.regress = wer, 0
        else:
            self.regress += 1
        stop = self.regress >= self.patience
        if control is not None and stop:
            control.should_training_stop = True
        self._log({
            "eval_index": self.eval_index,
            "clean_macro_wer": wer,
            "all_valid_macro_wer": result["all_valid_macro_wer"],
            "all_valid_corpus_wer": result["all_valid_corpus_wer"],
            "loop_pct": loop,
            "eligible": eligible,
            "is_best": is_best,
            "best_clean_macro_wer": self.best,
            "consecutive_regressions": self.regress,
            "stop": stop,
        })
        print(f"[dev-eval {self.eval_index}] clean {wer:.1f}% loop {loop:.1f}% "
              f"{'ELIGIBLE' if eligible else 'INELIGIBLE'} best {self.best:.1f}% "
              f"regress {self.regress}/{self.patience}{' STOP' if stop else ''}")


def main() -> None:
    import argparse
    import json as _json
    from pathlib import Path

    import torch
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import (Trainer, WhisperForConditionalGeneration,
                              WhisperProcessor, pipeline)

    from sawti.env import load_env
    load_env(override=True)  # operator entry edge (see env.py policy)
    from sawti.training.dataset import SadaDataset, WhisperCollator
    from sawti.training.eval_utils import run_eval

    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--dev", default="data/sada_training/val",
                   help="checkpoint-selection dev set — MUST be the "
                        "validation split, never test-derived data")
    p.add_argument("--out", required=True)
    p.add_argument("--flavor", default="qlora", choices=["qlora", "lora"])
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--base", default="openai/whisper-large-v3")
    a = p.parse_args()

    dtype = torch.float16
    load_kw = dict(dtype=dtype)
    if a.flavor == "qlora":
        from transformers import BitsAndBytesConfig

        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    model = WhisperForConditionalGeneration.from_pretrained(a.base, **load_kw)
    if a.flavor == "qlora":
        model = prepare_model_for_kbit_training(model)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model = get_peft_model(model, build_lora_config())
    model.print_trainable_parameters()

    processor = WhisperProcessor.from_pretrained(a.base)
    train_ds = SadaDataset(a.train, augment_enabled=True, seed=42)
    targs = build_training_args(a.out, flavor=a.flavor, max_steps=a.max_steps)

    def dev_eval_fn(m) -> dict:   # dev = VALIDATION split (never test)
        from sawti.training.eval_utils import aggregate

        asr = pipeline("automatic-speech-recognition", model=m,
                       tokenizer=processor.tokenizer,
                       feature_extractor=processor.feature_extractor,
                       torch_dtype=dtype, device=0, chunk_length_s=30)
        rows = run_eval(lambda w: asr(w, generate_kwargs={
            "language": "arabic", "task": "transcribe"})["text"].strip(), a.dev)
        return aggregate(rows)

    trainer = Trainer(
        model=model, args=targs, train_dataset=train_ds,
        data_collator=WhisperCollator(processor),
        callbacks=[
            SetEpochCallback(train_ds),
            DevEvalCallback(dev_eval_fn, str(Path(a.out) / "dev_log.jsonl")),
        ],
    )
    trainer.train()
    model.save_pretrained(str(Path(a.out) / "last_adapter"))
    processor.save_pretrained(str(Path(a.out) / "last_adapter"))


if __name__ == "__main__":
    main()
```
Add `import json` to the module's top-level imports (used by `_log` and probe docs).
- [ ] **Step 3:** `uv run pytest tests/test_train_qlora.py -v` → 6 PASS (3 config + 2 callback + epoch advancement). Commit `feat(training): dev-eval callback with early stop + training entrypoint`.

Add the epoch-advancement regression (the training-facing mechanism,
not manual augment() calls):

```python
def test_set_epoch_callback_advances_dataset_stream():
    import numpy as np
    from sawti.training.dataset import SadaDataset
    from sawti.training.train_qlora import SetEpochCallback

    class _State:
        epoch = 3
    class _DS:
        def __init__(self): self.epochs = []
        def set_epoch(self, e): self.epochs.append(e)
    ds = _DS()
    SetEpochCallback(ds).on_epoch_begin(None, _State(), None)
    assert ds.epochs == [3]                     # wired to Trainer state

    # And the dataset actually varies its augmentation per epoch:
    import json as _j, soundfile as sf, tempfile, pathlib
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td)
        sf.write(p / "x.wav", np.ones(1600, np.float32) * 0.5, 16000)
        (p / "manifest.jsonl").write_text(
            _j.dumps({"clip_id": "x", "cleaned_text": "كلمة"}), encoding="utf-8")
        d0 = SadaDataset(p, augment_enabled=True, seed=42, epoch=0)
        d1 = SadaDataset(p, augment_enabled=True, seed=42, epoch=1)
        assert not np.array_equal(d0[0]["audio"], d1[0]["audio"])
        assert np.array_equal(d0[0]["audio"], d0[0]["audio"])  # stable
```

---

## Task 6: `sawti/training/export_merge.py`

**Files:** Create `sawti/training/export_merge.py`; Test `tests/test_export_merge.py`.

- [ ] **Step 1: Failing test** — `tests/test_export_merge.py`:
```python
from unittest.mock import MagicMock, patch

from sawti.training.export_merge import merge_and_save


def test_merge_and_save_loads_base_adapter_and_saves(tmp_path):
    base = MagicMock()
    peft_model = MagicMock()
    with patch("sawti.training.export_merge.WhisperForConditionalGeneration") as M, \
         patch("sawti.training.export_merge.PeftModel") as P, \
         patch("sawti.training.export_merge.WhisperProcessor") as PR:
        M.from_pretrained.return_value = base
        P.from_pretrained.return_value = peft_model
        peft_model.merge_and_unload.return_value = base
        merge_and_save("ckpt_dir", str(tmp_path))
        P.from_pretrained.assert_called_once_with(base, "ckpt_dir")
        peft_model.merge_and_unload.assert_called_once()
        base.save_pretrained.assert_called_once_with(str(tmp_path))
        PR.from_pretrained.assert_called_once_with("openai/whisper-large-v3")
```
- [ ] **Step 2:** FAIL, then **implement**:
```python
"""Merge the best LoRA adapter into the base model and save a shippable dir.

OPERATOR (after training):
  uv run python -m sawti.training.export_merge \
    --adapter checkpoints/sa_qlora/checkpoint-XXXX --out models/sa_merged
The checkpoint to use = the eval_index with is_best=true in dev_log.jsonl
(mapped to its checkpoint-<save_step> directory; save_steps=500).
"""
from __future__ import annotations

import argparse


def merge_and_save(adapter_dir: str, out_dir: str,
                   base_id: str = "openai/whisper-large-v3") -> None:
    import torch
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    base = WhisperForConditionalGeneration.from_pretrained(base_id, dtype=torch.float16)
    model = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    model.save_pretrained(out_dir)
    WhisperProcessor.from_pretrained(base_id).save_pretrained(out_dir)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--base", default="openai/whisper-large-v3")
    a = p.parse_args()
    merge_and_save(a.adapter, a.out, a.base)
    print(f"merged -> {a.out}")


if __name__ == "__main__":
    main()
```
- [ ] **Step 3:** PASS + commit `feat(training): adapter merge/export`.

---

## Task 7 — OPERATOR: the training run

Not a subagent task (hours–days of GPU). Runbook:

- [ ] Confirm `data/sada_training/train/manifest.jsonl` exists (Task 2 Step 5).
- [ ] Launch in a detached terminal:
  ```bash
  uv run python -m sawti.training.train_qlora \
    --train data/sada_training/train --dev data/sada_spike \
    --out checkpoints/sa_qlora --flavor <qlora|lora per Task 0>
  ```
- [ ] Monitor `checkpoints/sa_qlora/dev_log.jsonl` — every 500 steps: dev WER,
  best marker, regression counter. Expected shape: WER falls from ~43 toward
  20s–30s within the first few k steps; loop-rate falls with it.
- [ ] Stop conditions (automatic): 3 consecutive dev regressions, or max_steps.
- [ ] Identify best checkpoint: the `eval_index` with `is_best: true` and the
  lowest `best_wer`; map to `checkpoint-<500*eval_index>`.
- [ ] Export: `uv run python -m sawti.training.export_merge --adapter
  checkpoints/sa_qlora/checkpoint-<N> --out models/sa_merged`.
- [ ] Record the run (flavor, steps reached, best WER, VRAM) in the research
  report addendum. **If OOM:** drop per-device batch by half, double accum
  (edit Task 4 constants), restart — do not change the LoRA recipe.

---

## Task 8: `sawti/mt_m4t.py` — T2TT wrapper (RECONCILED: thin, reused)

> M1 RECONCILIATION: `sawti/providers.py` already implements the
> Whisper→M4T provider with correct ara→arb source-code mapping and
> same-language verbatim semantics. Task 8's wrapper becomes a THIN
> reusable MT callable used by Task 9's `SaudiWhisperAsr` for its
> non-Arabic targets — the same mapping table, no parallel provider.

**Files:** Create `sawti/mt_m4t.py`; Test `tests/test_mt_m4t.py`.

- [ ] **Step 1: Failing test** — `tests/test_mt_m4t.py`:
```python
from unittest.mock import MagicMock

from sawti.mt_m4t import M4TTtTranslator


def test_translate_maps_langs_and_decodes():
    processor, model = MagicMock(), MagicMock()
    model.generate.return_value = [[7, 8, 9]]
    processor.tokenizer.decode.return_value = "i need a taxi"
    mt = M4TTtTranslator(processor=processor, model=model)
    text, conf = mt.translate("أحتاج سيارة أجرة", source="ara", target="eng")
    kwargs = processor.call_args.kwargs
    assert kwargs["src_lang"] == "arb" and kwargs["tgt_lang"] == "eng"
    assert text == "i need a taxi"
    assert 0.0 <= conf <= 1.0
```
- [ ] **Step 2:** FAIL, then **implement**:
```python
"""SeamlessM4T-v2 text-to-text translation wrapper (spec §4.1).

Arabic (dialectal transcript) -> eng/fra. Injectable processor/model; the
runtime never imports this module directly (built via build_sa.py).
"""
from __future__ import annotations

from sawti.lang_codes import to_m4t_lang


class M4TTtTranslator:
    def __init__(self, processor, model, device: str = "cpu") -> None:
        self.processor = processor
        self.model = model
        self.device = device
        if not _is_mock(model):
            try:
                model = model.to(device)
            except Exception:
                pass
        self.model = model

    def translate(self, text: str, source: str, target: str) -> tuple[str, float]:
        src, tgt = to_m4t_lang(source), to_m4t_lang(target)
        inputs = self.processor(text=text, src_lang=src, tgt_lang=tgt,
                                return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)
        out = self.model.generate(**inputs, tgt_lang=tgt)
        ids = out[0]
        ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        decoded = self.processor.tokenizer.decode(ids, skip_special_tokens=True).strip()
        return decoded, 0.8  # heuristic confidence; gate uses structural checks


def _is_mock(obj) -> bool:
    try:
        from unittest.mock import Mock

        return isinstance(obj, Mock)
    except Exception:
        return False
```
- [ ] **Step 3:** PASS + commit `feat(sa): SeamlessM4T T2TT wrapper with ara->arb mapping`.

---

## Task 9: `sawti/asr_whisper_sa.py` — SaudiWhisperAsr (AsrMtProvider impl) — REVIEW CHECKPOINT

**Files:** Create `sawti/asr_whisper_sa.py`; Test `tests/test_asr_whisper_sa.py`.

- [ ] **Step 1: Failing test** — `tests/test_asr_whisper_sa.py`:
```python
from unittest.mock import MagicMock

import numpy as np

from sawti.asr_whisper_sa import SaudiWhisperAsr
from sawti.types import AudioChunk


def _chunk():
    return AudioChunk(id="c0", audio=np.zeros(16000, np.float32),
                      sample_rate=16000, start_time=0.0, end_time=1.0)


def test_same_language_skips_mt():
    asr_fn = MagicMock(return_value="أحتاج سيارة")
    mt = MagicMock()
    s = SaudiWhisperAsr(transcribe_fn=asr_fn, mt=mt)
    r = s.asr_mt(_chunk(), "ara")
    assert r.raw_text == "أحتاج سيارة"
    assert r.target_lang == "ara"
    assert r.source_lang_guess == "ara"
    mt.translate.assert_not_called()


def test_other_language_routes_through_mt():
    asr_fn = MagicMock(return_value="أحتاج سيارة")
    mt = MagicMock()
    mt.translate.return_value = ("i need a car", 0.8)
    s = SaudiWhisperAsr(transcribe_fn=asr_fn, mt=mt)
    r = s.asr_mt(_chunk(), "eng")
    assert r.raw_text == "i need a car"
    assert r.target_lang == "eng"
    mt.translate.assert_called_once()
    kwargs = mt.translate.call_args
    assert kwargs.args[1] == "ara" and kwargs.args[2] == "eng"


def test_no_mt_available_returns_asr_text():
    asr_fn = MagicMock(return_value="نص")
    s = SaudiWhisperAsr(transcribe_fn=asr_fn, mt=None)
    r = s.asr_mt(_chunk(), "eng")
    assert r.raw_text == "نص" and r.target_lang == "eng"
```
- [ ] **Step 2:** FAIL, then **implement**:
```python
"""SaudiWhisperAsr: the real AsrMtProvider (spec §4.1).

Composes the fine-tuned Saudi Whisper (dialectal Arabic text) with the
M4T T2TT wrapper into the fallback protocol FallbackHandler already
expects. Same-language (ara) skips MT — the transcribe-verbatim mode.
`transcribe_fn(wav_array, sample_rate) -> (text, confidence)` is injected;
build_sa provides the real one; tests use fakes.
"""
from __future__ import annotations

import time

import numpy as np

from sawti.types import AudioChunk, EngineResult


class SaudiWhisperAsr:
    def __init__(self, transcribe_fn, mt=None) -> None:
        self.transcribe_fn = transcribe_fn
        self.mt = mt

    def asr_mt(self, chunk: AudioChunk, target_lang: str) -> EngineResult:
        t0 = time.perf_counter()
        audio = np.ascontiguousarray(chunk.audio, dtype=np.float32)
        result = self.transcribe_fn(audio, chunk.sample_rate)
        if not (isinstance(result, tuple) and len(result) == 2
                and isinstance(result[0], str)
                and isinstance(result[1], (int, float))):
            raise TypeError(
                f"transcribe_fn must return (text: str, confidence: float); "
                f"got {type(result).__name__}")
        text, conf = result
        timing = {"asr_ms": (time.perf_counter() - t0) * 1000}
        if target_lang != "ara" and self.mt is not None:
            t1 = time.perf_counter()
            text, mt_conf = self.mt.translate(text, "ara", target_lang)
            timing["mt_ms"] = (time.perf_counter() - t1) * 1000
            conf = min(conf, mt_conf)
        return EngineResult(
            chunk_id=chunk.id, raw_text=text, confidence=conf,
            source_lang_guess="ara", timing_ms=timing, target_lang=target_lang,
        )
```
- [ ] **Step 3:** 3 PASS. **Two-stage review** (spec compliance + code quality) before commit.
- [ ] **Step 3b — pinned mock return-contract (previously identified):**
  `transcribe_fn(audio, sample_rate) -> tuple[str, float]` is the frozen
  internal seam; test doubles MUST return exactly `(text, confidence)`
  tuples (not EngineResult, not strings) — any deviation is a contract
  failure caught by the type-shape assertions below:

```python
def test_transcribe_fn_contract_pinned():
    """Injected doubles must match the frozen seam exactly."""
    calls = []
    def good_fn(audio, sr):
        calls.append((type(audio).__name__, sr))
        return ("نص", 0.9)          # tuple[str, float] — the contract
    s = SaudiWhisperAsr(transcribe_fn=good_fn, mt=None)
    r = s.asr_mt(_chunk(), "ara")
    assert isinstance(r.raw_text, str) and isinstance(r.confidence, float)
    assert calls == [("ndarray", 16000)]        # list of TUPLES


def test_transcribe_fn_malformed_return_rejected():
    """The contract is ENFORCED, not documented: a double returning a
    bare string (not a (text, confidence) tuple) fails loudly."""
    import pytest

    s = SaudiWhisperAsr(transcribe_fn=lambda a, sr: "نص فقط", mt=None)
    with pytest.raises((TypeError, ValueError)):
        s.asr_mt(_chunk(), "ara")
```

  Commit `feat(sa): SaudiWhisperAsr AsrMtProvider (ara verbatim, mt otherwise)`.

---

## Task 10: `sawti/build_sa.py` + CLI integration — REVIEW CHECKPOINT

> **RECONCILED (rewritten):** NO new orchestration. SA integration
> specializes the ASR *component* inside the ONE M1 graph:
>
>     normal M4T S2TT primary
>              ↓ rejected
>     existing FallbackHandler (retry → rechunk → ASR+MT)
>              ↓
>     Saudi Whisper ASR → existing MT semantics
>
> `build_sawti_sa_pipeline()` delegates to `sawti.build.build_real_pipeline`
> with `provider=WhisperM4TProvider(whisper_model_id=<merged SA model>)`.
> The M4T primary, fallback ordering, gate, segmenter, and postprocessor
> are untouched M1 components.
>
> **Open design question (explicitly flagged, NOT hidden in this task):**
> the SA spec §4.2 also specifies an explicit `--engine sawti-sa` mode
> making the Saudi path PRIMARY. That is a distinct runtime architecture
> extension (primary-engine swap); it requires its own review round
> before implementation and is deliberately out of scope for Task 10.
> Task 10 ships the fallback-lane integration only.

**Files:** Create `sawti/build_sa.py`; Tests `tests/test_build_sa.py`.

- [ ] **Step 1: Failing test** — `tests/test_build_sa.py`:
```python
from unittest.mock import MagicMock

import numpy as np

from sawti.build_sa import build_sawti_sa_pipeline
from sawti.engine import StubEngine
from sawti.pipeline import Pipeline
from sawti.segmenter import StubSegmenter
from sawti.sources import StubAudioSource


def test_sa_build_delegates_to_m1_graph_with_sa_provider(tmp_path):
    """The SA build IS the M1 graph: full recovery stack present, with
    the fallback provider specialized to the SA model (injectable here;
    real runs point it at the merged SA Whisper)."""
    fake_provider = MagicMock()
    fake_provider.asr_mt.return_value = None  # never invoked on happy path

    pipe = build_sawti_sa_pipeline(
        sa_model_dir=str(tmp_path),          # unused with injected provider
        provider=fake_provider,
        # HERMETIC: inject the M1 builder's engine + segmenter too —
        # default load_policy is resident, so an uninjected engine would
        # eagerly load real M4T despite the fake provider.
        m4t_engine=StubEngine("hi", 0.9),
        segmenter=StubSegmenter(chunk_frames=2, sample_rate=16000),
    )
    assert isinstance(pipe, Pipeline)
    assert pipe.fallback is not None          # M1 recovery stack
    assert pipe.fallback.asr_mt is fake_provider
    assert pipe.fallback.conservative is not None
    assert pipe.fallback.rechunker is not None
    # Happy path through the stub M1 engine never calls SA:
    src = StubAudioSource(n_frames=2, samples_per_frame=16000)
    out = list(pipe.run(src, target_lang="eng"))
    assert fake_provider.asr_mt.call_count == 0
    assert all(isinstance(o.text, str) for o in out)
```

- [ ] **Step 2: Implement** — `sawti/build_sa.py`:
```python
"""Saudi-ASR pipeline: the M1 graph with a Saudi-specialized provider.

No orchestration is created here — build_real_pipeline (M1) owns the
graph; this module only swaps the fallback lane's ASR component to the
fine-tuned Saudi Whisper model. The primary M4T S2TT path is untouched.
"""
from __future__ import annotations

from sawti.build import build_real_pipeline
from sawti.config import SawtiConfig


def build_sawti_sa_pipeline(
    config: SawtiConfig | None = None,
    on_decision=None,
    *,
    sa_model_dir: str = "models/sa_merged",
    provider=None,   # injectable for hermetic tests
    device: str | None = None,
    **kwargs,
):
    cfg = config or SawtiConfig()
    dev = device or cfg.s2tt.device
    if provider is None:
        from sawti.providers import WhisperM4TProvider

        provider = WhisperM4TProvider(device=dev)  # sa_model_dir bound in
        # Task 6's export merges the adapter; the provider gains a
        # whisper_model_id parameter defaulting to sa_model_dir here.
        provider.whisper_model_id = sa_model_dir
    return build_real_pipeline(
        cfg, on_decision=on_decision, provider=provider, device=dev, **kwargs
    )
```

- [ ] **Step 3:** Hermetic test passes; full suite green. **Two-stage
  review** before commit. Commit `feat(sa): SA provider binding on the M1 graph`.

## Task 11: Opt-in integration tests

**Files:** Create `tests/test_sa_integration.py`.
```python
import json
from pathlib import Path

import numpy as np
import pytest

# (No env loading here: tests/conftest.py owns the test-runner edge.)

SA_DIR = "models/sa_merged"


@pytest.mark.integration
def test_merged_model_on_validation_dev():
    """End-to-end: merged SA model through the eval harness on 10
    VALIDATION-split clips (checkpoint-selection data — never test)."""
    from sawti.training.eval_utils import aggregate, run_eval

    if not Path("data/sada_training/val").exists():
        pytest.skip("run Task 2 validation materialization first")
    import torch
    from transformers import (WhisperForConditionalGeneration,
                              WhisperProcessor, pipeline)

    processor = WhisperProcessor.from_pretrained(SA_DIR)
    model = WhisperForConditionalGeneration.from_pretrained(
        SA_DIR, dtype=torch.float16).to("cuda")
    asr = pipeline("automatic-speech-recognition", model=model,
                   tokenizer=processor.tokenizer,
                   feature_extractor=processor.feature_extractor,
                   torch_dtype=torch.float16, device=0, chunk_length_s=30)
    rows = run_eval(lambda w: asr(w, generate_kwargs={
        "language": "arabic", "task": "transcribe"})["text"].strip(),
        "data/sada_training/val")
    agg = aggregate(rows)
    print(f"
[integration] clean {agg['clean_macro_wer']:.1f}% | loop "
          f"{agg['loop_pct']:.0f}% | allvalid macro/corpus "
          f"{agg['all_valid_macro_wer']:.1f}/{agg['all_valid_corpus_wer']:.1f}%")
    assert agg["n"] > 0


@pytest.mark.integration
def test_sa_fallback_lane_on_real_speech():
    """The SA model as the M1 fallback provider on the licensed
    real-speech fixture (fallback integration only — primary-mode is a
    separately-reviewed extension)."""
    if not Path(SA_DIR).exists() or not Path(
            "tests/fixtures/realspeech/hello.wav").exists():
        pytest.skip("needs merged SA model + fixture")
    from sawti.build_sa import build_sawti_sa_pipeline
    from sawti.audio_io import FileSource

    pipe = build_sawti_sa_pipeline()
    out = list(pipe.run(FileSource("tests/fixtures/realspeech/hello.wav",
                                   frame_samples=16000), "eng"))
    assert out, "no segments emitted"
    assert all(isinstance(s.text, str) for s in out)
```
- [ ] Verify default-skip: `uv run pytest` → integration SKIPPED, suite green. Commit `test(sa): opt-in integration tests (validation dev + fallback lane)`.

## Task 12: Final eval — full Saudi test slice + report — OPERATOR

- [ ] **Step 1:** Materialize the test slice:
  ```bash
  uv run python -m sawti.training.data_prep --split test --out data/sada_training/test
  ```
- [ ] **Step 2:** Run the final eval (reuses harness logic; writes per-clip results + aggregates):
  ```bash
  uv run python - <<'PY'
import sawti.env, torch, json
from pathlib import Path
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
from sawti.training.eval_utils import aggregate, run_eval
OUT = Path("data/sada_training/test")
proc = WhisperProcessor.from_pretrained("models/sa_merged")
model = WhisperForConditionalGeneration.from_pretrained("models/sa_merged", dtype=torch.float16).to("cuda")
asr = pipeline("automatic-speech-recognition", model=model, tokenizer=proc.tokenizer,
               feature_extractor=proc.feature_extractor, torch_dtype=torch.float16,
               device=0, chunk_length_s=30)
rows = run_eval(lambda w: asr(w, generate_kwargs={"language":"arabic","task":"transcribe"})["text"].strip(), OUT)
agg = aggregate(rows)
Path("data/sada_training/test/final_eval.json").write_text(
    json.dumps({"aggregate": agg, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(agg, indent=1, ensure_ascii=False))
PY
  ```
- [ ] **Step 3:** Write the success verdict into the research report (Addendum 5), reading the emitted `aggregate()` fields: `clean_macro_wer` ≤ 20%? `loop_pct` < 5? every `per_dialect[d]['clean_macro_wer']` < its zero-shot (Najdi 29.4 / Hijazi 44.2 / Khaliji 53.7)? Also REPORT (not gate) `all_valid_macro_wer` AND `all_valid_corpus_wer` vs base (287.4% / 76.2%). State PASS/FAIL per criterion — an honest FAIL is a valid milestone outcome and triggers iteration (more steps/data), not reframing.
- [ ] **Step 4:** Commit report + any fixes. Merge PR.

---

## Self-review

**Spec coverage:** §2.1 base → Tasks 4/7; §2.2 recipe → Task 4 (exact constants); §2.3 data+census → Task 2; §2.4 per-dialect checkpoint selection + early stop → Task 5; §2.5 budget/OOM fallback → Tasks 0/4/7; §3 success criteria → Task 12 Step 3 (verbatim thresholds); §4.1 components → Tasks 8/9 (+mt NLLB alternative documented in spec only); §4.2 wiring → Tasks 9/10; §4.3 frozen contracts → constraint 1 (no task touches them; CLI edit is the sanctioned exception per M1 precedent); §5 layout → file structure above; §6 artifacts → Task 12 + model card is OPERATOR post-step (noted in export docstring).

**Placeholder scan:** none. `<qlora|lora>`, `<N>` are operator-supplied runtime values with defined selection procedures, not plan placeholders.

**Type consistency:** `AsrMtProvider.asr_mt(chunk, target_lang) -> EngineResult` matches M1's `fallback.py`; `M4TTtTranslator.translate(text, source, target) -> (str, float)` matches its call in Task 9; `transcribe_fn(audio, sr) -> (str, float)` consistent across Tasks 9/10; `eval_fn(model) -> aggregate()` shape consistent Tasks 5/11/12.

**Known honest risks encoded:** bnb-Windows (Task 0 decision procedure); OOM (Task 7 runbook); operator-supplied checkpoint mapping (Task 6 docstring formula).
