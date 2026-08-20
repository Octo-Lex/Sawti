"""Post-hoc checkpoint evaluation for the SA QLoRA run (Run 2 contract).

Run 1 finding: live per-clip 4-bit evaluation inside on_save blocks the GPU
for hours per checkpoint (3,423 clips never finished in 14h). The approved
Run 2 architecture decouples the two: train_qlora trains continuously and
saves sparse adapter checkpoints (2k/4k/6k/8k/10k); THIS module evaluates
them afterwards.

Correctness contract (reviewer-approved 2026-08-20):
- FP16 base + attached PEFT adapter in eval() mode — the same precision
  family as the zero-shot baseline; no 4-bit, no merge (merge/export is
  Task 6, after selection).
- EXPLICIT pinned decoding (GREEDY_KWARGS), passed straight to
  model.generate. The HF ASR pipeline otherwise defaults num_beams=5
  (transformers 4.57.6, automatic_speech_recognition.py:189) — Run 1's
  callback and the spike recipe both omitted num_beams, so their "greedy"
  labels were wrong and the pinned v1 baselines' decoding regime is
  unverifiable from artifacts. Baselines are therefore RECOMPUTED with
  this module so both sides share the regime by construction (v2).
- No chunk_length_s: the materializer already rejects clips > 30s, so
  there is nothing for a chunking pipeline to do.
- Batched generation passes the feature extractor's attention_mask into
  model.generate (HF guidance for batched Whisper; mandatory reviewer
  check 2026-08-20) — padded frames must not leak into encoder attention.
- Observable + atomic: progress every clip batch; the record is written
  tmp + os.replace ONLY after every clip completed, so an interrupted
  evaluation can never masquerade as a checkpoint score. Full per-clip
  rows are stored (the v1 baseline JSON contained only aggregates).
- Regime persistence (reviewer requirement 2026-08-20): every result
  artifact records base model ID, adapter path + per-file SHA-256,
  transformers/torch versions, dtype, device, batch size, exact
  generation kwargs, attention_mask flag, validation manifest SHA-256,
  and the evaluator's own commit SHA — reproducible evidence, not
  prose-described runs.

OPERATOR (baseline recompute — no --checkpoint, stock model):
  uv run python -m sawti.training.eval_checkpoint \
    --dev data/sada_training/val \
    --out data/sada_training/val/zero_shot_baseline_v2.json --batch-size 8

OPERATOR (checkpoint evaluation):
  uv run python -m sawti.training.eval_checkpoint \
    --checkpoint checkpoints/sa_qlora_run2/checkpoint-2000 \
    --dev data/sada_training/val \
    --out checkpoints/sa_qlora_run2/eval/checkpoint-2000.json

OPERATOR (batch-size benchmark on an existing adapter; writes nothing):
  uv run python -m sawti.training.eval_checkpoint \
    --checkpoint checkpoints/sa_qlora/checkpoint-500 \
    --dev data/sada_training/val --benchmark 24
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

# The single decoding authority for selection evaluation. num_beams=1 +
# do_sample=False is greedy, by construction — never the pipeline default.
GREEDY_KWARGS = {
    "language": "arabic",
    "task": "transcribe",
    "num_beams": 1,
    "do_sample": False,
}

# The regime the HF ASR pipeline silently runs when num_beams is omitted
# (transformers 4.57.6 default). Used ONLY by --benchmark to probe how far
# regime ambiguity could have shifted v1 baseline numbers.
BEAM5_KWARGS = {
    "language": "arabic",
    "task": "transcribe",
    "num_beams": 5,
    "do_sample": False,
}

DEFAULT_BASE = "openai/whisper-large-v3"


def sha256_file(path: str | Path) -> str:
    """Streaming SHA-256 (1 MiB chunks) — manifests and adapter files are
    tens-to-hundreds of MB; never read them whole."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_identity(dev_dir: str | Path) -> dict:
    """Validation-set identity for artifacts: path + content hash, so a
    result record pins the exact manifest it was computed against."""
    m = Path(dev_dir) / "manifest.jsonl"
    if not m.exists():
        raise FileNotFoundError(f"{m} not found — no manifest, no eval")
    return {"path": str(m), "sha256": sha256_file(m)}


def adapter_identity(checkpoint: str | Path | None) -> dict | None:
    """Adapter identity for artifacts: path + per-file hashes. None for
    the stock baseline run (no adapter)."""
    if not checkpoint:
        return None
    ckpt = Path(checkpoint)
    files: dict[str, str] = {}
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        p = ckpt / name
        if p.exists():
            files[name] = sha256_file(p)
    if not files:
        raise FileNotFoundError(f"no adapter files under {ckpt}")
    return {"path": str(ckpt), **files}


def evaluator_commit() -> str:
    """Commit SHA of the evaluator code itself, recorded in every artifact
    (reviewer requirement: results must be reproducible evidence, not
    prose-described runs). 'unknown' outside a git checkout."""
    import subprocess

    repo = Path(__file__).resolve().parents[2]
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                             capture_output=True, text=True, check=True)
        sha = out.stdout.strip()
        if len(sha) == 40:
            return sha
    except Exception:
        pass
    return "unknown"


def stride_sample(n: int, k: int) -> list[int]:
    """Deterministic stride indices covering the full manifest range:
    clip i*n//k for i in range(k). No RNG, no duplicates when k <= n,
    and duration/speaker ordering spread (the manifest is source-ordered)."""
    if k >= n:
        return list(range(n))
    return [i * n // k for i in range(k)]


def load_wav_mono_16k(path: str | Path):
    """soundfile read -> (float32 mono waveform, sample_rate); fails loudly
    on non-mono-collapsible or non-16k audio (matches dataset.EXPECTED_SR)."""
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        raise ValueError(f"{path}: sample rate {sr} != 16000 — the "
                         f"materializer guarantees 16k; this is a wiring bug")
    return audio


def transcribe_wavs(model, tokenizer, feature_extractor, wav_paths,
                    device, generate_kwargs=None) -> list[str]:
    """One explicit-decoding batched generate() call per wav_paths slice.
    No pipeline in the path: every generation parameter is visible here.

    Batched Whisper inference MUST pass the feature extractor's
    attention_mask into generate (HF guidance; mandatory reviewer check
    2026-08-20) so padded frames cannot leak into encoder attention."""
    import torch

    if generate_kwargs is None:
        generate_kwargs = GREEDY_KWARGS
    audios = [load_wav_mono_16k(p) for p in wav_paths]
    batch = feature_extractor(audios, sampling_rate=16000,
                              return_tensors="pt",
                              return_attention_mask=True)
    feats = batch.input_features.to(device, torch.float16)
    mask = batch.attention_mask.to(device)
    pred = model.generate(input_features=feats, attention_mask=mask,
                          **generate_kwargs)
    return [t.strip()
            for t in tokenizer.batch_decode(pred, skip_special_tokens=True)]


def run_validation(model, tokenizer, feature_extractor, device, data_dir,
                   batch_size: int, limit: int | None = None,
                   progress_every: int = 200, echo=print) -> list[dict]:
    """Full validation pass in manifest order, batch_size clips per
    generate() call. Returns annotated rows (writes nothing — the caller
    owns atomicity)."""
    from sawti.training.eval_utils import (annotate_degenerate, load_manifest,
                                           wer_clean)

    manifest = load_manifest(data_dir)
    if limit is not None:
        manifest = manifest[:limit]
    paths = [str(Path(data_dir) / f"{m['clip_id']}.wav") for m in manifest]
    rows = []
    t0 = time.perf_counter()
    for lo in range(0, len(paths), batch_size):
        hyps = transcribe_wavs(model, tokenizer, feature_extractor,
                               paths[lo:lo + batch_size], device)
        for m, hyp in zip(manifest[lo:lo + batch_size], hyps):
            ref = (m.get("cleaned_text") or m.get("text") or "").strip()
            rows.append({**m, "hyp": hyp, "wer": wer_clean(ref, hyp)})
        done = min(lo + batch_size, len(paths))
        if done % progress_every < batch_size or done == len(paths):
            rate = done / max(1e-9, time.perf_counter() - t0)
            echo(f"[eval {done}/{len(paths)}] {rate:.1f} clips/s")
    return annotate_degenerate(rows)


def build_record(rows: list[dict], baselines: dict | None, config: dict) -> dict:
    """Assemble the authoritative per-checkpoint record. baselines=None
    (stock baseline mode) skips selection — a model cannot be screened
    against guards derived from itself."""
    from sawti.training.eval_utils import aggregate
    from sawti.training.train_qlora import compute_selection

    agg = aggregate(rows)
    selection = (compute_selection(agg, baselines)
                 if baselines is not None else None)
    return {"aggregate": agg, "selection": selection, "config": config,
            "clips": rows}


def atomic_write_json(path: str | Path, record: dict) -> Path:
    """Serialize fully, then tmp + fsync + os.replace. An interrupted or
    failing write can never leave a complete-looking authoritative file,
    and no .tmp debris survives success."""
    path = Path(path)
    payload = json.dumps(record, ensure_ascii=False)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def main() -> None:
    import argparse

    import torch

    from sawti.env import load_env
    load_env(override=True)  # operator entry edge (see env.py policy)

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dev", default="data/sada_training/val",
                   help="validation split dir (selection data; NEVER test)")
    p.add_argument("--checkpoint", default=None,
                   help="PEFT adapter dir; omit for stock zero-shot baseline")
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--out", default=None, help="output JSON (required unless "
                   "--benchmark)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=None,
                   help="evaluate only the first N clips (smoke tests)")
    p.add_argument("--benchmark", type=int, default=None, metavar="K",
                   help="time batch sizes 1/2/4/8 on K stride-sampled clips, "
                        "check hypothesis identity vs batch-1, and probe "
                        "5-beam regime drift; writes no files")
    a = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — CPU evaluation of whisper-large-v3 "
                         "is the 14-hour trap this module exists to end")
    device = "cuda"

    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    model = WhisperForConditionalGeneration.from_pretrained(
        a.base, dtype=torch.float16).to(device)
    if a.checkpoint:
        model = PeftModel.from_pretrained(model, a.checkpoint)
    model.eval()
    processor = WhisperProcessor.from_pretrained(a.base)

    if a.benchmark:
        run_benchmark(model, processor.tokenizer,
                      processor.feature_extractor, device,
                      a.dev, a.benchmark)
        return

    if not a.out:
        p.error("--out is required unless --benchmark is given")
    import transformers

    baselines = None
    if a.checkpoint:
        from sawti.training.baselines import VALIDATION_BASELINES
        baselines = VALIDATION_BASELINES
    config = {
        "model": a.base,
        "checkpoint": a.checkpoint,
        "adapter": adapter_identity(a.checkpoint),
        "batch_size": a.batch_size,
        "generate_kwargs": dict(GREEDY_KWARGS),
        "attention_mask": True,
        "dev": str(a.dev),
        "manifest": manifest_identity(a.dev),
        "limit": a.limit,
        "dtype": "float16",
        "device": torch.cuda.get_device_name(0),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "evaluator_commit": evaluator_commit(),
    }
    rows = run_validation(model, processor.tokenizer,
                          processor.feature_extractor, device,
                          a.dev, a.batch_size, limit=a.limit)
    record = build_record(rows, baselines, config)
    out = atomic_write_json(a.out, record)
    agg, sel = record["aggregate"], record["selection"]
    print(f"wrote {out}")
    print(f"clean_macro_wer {agg['clean_macro_wer']:.1f} | "
          f"all_valid_macro {agg['all_valid_macro_wer']:.1f} | "
          f"loop {agg['loop_pct']:.1f}% | degenerate "
          f"{agg['degenerate_rate']:.1f}%")
    if sel is not None:
        print(f"selection_score {sel['selection_score']:.1f} | eligible "
              f"{sel['eligible']} | guards {sel['guard_fail'] or 'ok'}")


def run_benchmark(model, tokenizer, feature_extractor, device, data_dir,
                  k: int, batch_sizes=(1, 2, 4, 8)) -> None:
    """Throughput + correctness probe. Verifies (a) every batch size yields
    IDENTICAL hypotheses to batch-1 under greedy decoding — the precondition
    for batched selection eval — and (b) how much the pipeline-default
    5-beam regime would have drifted, quantifying the v1 baseline ambiguity."""
    from sawti.training.eval_utils import load_manifest

    manifest = load_manifest(data_dir)
    idx = stride_sample(len(manifest), k)
    paths = [str(Path(data_dir) / f"{manifest[i]['clip_id']}.wav")
             for i in idx]
    print(f"benchmark: {len(paths)} stride-sampled clips from "
          f"{len(manifest)} manifest rows")

    results: dict[int, list[str]] = {}
    for bs in batch_sizes:
        t0 = time.perf_counter()
        hyps: list[str] = []
        for lo in range(0, len(paths), bs):
            hyps += transcribe_wavs(model, tokenizer, feature_extractor,
                                    paths[lo:lo + bs], device)
        wall = time.perf_counter() - t0
        results[bs] = hyps
        print(f"batch_size={bs}: {wall:.1f}s wall, "
              f"{wall / len(paths):.2f}s/clip")

    base = results[batch_sizes[0]]
    for bs in batch_sizes[1:]:
        mism = [(i, base[i], results[bs][i])
                for i in range(len(paths)) if base[i] != results[bs][i]]
        print(f"identity vs batch-1 at batch_size={bs}: "
              f"{'IDENTICAL' if not mism else f'{len(mism)} MISMATCHES'}")
        for i, b, g in mism[:3]:
            print(f"  clip[{i}] bs1={b!r} bs{bs}={g!r}")

    beam5: list[str] = []
    t0 = time.perf_counter()
    for path in paths:
        beam5 += transcribe_wavs(model, tokenizer, feature_extractor,
                                 [path], device, BEAM5_KWARGS)
    wall = time.perf_counter() - t0
    diff = sum(1 for x, y in zip(base, beam5) if x != y)
    print(f"5-beam probe (batch-1): {wall:.1f}s wall, "
          f"{diff}/{len(paths)} hypotheses differ from greedy")


if __name__ == "__main__":
    main()
