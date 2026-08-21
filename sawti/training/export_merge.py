"""Task 6: merge the selected QLoRA adapter into an exportable FP16 model.

Run 2 selection (reviewer-closed 2026-08-20): checkpoint-10000,
selection_score 42.26 vs stock 54.60, all guards pass.

OPERATOR:
  uv run python -m sawti.training.export_merge \\
    --checkpoint checkpoints/sa_qlora_run2/checkpoint-10000 \\
    --selection checkpoints/sa_qlora_run2/eval/checkpoint-10000.json \\
    --out models/sa_whisper_v1 --verify 16

Contract (reviewer-released 2026-08-20):
- The merged model is created from a CLEAN FP16 openai/whisper-large-v3
  base plus the EXACT selected adapter via merge_and_unload() — the
  exported weights carry no PEFT dependency at inference time.
- Processor assets (tokenizer + feature extractor) are saved alongside.
- Generation config pinned to the inference contract:
  language=arabic, task=transcribe, forced_decoder_ids=None.
- Provenance JSON in the export dir records the selected adapter's
  per-file SHA-256 (selection identity), base model ID, library
  versions, exporter commit, and the selection artifact reference.
- --verify K (default 16) reloads the EXPORTED artifact standalone —
  model AND processor (tokenizer + feature extractor) loaded from the
  export dir — and checks hypothesis parity against the
  adapter-attached model on K stride-sampled validation clips under the
  frozen evaluator regime: a genuine self-contained export smoke.
- NOT done here: pipeline integration (gated on exporter review + a
  direct ASR smoke of the merged artifact); test-split evaluation
  (LOCKED — one-shot final acceptance only).
"""
from __future__ import annotations

import json
from pathlib import Path

from sawti.training.eval_checkpoint import (
    GREEDY_KWARGS,
    SELECTION_BATCH_SIZE,
    adapter_identity,
    evaluator_commit,
    sha256_file,
    stride_sample,
)


def verify_selection_binding(checkpoint: str, selection_artifact: str) -> dict:
    """Fail-closed semantic binding (reviewer correction 1): existence +
    SHA alone would let `--checkpoint checkpoint-8000 --selection
    eval/checkpoint-10000.json` claim traceability to the wrong evidence.
    The artifact must PARSE as an eligible selection whose stored adapter
    hashes ARE the adapter being exported."""
    rec = json.loads(Path(selection_artifact).read_text(encoding="utf-8"))
    sel = rec.get("selection")
    if sel is None:
        raise ValueError(
            f"{selection_artifact}: no selection record (baseline-mode "
            f"artifact) — it cannot justify exporting any checkpoint")
    if sel.get("eligible") is not True:
        raise ValueError(
            f"{selection_artifact}: selection INELIGIBLE "
            f"(guard_fail={sel.get('guard_fail')}, loop_ok="
            f"{sel.get('loop_ok')}) — refusing to export")
    stored = (rec.get("config") or {}).get("adapter")
    if not stored:
        raise ValueError(f"{selection_artifact}: no adapter identity stored")
    actual = adapter_identity(checkpoint)
    for key in ("adapter_config.json", "adapter_model.safetensors"):
        if stored.get(key) != actual[key]:
            raise ValueError(
                f"selection/checkpoint MISMATCH on {key}: artifact has "
                f"{str(stored.get(key))[:12]}…, checkpoint being exported "
                f"is {actual[key][:12]}… — export exactly the selected "
                f"adapter, nothing else")
    return {"eligible": True, "adapter_hashes_match": True,
            "selection_score": sel.get("selection_score")}


def build_provenance(checkpoint: str, base: str, out_dir: str | Path,
                     selection_artifact: str | None) -> dict:
    """Pure provenance assembly — the export's identity card. Everything
    heavy (versions) is injected by the caller so this stays testable."""
    prov = {
        "base_model": base,
        "adapter": adapter_identity(checkpoint),
        "export_dir": str(out_dir),
        "dtype": "float16",
        "merge": "peft merge_and_unload (LoRA folded into base weights; "
                 "no PEFT dependency at inference time)",
        "generation_config": {"language": "arabic", "task": "transcribe",
                              "forced_decoder_ids": None},
        "evaluator_greedy_kwargs": dict(GREEDY_KWARGS),
        "selection_batch_size": SELECTION_BATCH_SIZE,
        "selection_artifact": selection_artifact,
        "exporter_commit": evaluator_commit(),
    }
    if selection_artifact:
        p = Path(selection_artifact)
        if not p.exists():
            raise FileNotFoundError(
                f"selection artifact {p} not found — the export must "
                f"reference the evidence that selected this adapter")
        # SEMANTIC BINDING, not just hashing: eligible record + the exact
        # adapter hashes being exported (fails closed on any mismatch).
        prov["selection_binding"] = verify_selection_binding(
            checkpoint, selection_artifact)
        prov["selection_artifact_sha256"] = sha256_file(p)
    return prov


def merge_and_export(checkpoint: str, base: str, out_dir: str | Path,
                     selection_artifact: str | None = None,
                     versions: dict | None = None) -> Path:
    """Heavy path: clean FP16 base + adapter -> merge -> save. The
    provenance is written AFTER the weights so a partial export can never
    masquerade as a complete one (same atomicity philosophy as the
    evaluator)."""
    import torch
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    out_dir = Path(out_dir)
    prov = build_provenance(checkpoint, base, out_dir, selection_artifact)
    prov["versions"] = versions or {}

    model = WhisperForConditionalGeneration.from_pretrained(
        base, dtype=torch.float16)
    model = PeftModel.from_pretrained(model, checkpoint)
    merged = model.merge_and_unload()
    # Inference contract (matches training labels + eval generate kwargs):
    merged.generation_config.language = "arabic"
    merged.generation_config.task = "transcribe"
    merged.generation_config.forced_decoder_ids = None
    merged.save_pretrained(str(out_dir))

    processor = WhisperProcessor.from_pretrained(base)
    processor.save_pretrained(str(out_dir))

    tmp = out_dir / "provenance.json.tmp"
    tmp.write_text(json.dumps(prov, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    tmp.replace(out_dir / "provenance.json")
    return out_dir


def verify_parity(checkpoint: str, base: str, export_dir: str | Path,
                  dev_dir: str, k: int = 16,
                  batch_size: int = SELECTION_BATCH_SIZE) -> dict:
    """Self-contained export smoke (reviewer correction 2): the merged
    side runs on WhisperProcessor.from_pretrained(export_dir) — tokenizer
    AND feature extractor loaded from the exported package — so the check
    proves the complete artifact, not just the weights. The reference side
    keeps the base processor with the adapter-attached model. Hypotheses
    must match on k stride-sampled validation clips under the frozen
    regime; merge is exact algebra, but fp16 rounding can flip near-ties —
    diffs are REPORTED, not hidden."""
    import torch

    from sawti.training.eval_checkpoint import transcribe_wavs
    from sawti.training.eval_utils import load_manifest

    device = "cuda"
    manifest = load_manifest(dev_dir)
    idx = stride_sample(len(manifest), k)
    paths = [str(Path(dev_dir) / f"{manifest[i]['clip_id']}.wav")
             for i in idx]

    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    base_proc = WhisperProcessor.from_pretrained(base)
    export_proc = WhisperProcessor.from_pretrained(str(export_dir))
    ref = WhisperForConditionalGeneration.from_pretrained(
        base, dtype=torch.float16).to(device)
    ref = PeftModel.from_pretrained(ref, checkpoint).eval()
    merged = WhisperForConditionalGeneration.from_pretrained(
        str(export_dir), dtype=torch.float16).to(device).eval()

    ref_hyps: list[str] = []
    merged_hyps: list[str] = []
    for lo in range(0, len(paths), batch_size):
        sl = paths[lo:lo + batch_size]
        ref_hyps += transcribe_wavs(ref, base_proc.tokenizer,
                                    base_proc.feature_extractor, sl, device)
        merged_hyps += transcribe_wavs(merged, export_proc.tokenizer,
                                       export_proc.feature_extractor, sl,
                                       device)
    diffs = [{"i": i, "adapter": a, "merged": m}
             for i, (a, m) in enumerate(zip(ref_hyps, merged_hyps))
             if a != m]
    return {"n": len(paths), "identical": len(paths) - len(diffs),
            "diffs": diffs}


def main() -> None:
    import argparse

    import peft
    import torch
    import transformers

    from sawti.env import load_env
    load_env(override=True)  # operator entry edge (see env.py policy)

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True,
                   help="the SELECTED adapter dir (checkpoint-10000)")
    p.add_argument("--out", required=True,
                   help="export dir (e.g. models/sa_whisper_v1)")
    p.add_argument("--base", default="openai/whisper-large-v3")
    p.add_argument("--selection", default=None,
                   help="selection artifact JSON — provenance records its "
                        "path + SHA-256 so the export is traceable to the "
                        "evidence that selected the adapter")
    p.add_argument("--verify", type=int, default=16, metavar="K",
                   help="parity-check K clips after export (0 disables)")
    a = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for the parity check")

    versions = {"transformers": transformers.__version__,
                "peft": peft.__version__, "torch": torch.__version__}
    out = merge_and_export(a.checkpoint, a.base, a.out, a.selection,
                           versions=versions)
    prov = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    print(f"exported {out}")
    print(f"adapter sha256: config {prov['adapter']['adapter_config.json'][:12]}… "
          f"weights {prov['adapter']['adapter_model.safetensors'][:12]}…")

    if a.verify:
        dev = "data/sada_training/val"
        res = verify_parity(a.checkpoint, a.base, out, dev, k=a.verify)
        print(f"parity vs adapter-attached on {res['n']} clips: "
              f"{res['identical']} identical, {len(res['diffs'])} diffs")
        for d in res["diffs"][:5]:
            print(f"  clip[{d['i']}] adapter={d['adapter']!r} "
                  f"merged={d['merged']!r}")


if __name__ == "__main__":
    main()
