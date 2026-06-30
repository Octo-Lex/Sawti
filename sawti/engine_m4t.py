"""SeamlessM4T-v2 S2TT engine wrapper (spec §3.1–§3.3).

Takes processor/model as constructor args (dependency injection) so unit
tests use fakes and only integration tests load the real ~2.3GB model.
Maps Sawti codes (ara) -> SeamlessM4T codes (arb) internally.
"""
from __future__ import annotations

import time

import numpy as np

from sawti.lang_codes import to_m4t_lang
from sawti.types import AudioChunk, EngineResult


class SeamlessM4TEngine:
    def __init__(
        self,
        processor,          # transformers AutoProcessor (or fake)
        model,              # SeamlessM4Tv2ForS2T (or fake)
        device: str = "cpu",
        return_scores: bool = True,
    ) -> None:
        self.processor = processor
        self.model = model
        self.device = device
        self.return_scores = return_scores

    def translate(self, chunk: AudioChunk, target_lang: str) -> EngineResult:
        tgt = to_m4t_lang(target_lang)
        t0 = time.perf_counter()
        # processor expects a raw audio array (float32, 16kHz mono).
        audio = np.ascontiguousarray(chunk.audio, dtype=np.float32)
        inputs = self.processor(audios=audio, sampling_rate=chunk.sample_rate,
                                return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)
        gen_kwargs = dict(tgt_lang=tgt, generate_speech=False)
        if self.return_scores:
            gen_kwargs.update(return_dict_in_generate=True, output_scores=True)
        out = self.model.generate(**inputs, **gen_kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        confidence = self._confidence(out)
        text = self._decode(out)
        return EngineResult(
            chunk_id=chunk.id,
            raw_text=text,
            confidence=confidence,
            source_lang_guess=None,  # SeamlessM4T infers source; not exposed here
            timing_ms={"engine": elapsed_ms, "path": "seamless_m4t"},
            target_lang=target_lang,  # report Sawti code back
        )

    def _decode(self, out) -> str:
        # Handle both plain token-id output and GenerateDecoderOutput.
        seqs = getattr(out, "sequences", out)
        first = seqs[0]
        ids = first.tolist() if hasattr(first, "tolist") else list(first)
        return self.processor.decode(ids, skip_special_tokens=True).strip()

    def _confidence(self, out) -> float:
        if not self.return_scores:
            return 0.8  # default heuristic when scores unavailable
        scores = getattr(out, "scores", None)
        if not scores:
            return 0.8
        try:
            import torch
            probs = []
            for step_scores in scores:
                # step_scores: (batch, vocab) logits; take max softmax prob.
                p = torch.softmax(step_scores, dim=-1).max(dim=-1).values
                probs.append(float(p.mean().item()))
            return sum(probs) / len(probs) if probs else 0.8
        except Exception:
            return 0.8
