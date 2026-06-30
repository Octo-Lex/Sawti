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


def _is_mock(obj) -> bool:
    """Detect unittest.mock.Mock/MagicMock so we skip device-moving them.

    MagicMock responds to `.to()` but returns a *different* mock, which would
    break unit tests that assert on the original model's `.generate` calls.
    """
    try:
        from unittest.mock import Mock

        return isinstance(obj, Mock)
    except Exception:
        return False


class SeamlessM4TEngine:
    def __init__(
        self,
        processor,          # transformers AutoProcessor (or fake)
        model,              # SeamlessM4Tv2ForSpeechToText (or fake)
        device: str = "cpu",
        return_scores: bool = True,
    ) -> None:
        self.processor = processor
        self.device = device
        self.return_scores = return_scores
        # Own device placement so callers don't have to remember `.to(device)`.
        # For real torch models, move them to the target device. Fakes used in
        # unit tests (MagicMock) respond to `.to` but return a different mock,
        # which would break later assertions — so we skip moving fakes.
        if not _is_mock(model):
            try:
                model = model.to(device)
            except Exception:
                pass  # fakes without a working .to() are left as-is
        self.model = model

    def translate(self, chunk: AudioChunk, target_lang: str) -> EngineResult:
        tgt = to_m4t_lang(target_lang)
        t0 = time.perf_counter()
        # processor expects a raw audio array (float32, 16kHz mono).
        audio = np.ascontiguousarray(chunk.audio, dtype=np.float32)
        # `audio=` (singular) is the current param name; `audios=` is deprecated
        # and removed in transformers v4.59.
        inputs = self.processor(audio=audio, sampling_rate=chunk.sample_rate,
                                return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)
        # SeamlessM4Tv2ForSpeechToText only does S2T, so there is no
        # `generate_speech` flag (that exists only on the speech-to-speech
        # class). Pass just the target lang (+ optional score return).
        gen_kwargs = dict(tgt_lang=tgt)
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
