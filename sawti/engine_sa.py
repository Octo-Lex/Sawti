"""Saudi Whisper as a PRIMARY S2TT engine (spec SA §4.2: --engine sawti-sa).

Pipeline position: Segmenter -> THIS ENGINE -> BalancedQualityGate ->
FallbackHandler (restart / beam retry / rechunk / whisper-medium ASR+MT)
-> RealPostProcessor. The merged Saudi Whisper export
(models/sa_whisper_v1) transcribes dialectal Arabic; non-Arabic targets
go through the M4T T2TT lane (mt_m4t.M4TTtTranslator).

Decode modes mirror engine_m4t so the whole fallback machinery binds
unchanged:
- primary: the FROZEN SA regime (greedy, language=arabic, attention_mask)
- conservative: beam search, no sampling
- restart: greedy with n-gram repetition blocked (degenerate recovery)

Scope (honest limitation, spec-consistent): the model is
Arabic-transcribe by construction — this engine mode assumes
Arabic-dominant input, exactly as the SA milestone defined it. No
language-ID routing is added; non-Arabic speech is the stock graph's
job (`--engine m4t`), and catastrophic output here is caught by the
gate's hard checks plus the whisper-medium fallback lane (a DIFFERENT
model, so escalation is a genuine second opinion).

Evidence (2026-10-12 holdout, 81 real dialectal clips): SA model direct
24.3% clean WER vs the m4t-primary graph's 37.2% — this mode exists to
close that gap.
"""
from __future__ import annotations

import time

import numpy as np

from sawti.engine_m4t import _is_mock
from sawti.mt_m4t import M4TTtTranslator
from sawti.providers_sa import DEFAULT_SA_MODEL_DIR, SA_GREEDY_KWARGS
from sawti.text_normalize import collapse_repeated_loops
from sawti.types import AudioChunk, EngineResult


class SaudiWhisperEngine:
    """S2TTEngine over the merged Saudi Whisper export."""

    def __init__(self, processor, model, device: str = "cuda",
                 mt: M4TTtTranslator | None = None,
                 return_scores: bool = True) -> None:
        self.processor = processor
        self.device = device
        self.mt = mt or M4TTtTranslator(device=device)
        self.return_scores = return_scores
        if not _is_mock(model):
            model = model.to(device)  # placement failures must surface
        self.model = model

    def translate(self, chunk: AudioChunk, target_lang: str,
                  conservative: bool = False,
                  restart: bool = False) -> EngineResult:
        t0 = time.perf_counter()
        import torch

        audio = np.ascontiguousarray(chunk.audio, dtype=np.float32)
        batch = self.processor.feature_extractor(
            [audio], sampling_rate=chunk.sample_rate, return_tensors="pt",
            return_attention_mask=True)
        feats = batch.input_features.to(self.device, torch.float16)
        mask = batch.attention_mask.to(self.device)
        gen_kwargs = dict(SA_GREEDY_KWARGS)  # the frozen regime, copied
        if conservative:
            gen_kwargs.update(num_beams=5, do_sample=False)
        if restart:
            # Degenerate recovery: block the loop detector's span class.
            gen_kwargs.update(no_repeat_ngram_size=8)
        if self.return_scores:
            gen_kwargs.update(return_dict_in_generate=True,
                              output_scores=True)
        out = self.model.generate(input_features=feats,
                                  attention_mask=mask, **gen_kwargs)
        asr_ms = (time.perf_counter() - t0) * 1000.0
        text = self.processor.tokenizer.batch_decode(
            getattr(out, "sequences", out), skip_special_tokens=True
        )[0].strip()
        # Degenerate transcripts collapse before translation and the gate
        # (the same guard the fallback lane applies at its seam).
        text = collapse_repeated_loops(text)
        confidence = self._confidence(out)
        mode = ("restart" if restart
                else "conservative" if conservative else "primary")
        if target_lang != "ara":
            mt_t0 = time.perf_counter()
            text = self.mt.translate(text, "ara", target_lang)
            timing = {"engine": (time.perf_counter() - t0) * 1000.0,
                      "asr_ms": asr_ms,
                      "mt_ms": (time.perf_counter() - mt_t0) * 1000.0,
                      "path": "sa_whisper+m4t", "mode": mode}
        else:
            timing = {"engine": asr_ms, "path": "sa_whisper",
                      "mode": mode}
        return EngineResult(
            chunk_id=chunk.id,
            raw_text=text,
            confidence=confidence,
            source_lang_guess="ara",  # Arabic-transcribe by construction
            timing_ms=timing,
            target_lang=target_lang,
        )

    def _confidence(self, out) -> float:
        """Mean max-softmax over decode steps (the engine_m4t convention);
        0.8 heuristic when scores are unavailable."""
        scores = getattr(out, "scores", None)
        if not self.return_scores or not scores:
            return 0.8
        try:
            import torch

            probs = []
            for step in scores:
                p = torch.softmax(step, dim=-1).max(dim=-1).values
                probs.append(float(p.mean().item()))
            return sum(probs) / len(probs) if probs else 0.8
        except Exception:
            return 0.8


def load_sa_engine(device: str = "cuda",
                   sa_model_dir: str = DEFAULT_SA_MODEL_DIR) -> SaudiWhisperEngine:
    """Real loader: merged export (fp16) + the shared T2TT lane."""
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(sa_model_dir)
    model = (WhisperForConditionalGeneration.from_pretrained(
        sa_model_dir, dtype=torch.float16).to(device).eval())
    return SaudiWhisperEngine(processor=processor, model=model, device=device)
