"""Saudi ASR provider (SA integration, locked shape): merged Saudi
Whisper (Task 6 export, models/sa_whisper_v1) in the ASR seat -> the
existing M4T T2TT lane for translation — wired via
build_real_pipeline(provider="sa").

ASR decoding mirrors the FROZEN evaluator contract: explicit greedy
(num_beams=1, do_sample=False) with the feature extractor's
attention_mask plumbed into generate. Never the HF ASR pipeline — its
silent num_beams=5 default is exactly the regime bug Run 2 eliminated,
and this provider must produce the same transcripts the selection
evidence measured.

The merged model is Arabic-transcribe BY CONSTRUCTION (trained with
language=arabic/task=transcribe labels), so source_lang_guess is always
"ara": Arabic targets return the transcript verbatim; other targets go
through M4T arb->X. The stock provider's unmapped-language branch does
not apply.
"""
from __future__ import annotations

import time

import numpy as np

from sawti.providers import WhisperM4TProvider
from sawti.types import AudioChunk, EngineResult

# Must stay identical to the evaluator's frozen regime
# (sawti.training.eval_checkpoint.GREEDY_KWARGS) — duplicated here only so
# the runtime path does not import the training package; a regression
# pins the equality.
SA_GREEDY_KWARGS = {
    "language": "arabic",
    "task": "transcribe",
    "num_beams": 1,
    "do_sample": False,
}

DEFAULT_SA_MODEL_DIR = "models/sa_whisper_v1"


class SaudiWhisperM4TProvider(WhisperM4TProvider):
    """AsrMtProvider: Saudi Whisper ASR (merged export) + M4T T2TT."""

    def __init__(
        self,
        sa_model_dir: str = DEFAULT_SA_MODEL_DIR,
        m4t_model_id: str = "facebook/seamless-m4t-v2-large",
        device: str = "cuda",
        _sa=None,                 # injectable (processor, model) for tests
        _mt=None,                 # injectable (processor, model) tuple
    ) -> None:
        # Parent state is reused for the MT lane only; the stock-Whisper
        # ASR lane never loads (all loaders are lazy; nothing loads here).
        super().__init__(m4t_model_id=m4t_model_id, device=device, _mt=_mt)
        self.sa_model_dir = sa_model_dir
        self._sa = _sa

    def _ensure_sa(self):
        if self._sa is None:
            import torch
            from transformers import (WhisperForConditionalGeneration,
                                      WhisperProcessor)

            processor = WhisperProcessor.from_pretrained(self.sa_model_dir)
            model = (WhisperForConditionalGeneration.from_pretrained(
                self.sa_model_dir, dtype=torch.float16)
                .to(self.device).eval())
            self._sa = (processor, model)
        return self._sa

    def asr_mt(self, chunk: AudioChunk, target_lang: str) -> EngineResult:
        t0 = time.perf_counter()
        import torch

        processor, model = self._ensure_sa()
        audio = np.ascontiguousarray(chunk.audio, dtype=np.float32)
        batch = processor.feature_extractor(
            [audio], sampling_rate=16000, return_tensors="pt",
            return_attention_mask=True)
        feats = batch.input_features.to(self.device, torch.float16)
        mask = batch.attention_mask.to(self.device)
        pred = model.generate(input_features=feats, attention_mask=mask,
                              **SA_GREEDY_KWARGS)
        text = processor.tokenizer.batch_decode(
            pred, skip_special_tokens=True)[0].strip()
        if target_lang != "ara":
            text = self._translate(text, "ara", target_lang)
        return EngineResult(
            chunk_id=chunk.id,
            raw_text=text,
            confidence=0.8,
            source_lang_guess="ara",
            timing_ms={"asr_mt_ms": (time.perf_counter() - t0) * 1000.0,
                       "path": "sa_whisper+m4t",
                       "sa_model_dir": self.sa_model_dir},
            target_lang=target_lang,
        )
