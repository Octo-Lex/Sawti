"""Real ASR+MT provider for the fallback lane (spec §3.6 stage 4).

Lane: Whisper ASR (source-language transcript + detected language) ->
SeamlessM4T text-to-text translation into the target language. When the
detected source already IS the target, the transcript is returned
verbatim (the transcribe-same-language mode). Models load lazily and
stay resident; hermetic tests never instantiate the real loaders.
"""
from __future__ import annotations

import time

import numpy as np

from sawti.types import AudioChunk, EngineResult

# Whisper language names -> Sawti codes (subset we support).
_WHISPER_TO_SAWTI = {"english": "eng", "arabic": "ara", "french": "fra"}
_SAWTI_TO_M4T_SOURCE = {"eng": "eng", "ara": "arb", "fra": "fra"}


class WhisperM4TProvider:
    """AsrMtProvider: Whisper ASR + M4T T2TT, both lazy-resident."""

    def __init__(
        self,
        whisper_model_id: str = "openai/whisper-medium",
        m4t_model_id: str = "facebook/seamless-m4t-v2-large",
        device: str = "cuda",
        _asr_pipeline=None,       # injectable for tests / preloaded models
        _mt=None,                 # injectable (processor, model) tuple
    ) -> None:
        self.whisper_model_id = whisper_model_id
        self.m4t_model_id = m4t_model_id
        self.device = device
        self._asr = _asr_pipeline
        self._mt = _mt

    def _ensure_asr(self):
        if self._asr is None:
            import torch
            from transformers import pipeline as hf_pipeline

            self._asr = hf_pipeline(
                "automatic-speech-recognition",
                model=self.whisper_model_id,
                torch_dtype=torch.float16,
                device=0 if self.device == "cuda" else -1,
            )
        return self._asr

    def _ensure_mt(self):
        if self._mt is None:
            import torch
            from transformers import (AutoProcessor,
                                      SeamlessM4Tv2ForTextToText)

            processor = AutoProcessor.from_pretrained(self.m4t_model_id)
            model = SeamlessM4Tv2ForTextToText.from_pretrained(
                self.m4t_model_id).to(self.device)
            self._mt = (processor, model)
        return self._mt

    def _translate(self, text: str, src: str, tgt: str) -> str:
        """M4T T2TT lane shared with the SA provider (extracted from
        asr_mt; behavior identical)."""
        processor, model = self._ensure_mt()
        src_code = _SAWTI_TO_M4T_SOURCE[src]
        tgt_code = _SAWTI_TO_M4T_SOURCE[tgt]
        inputs = processor(text=text, src_lang=src_code, tgt_lang=tgt_code,
                           return_tensors="pt").to(self.device)
        ids = model.generate(**inputs, tgt_lang=tgt_code)[0]
        ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        return processor.tokenizer.decode(
            ids, skip_special_tokens=True).strip()

    def asr_mt(self, chunk: AudioChunk, target_lang: str) -> EngineResult:
        t0 = time.perf_counter()
        audio = np.ascontiguousarray(chunk.audio, dtype=np.float32)
        asr = self._ensure_asr()
        out = asr(audio, return_language=True)
        text = out["text"].strip()
        raw_lang = (out.get("language") or "").lower()
        detected = _WHISPER_TO_SAWTI.get(raw_lang)
        unmapped = detected is None and bool(raw_lang)
        if detected is not None and detected != target_lang:
            text = self._translate(text, detected, target_lang)
        # UNMAPPED detected language (e.g. Whisper reports a language we
        # do not support): the ASR transcript is preserved but the result
        # is explicitly UNTRUSTED — confidence 0.0 and a marker in
        # timing_ms — so the quality gate flags it rather than the text
        # silently passing as target-language output.
        return EngineResult(
            chunk_id=chunk.id,
            raw_text=text,
            confidence=0.0 if unmapped else 0.8,
            source_lang_guess=detected if detected is not None else (
                raw_lang or None),
            timing_ms={"asr_mt_ms": (time.perf_counter() - t0) * 1000.0,
                       "path": "whisper+m4t",
                       "unmapped_language": raw_lang if unmapped else None},
            target_lang=target_lang,
        )
