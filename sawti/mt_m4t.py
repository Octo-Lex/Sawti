"""M4T T2TT translator (thin wrapper, SA spec Task 8's reconciled shape).

One lazy-resident text-to-text lane: Sawti codes in, translated text out.
Used by the sawti-sa primary engine (engine_sa.SaudiWhisperEngine); the
fallback providers keep their own identical lane (providers.py) so their
injection seams stay untouched — the duplication is deliberate and
documented there.
"""
from __future__ import annotations

from sawti.providers import _SAWTI_TO_M4T_SOURCE


class M4TTtTranslator:
    """SeamlessM4T text-to-text, lazy-resident, injectable for tests."""

    def __init__(self, model_id: str = "facebook/seamless-m4t-v2-large",
                 device: str = "cuda", _mt=None) -> None:
        self.model_id = model_id
        self.device = device
        self._mt = _mt  # injectable (processor, model) tuple

    def _ensure(self):
        if self._mt is None:
            from transformers import (AutoProcessor,
                                      SeamlessM4Tv2ForTextToText)

            processor = AutoProcessor.from_pretrained(self.model_id)
            model = SeamlessM4Tv2ForTextToText.from_pretrained(
                self.model_id).to(self.device)
            self._mt = (processor, model)
        return self._mt

    def translate(self, text: str, src: str, tgt: str) -> str:
        """Translate `text` between Sawti codes (eng|ara|fra)."""
        processor, model = self._ensure()
        src_code = _SAWTI_TO_M4T_SOURCE[src]
        tgt_code = _SAWTI_TO_M4T_SOURCE[tgt]
        inputs = processor(text=text, src_lang=src_code, tgt_lang=tgt_code,
                           return_tensors="pt").to(self.device)
        ids = model.generate(**inputs, tgt_lang=tgt_code)[0]
        ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        return processor.tokenizer.decode(
            ids, skip_special_tokens=True).strip()
