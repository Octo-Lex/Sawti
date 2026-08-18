from unittest.mock import MagicMock

import pytest

import numpy as np

from sawti.engine_m4t import SeamlessM4TEngine
from sawti.types import AudioChunk


def _chunk(cid="c0", target="eng"):
    return AudioChunk(id=cid, audio=np.zeros(16000, np.float32),
                      sample_rate=16000, start_time=0.0, end_time=1.0)


def test_engine_maps_ara_to_arb_and_translates():
    """The wrapper maps Sawti 'ara' -> SeamlessM4T 'arb' and decodes output."""
    processor = MagicMock()
    model = MagicMock()
    # generate returns token ids; processor.decode returns text.
    model.generate.return_value = [[101, 2009, 102]]  # [BOS, 'hello', EOS]
    processor.decode.return_value = "hello"

    eng = SeamlessM4TEngine(processor=processor, model=model)
    r = eng.translate(_chunk(), target_lang="ara")
    # The tgt_lang passed to generate must be 'arb', not 'ara'.
    _kwargs = model.generate.call_args.kwargs
    assert _kwargs["tgt_lang"] == "arb"
    assert r.raw_text == "hello"
    assert r.target_lang == "ara"  # reported back in Sawti codes
    assert r.source_lang_guess is None
    assert r.chunk_id == "c0"


def test_engine_confidence_from_scores():
    import torch
    processor = MagicMock()
    model = MagicMock()
    # scores: one step, chosen token prob ~0.9
    scores = (torch.tensor([[0.05, 0.9, 0.05]]),)
    model.generate.return_value = MagicMock(sequences=[[101, 1, 102]], scores=scores)
    processor.decode.return_value = "hi"
    eng = SeamlessM4TEngine(processor=processor, model=model, return_scores=True)
    r = eng.translate(_chunk(), target_lang="eng")
    assert 0.0 <= r.confidence <= 1.0


def test_conservative_mode_passes_distinct_generation_kwargs():
    """Primary and conservative passes use the SAME model but observably
    different generation kwargs (beam search, no sampling)."""
    processor = MagicMock()
    model = MagicMock()
    model.generate.return_value = [[1, 2, 3]]
    processor.decode.return_value = "text"
    eng = SeamlessM4TEngine(processor=processor, model=model)
    eng.translate(_chunk(), target_lang="eng")                     # primary
    eng.translate(_chunk(), target_lang="eng", conservative=True)  # retry
    calls = model.generate.call_args_list
    primary_kwargs = calls[0].kwargs
    conservative_kwargs = calls[1].kwargs
    assert "num_beams" not in primary_kwargs
    assert conservative_kwargs["num_beams"] == 5
    assert conservative_kwargs["do_sample"] is False


def test_device_placement_failure_surfaces():
    """A failing .to(device) must raise, not silently continue. Uses a
    plain fake (NOT a MagicMock — the _is_mock guard intentionally skips
    device-moving test doubles)."""

    class FailingToDevice:
        def to(self, device):
            raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        SeamlessM4TEngine(processor=MagicMock(), model=FailingToDevice(),
                          device="cuda")
