"""Quantify the T2TT stall: translation latency on a Whisper-style
hallucination-loop transcript vs its n-gram-collapsed form (the Q1 fix).

If decoding the loop takes tens of seconds, this is the live-session
65s stall mechanism, and collapsing BEFORE translation is the fix.
"""
import time

import torch
from transformers import AutoProcessor, SeamlessM4Tv2ForTextToText

phrase = "لكي نتحدث عن هذه المشكلة"
loop_text = "لذلك يجب أن نتحدث عن هذه المشكلة " + " ".join([phrase] * 40)
collapsed = "لذلك يجب أن نتحدث عن هذه المشكلة لكي نتحدث عن هذه المشكلة"

from sawti.text_normalize import collapse_repeated_loops

assert collapse_repeated_loops(loop_text) == collapsed

processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = (SeamlessM4Tv2ForTextToText.from_pretrained(
    "facebook/seamless-m4t-v2-large").to("cuda").eval())

for name, text in [("loop(40x, ~200 tok)", loop_text),
                   ("collapsed (11 tok)", collapsed)]:
    inputs = processor(text=text, src_lang="arb", tgt_lang="eng",
                       return_tensors="pt").to("cuda")
    n_tok = inputs.input_ids.shape[-1]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ids = model.generate(**inputs, tgt_lang="eng", num_beams=1)[0]
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    out = processor.tokenizer.decode(ids, skip_special_tokens=True)
    print(f"{name}: input {n_tok} tokens -> {dt:.1f}s | {out[:70]!r}",
          flush=True)
