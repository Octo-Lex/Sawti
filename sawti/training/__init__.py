"""Saudi ASR training package (SA milestone).

TRAIN_FLAVOR = "qlora" — decided 2026-08-18 by the Task 0 probe on this
workstation: bitsandbytes 0.50.1 on Windows/CUDA successfully constructs
Linear4bit and PagedAdamW8bit (triton absent; only flop-counting warns).
Training therefore uses NF4 4-bit + paged_adamw_8bit at effective batch
16 (8 x accum 2) per the adopted recipe. The "lora" fallback branch in
train_qlora.py remains implemented and selectable if a future
environment fails the probe.
"""
