"""ablit2lora: abliteration as a rank-1 (or rank-3) LoRA adapter.

Refusal-direction orthogonalization W' = W - d d^T W is exactly rank 1, so
the edit ships as a standard PEFT LoRA adapter: one base model copy on
disk, hot-swappable in vLLM via --enable-lora.
"""

__version__ = "0.1.0"
