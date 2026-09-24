"""PEFT LoRA adapter I/O with exact scaling: lora_alpha == r, scale 1.0.

Converted adapters carry factors with B @ A == delta per module (from the
SVD in convert.py). Modules may have different true ranks; factors are
zero-padded to one uniform rank r so a single PEFT config (and vLLM) can
serve them all, with lora_alpha == r so the applied scale is exactly 1.0.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ADAPTER_PREFIX = "base_model.model."


def write_adapter(
    out_dir: str | Path,
    *,
    base_model: str,
    entries: dict[str, tuple[torch.Tensor, torch.Tensor]],
    rank: int,
    dtype: torch.dtype = torch.float32,
) -> Path:
    """Write a PEFT-compatible LoRA adapter directory.

    entries maps module path -> (A [k, in], B [out, k]) with k <= rank.
    Factors are zero-padded to "rank" (padding contributes exactly 0 to
    B @ A) and stored as "dtype".
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if rank < 1:
        raise ValueError("adapter rank must be >= 1")
    state: dict[str, torch.Tensor] = {}
    for module, (A, B) in sorted(entries.items()):
        k = A.shape[0]
        if B.shape[1] != k:
            raise ValueError(f"{module}: A rank {k} != B rank {B.shape[1]}")
        if k > rank:
            raise ValueError(f"{module}: rank {k} exceeds adapter rank {rank}")
        Ap = torch.zeros(rank, A.shape[1], dtype=dtype)
        Bp = torch.zeros(B.shape[0], rank, dtype=dtype)
        Ap[:k] = A.to(dtype)
        Bp[:, :k] = B.to(dtype)
        state[f"{ADAPTER_PREFIX}{module}.lora_A.weight"] = Ap.contiguous()
        state[f"{ADAPTER_PREFIX}{module}.lora_B.weight"] = Bp.contiguous()
    config = {
        "auto_mapping": None,
        "base_model_name_or_path": base_model,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layer_replication": None,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "lora_alpha": rank,
        "lora_dropout": 0.0,
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": rank,
        "rank_pattern": {},
        "revision": None,
        "target_modules": sorted(entries),
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }
    save_file(state, str(out / "adapter_model.safetensors"), metadata={"format": "pt"})
    (out / "adapter_config.json").write_text(json.dumps(config, indent=2))
    return out


def read_adapter(
    adapter_dir: str | Path,
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict]:
    """Read an adapter back: module path -> (A, B), plus its config."""
    d = Path(adapter_dir)
    cfg = json.loads((d / "adapter_config.json").read_text())
    state = load_file(str(d / "adapter_model.safetensors"))
    grouped: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in state.items():
        if not key.startswith(ADAPTER_PREFIX):
            raise ValueError(f"unexpected adapter key {key!r}")
        rest = key[len(ADAPTER_PREFIX):]
        for which in ("lora_A", "lora_B"):
            suffix = f".{which}.weight"
            if rest.endswith(suffix):
                grouped.setdefault(rest[: -len(suffix)], {})[which] = tensor
                break
    entries = {m: (p["lora_A"], p["lora_B"]) for m, p in grouped.items()}
    return entries, cfg
