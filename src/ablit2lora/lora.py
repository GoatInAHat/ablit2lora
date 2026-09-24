"""The core algebra: orthogonalization deltas as exact LoRA factors.

For a linear module y = W x (+ b) and a unit refusal direction d:

    output-side orthogonalization:  W' = W - alpha * d d^T W
    input-side orthogonalization:   W' = W - alpha * W d d^T
    both-sided projection:          W' = (1-alpha) W + alpha * P_out W P_in

Every removed term is an outer product (rank <= 1 each, cross term included),
so the whole edit has rank <= 3 and ships as a PEFT LoRA adapter with
lora_alpha == r (PEFT scaling alpha/r = 1): W' == W + B @ A exactly, up to
float rounding. That identity is what makes abliteration expressible as an
adapter instead of a second copy of the weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ADAPTER_PREFIX = "base_model.model."


def _check_dim(d: torch.Tensor, n: int, name: str) -> None:
    if d.ndim != 1 or d.numel() != n:
        raise ValueError(
            f"{name} has shape {tuple(d.shape)} but the weight expects {n}; "
            "a residual-stream direction only fits modules that write into "
            "(output side) or read from (input side) the residual stream"
        )


def lora_delta(
    W: torch.Tensor,
    *,
    d_out: torch.Tensor | None = None,
    d_in: torch.Tensor | None = None,
    alpha: float = 1.0,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (A, B) with W + B @ A equal to the requested orthogonalization.

    d_out acts on the output space (len == W.shape[0]); d_in on the input
    space (len == W.shape[1]). At least one must be given. One side yields
    rank 1; both sides yield the exact rank-3 double projection.
    """
    if d_out is None and d_in is None:
        raise ValueError("provide d_out and/or d_in")
    Wf = W.float()
    do = None
    di = None
    rows: list[torch.Tensor] = []
    cols: list[torch.Tensor] = []
    if d_out is not None:
        _check_dim(d_out, Wf.shape[0], "d_out")
        do = d_out.float()
        rows.append(do @ Wf)          # A row: d^T W
        cols.append(-alpha * do)      # B col: -alpha d
    if d_in is not None:
        _check_dim(d_in, Wf.shape[1], "d_in")
        di = d_in.float()
        rows.append(di)               # A row: d^T
        cols.append(-alpha * (Wf @ di))  # B col: -alpha W d
    if do is not None and di is not None:
        c = do @ Wf @ di              # d_out^T W d_in (scalar)
        rows.append(di)               # A row: d_in^T
        cols.append(alpha * c * do)   # B col: +alpha c d_out
    A = torch.stack(rows)             # [r, in]
    B = torch.stack(cols, dim=1)      # [out, r]
    if dtype is not None:
        A = A.to(dtype)
        B = B.to(dtype)
    return A.contiguous(), B.contiguous()


def write_adapter(
    out_dir: str | Path,
    *,
    base_model: str,
    entries: dict[str, tuple[torch.Tensor, torch.Tensor]],
    alpha: float,
    side: str,
    direction_file: str,
) -> Path:
    """Write a PEFT-compatible LoRA adapter directory (r == lora_alpha).

    entries maps module path -> (A [r, in], B [out, r]) with PEFT scaling 1.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state: dict[str, torch.Tensor] = {}
    ranks: set[int] = set()
    for module, (A, B) in entries.items():
        r = A.shape[0]
        if B.shape[1] != r:
            raise ValueError(f"{module}: A rank {r} != B rank {B.shape[1]}")
        ranks.add(r)
        state[f"{ADAPTER_PREFIX}{module}.lora_A.weight"] = A
        state[f"{ADAPTER_PREFIX}{module}.lora_B.weight"] = B
    if len(ranks) != 1:
        raise ValueError(f"mixed LoRA ranks {sorted(ranks)}; emit one side at a time")
    r = ranks.pop()
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
        "lora_alpha": r,
        "lora_dropout": 0.0,
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": r,
        "rank_pattern": {},
        "revision": None,
        "target_modules": sorted(entries),
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }
    save_file(state, str(out / "adapter_model.safetensors"), metadata={"format": "pt"})
    (out / "adapter_config.json").write_text(json.dumps(config, indent=2))
    meta = {
        "alpha": alpha,
        "side": side,
        "direction_file": direction_file,
        "rank": r,
        "num_target_modules": len(entries),
        "generator": "ablit2lora",
    }
    (out / "ablit2lora.json").write_text(json.dumps(meta, indent=2))
    return out


def read_adapter(
    adapter_dir: str | Path,
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict]:
    """Read an emitted adapter back: module path -> (A, B), plus its config."""
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
