"""Emit a PEFT LoRA adapter that applies refusal-direction orthogonalization.

For each target module W (default: every o_proj / down_proj in the selected
layers, including each MoE expert and shared-expert down_proj), emits
A = d^T W and B = -alpha*d so that W + B @ A == W - alpha * d d^T W exactly
(PEFT scaling 1 via lora_alpha == r).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from .directions import load_directions
from .lora import lora_delta, write_adapter
from .names import is_expert_module, layer_of, select_targets
from .weights import LazyWeights, model_num_layers

log = logging.getLogger(__name__)

_DTYPE_MAP = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def run(
    direction: str,
    model: str,
    layers: str = "all",
    modules: str = "o_proj,down_proj",
    side: str = "output",
    alpha: float = 1.0,
    adapter_dtype: str = "auto",
    shared_direction: int | None = None,
    out: str = "adapter",
) -> None:
    """Write the rank-1 (or rank-3 for side=both) LoRA adapter directory."""
    dirs, _meta = load_directions(direction)
    if not dirs:
        raise ValueError(f"no directions in {direction}")
    weights = LazyWeights(model)
    num_layers = model_num_layers(weights.path, weights.keys())
    targets = select_targets(weights.keys(), num_layers, layers, modules)
    if not targets:
        raise ValueError("no target modules matched; check --layers/--modules")
    log.info("%d target modules across %d layers", len(targets), num_layers)
    if shared_direction is not None and shared_direction not in dirs:
        raise ValueError(f"--shared-direction {shared_direction} not in {direction}")

    entries: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    report: list[dict] = []
    missing: set[int] = set()
    for module in targets:
        idx = layer_of(module)
        di = shared_direction if shared_direction is not None else idx
        if di not in dirs:
            missing.add(di)
            continue
        d = dirs[di]
        W = weights.get(module + ".weight")
        d_out = d if side in ("output", "both") else None
        d_in = d if side in ("input", "both") else None
        dtype = W.dtype if adapter_dtype == "auto" else _DTYPE_MAP[adapter_dtype]
        A, B = lora_delta(W, d_out=d_out, d_in=d_in, alpha=alpha, dtype=dtype)
        entries[module] = (A, B)
        bias_comp = None
        if module + ".bias" in weights:
            b = weights.get(module + ".bias")
            bias_comp = float(torch.dot(d.float(), b.float().cpu()))
        report.append(
            {
                "module": module,
                "layer": idx,
                "expert": is_expert_module(module),
                "rank": int(A.shape[0]),
                "delta_fro": float((B.float() @ A.float()).norm()),
                "w_fro": float(W.float().norm()),
                "bias_d_component": bias_comp,
            }
        )
    if missing:
        raise ValueError(
            f"no direction for layer(s) {sorted(missing)}; re-run extract for those "
            "layers or pass --shared-direction N"
        )

    out_path = write_adapter(
        out,
        base_model=model,
        entries=entries,
        alpha=alpha,
        side=side,
        direction_file=str(direction),
    )
    (Path(out_path) / "emit-report.json").write_text(json.dumps(report, indent=2))
    log.info(
        "wrote adapter for %d modules (sum delta_F %.4e) -> %s",
        len(entries),
        sum(r["delta_fro"] for r in report),
        out_path,
    )
    print(f"adapter: {out_path} ({len(entries)} modules, rank "
          f"{report[0]['rank'] if report else '?'}, side={side}, alpha={alpha})")
    for r in sorted(report, key=lambda r: -r["delta_fro"])[:5]:
        print(f"  {r['module']}: dW_F={r['delta_fro']:.4f} of W_F={r['w_fro']:.1f}")
    biased = [r for r in report if r["bias_d_component"]]
    if biased:
        worst = max(biased, key=lambda r: abs(r["bias_d_component"]))
        print(
            f"note: {len(biased)} targeted modules keep a bias component along d "
            f"(largest |d.b| = {abs(worst['bias_d_component']):.4f} at "
            f"{worst['module']}); LoRA cannot edit biases - use bake --fix-bias "
            "to remove that term in a fused copy"
        )
