"""Bake (fuse) an emitted adapter into full weights, shard by shard.

Optional escape hatch for pipelines that cannot carry a LoRA at serve time
(e.g. some quantized runtimes). Tensors stream one shard at a time, so RAM
stays flat; steady-state disk keeps only one serving copy (peak = base +
baked during the pass - verify, then delete the source and adapter).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .lora import read_adapter
from .weights import LazyWeights

log = logging.getLogger(__name__)


def run(adapter: str, model: str, output: str, fix_bias: bool = False) -> None:
    """Fuse W <- W + B @ A into a new model directory, preserving layout."""
    entries, _cfg = read_adapter(adapter)
    if not entries:
        raise ValueError("adapter has no LoRA entries")
    ranks = {A.shape[0] for A, _ in entries.values()}
    if fix_bias and ranks != {1}:
        raise ValueError("--fix-bias requires a rank-1 output-side adapter")

    weights = LazyWeights(model)
    src = Path(weights.path)
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"output dir {out} is not empty")
    out.mkdir(parents=True, exist_ok=True)

    for f in sorted(src.iterdir()):
        if f.is_file() and not f.name.endswith(".safetensors"):
            shutil.copy2(f, out / f.name)

    shards: list[str] = []
    for ref in weights.tensors.values():
        if ref.file not in shards:
            shards.append(ref.file)

    n_edited = 0
    n_bias = 0
    for shard in shards:
        block = {}
        with safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                module = key[: -len(".weight")] if key.endswith(".weight") else key
                if key.endswith(".weight") and module in entries:
                    A, B = entries[module]
                    t = (t.float() + B.float() @ A.float()).to(t.dtype)
                    n_edited += 1
                elif fix_bias and key.endswith(".bias") and module in entries:
                    _A, B = entries[module]
                    d_hat = B[:, 0].float()
                    d_norm = float(d_hat.norm())
                    if d_norm > 0:
                        b = t.float()
                        # B column is -alpha*d for unit d, so alpha*d(d.b) ==
                        # d_hat * (d_hat . b) / ||d_hat||
                        t = (b - d_hat * (torch.dot(d_hat, b) / d_norm)).to(t.dtype)
                        n_bias += 1
                block[key] = t
        save_file(block, str(out / Path(shard).name), metadata={"format": "pt"})
        log.info("wrote %s", out / Path(shard).name)

    idx = src / "model.safetensors.index.json"
    if idx.exists():
        shutil.copy2(idx, out / idx.name)
    print(f"baked {n_edited} module weights (+{n_bias} bias fixes) -> {out}")
    print(
        "peak disk during bake = base + baked; after verifying the output, "
        "delete the source and adapter to keep a single serving copy."
    )
