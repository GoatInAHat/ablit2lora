"""Optional fused export for pipelines that cannot carry a LoRA at serve time.

Prefer 'ablit2lora serve' (base + adapter, one copy, hot swap). bake is the
escape hatch for runtimes that cannot use adapters (some quantized/edge
stacks): it fuses W <- W + B @ A shard by shard -- one shard's tensors in
RAM at a time -- preserving the base's shard layout and configs.
Steady-state disk keeps a single serving copy (peak = base + fused during
the pass; verify, then delete the source and adapter).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

from .adapter import read_adapter
from .weights import LazyWeights

log = logging.getLogger(__name__)


def run(adapter: str, model: str, output: str) -> None:
    """Fuse W <- W + B @ A into a new model directory, preserving layout."""
    entries, _cfg = read_adapter(adapter)
    if not entries:
        raise ValueError("adapter has no LoRA entries")

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
                block[key] = t
        save_file(block, str(out / Path(shard).name), metadata={"format": "pt"})
        log.info("wrote %s", out / Path(shard).name)

    idx = src / "model.safetensors.index.json"
    if idx.exists():
        shutil.copy2(idx, out / idx.name)
    print(f"baked {n_edited} module weights -> {out}")
    print(
        "peak disk during bake = base + fused; after verifying the output, "
        "delete the source copy and adapter to keep a single serving copy."
    )
