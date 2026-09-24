"""Lazy, low-memory access to a model's safetensors shards.

emit/bake never need the whole model in RAM: they map tensor names to shards
up front and read one tensor at a time via safetensors memory-mapping.
"""

from __future__ import annotations

import glob
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass

from safetensors import safe_open


def resolve_model_path(model: str) -> str:
    """Local dir passes through; an HF id is snapshotted (safetensors+json only)."""
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download

    return snapshot_download(model, allow_patterns=["*.safetensors", "*.json"])


@dataclass(frozen=True)
class TensorRef:
    file: str
    shape: tuple[int, ...]
    dtype: str


class LazyWeights:
    """Tensor-name -> shard mapping with per-tensor lazy reads."""

    def __init__(self, model: str):
        self.path = resolve_model_path(model)
        shards = sorted(glob.glob(os.path.join(self.path, "*.safetensors")))
        if not shards:
            raise FileNotFoundError(f"no .safetensors shards found in {self.path}")
        self.tensors: dict[str, TensorRef] = {}
        for shard in shards:
            with safe_open(shard, framework="pt", device="cpu") as f:
                for key in f.keys():
                    sl = f.get_slice(key)
                    self.tensors[key] = TensorRef(
                        shard, tuple(sl.get_shape()), str(sl.get_dtype())
                    )

    def get(self, name: str):
        ref = self.tensors[name]
        with safe_open(ref.file, framework="pt", device="cpu") as f:
            return f.get_tensor(name)

    def __contains__(self, name: object) -> bool:
        return name in self.tensors

    def __iter__(self):
        return iter(self.tensors)

    def __len__(self) -> int:
        return len(self.tensors)

    def keys(self):
        return self.tensors.keys()


def model_num_layers(model_path: str, weight_names: Iterable[str]) -> int:
    """Decoder layer count from config.json if possible, else inferred."""
    cfg_path = os.path.join(model_path, "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as fh:
            cfg = json.load(fh)
        tc = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
        for key in ("num_hidden_layers", "num_layers", "n_layer"):
            if isinstance(tc, dict) and tc.get(key) is not None:
                return int(tc[key])
    from .names import layer_of

    best = -1
    for name in weight_names:
        idx = layer_of(name)
        if idx is not None:
            best = max(best, idx)
    if best < 0:
        raise ValueError("cannot determine decoder layer count for model")
    return best + 1
