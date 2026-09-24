"""Direction-file I/O: per-layer refusal directions in one small safetensors."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def save_directions(
    path: str | Path,
    *,
    model_id: str,
    directions: dict[int, torch.Tensor],
    scores: dict[int, float] | None = None,
    meta: dict | None = None,
) -> Path:
    """Write directions plus a sidecar JSON with capture metadata."""
    p = Path(path)
    tensors = {
        f"layer.{i}": d.detach().to(torch.float32).contiguous().cpu()
        for i, d in sorted(directions.items())
    }
    if not tensors:
        raise ValueError("refusing to write an empty direction file")
    save_file(tensors, str(p), metadata={"format": "pt", "ablit2lora": "1", "model": str(model_id)})
    sidecar = {
        "model": str(model_id),
        "num_directions": len(tensors),
        "layers": sorted(directions),
        "scores": {str(k): float(v) for k, v in (scores or {}).items()},
        **(meta or {}),
    }
    p.with_suffix(p.suffix + ".json").write_text(json.dumps(sidecar, indent=2))
    return p


def load_directions(path: str | Path) -> tuple[dict[int, torch.Tensor], dict]:
    """Load layer->direction map and the sidecar metadata (empty dict if absent)."""
    p = Path(path)
    state = load_file(str(p))
    dirs = {int(k.split(".")[1]): v for k, v in state.items() if k.startswith("layer.")}
    if not dirs:
        raise ValueError(f"no layer.N tensors found in {p}")
    sidecar = p.with_suffix(p.suffix + ".json")
    meta = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    return dirs, meta
