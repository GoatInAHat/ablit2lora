"""Module-name targeting: layer specs, suffix matching, MoE-aware selection."""

from __future__ import annotations

import re
from collections.abc import Iterable

DEFAULT_MODULES = ("o_proj", "down_proj")

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_EXPERT_RE = re.compile(r"(?:^|\.)experts?\.(\d+)(?:\.|$)")


def parse_layers(spec: str, num_layers: int) -> list[int]:
    """Parse a layer spec: all | even | odd | 0,3,7 | 8-24 | 8-24:2 | 5- | -5."""
    spec = (spec or "all").strip().lower()
    if spec in ("", "all"):
        return list(range(num_layers))
    if spec == "even":
        return list(range(0, num_layers, 2))
    if spec == "odd":
        return list(range(1, num_layers, 2))
    picked: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            rng, _, step_s = part.partition(":")
            step = int(step_s) if step_s else 1
            if step < 1:
                raise ValueError(f"invalid step in layer spec: {part!r}")
            lo_s, _, hi_s = rng.partition("-")
            lo = int(lo_s) if lo_s else 0
            hi = int(hi_s) if hi_s else num_layers - 1
        else:
            lo = hi = int(part)
            step = 1
        picked.extend(range(lo, hi + 1, step))
    picked = sorted(set(picked))
    bad = [i for i in picked if not 0 <= i < num_layers]
    if bad:
        raise ValueError(f"layer indices {bad} out of range 0..{num_layers - 1}")
    return picked


def layer_of(module: str) -> int | None:
    """Decoder layer index embedded in a module path, if any."""
    m = _LAYER_RE.search(module)
    return int(m.group(1)) if m else None


def is_expert_module(module: str) -> bool:
    """True for per-expert MoE modules (shared experts report False)."""
    return bool(_EXPERT_RE.search(module)) and ".shared_experts" not in module


def module_from_weight(tensor_name: str) -> str:
    """'model.layers.3.self_attn.o_proj.weight' -> strip the '.weight'."""
    suffix = ".weight"
    return tensor_name[: -len(suffix)] if tensor_name.endswith(suffix) else tensor_name


def match_suffix(module: str, suffixes: Iterable[str]) -> str | None:
    """Return the suffix (e.g. 'down_proj') that terminates the module path."""
    for s in suffixes:
        if module.endswith("." + s):
            return s
    return None


def select_targets(
    weight_names: Iterable[str],
    num_layers: int,
    layers_spec: str = "all",
    modules_spec: str = ",".join(DEFAULT_MODULES),
) -> list[str]:
    """Module paths to orthogonalize, honoring layers + module suffixes.

    MoE-safe: matching is per fully-qualified module path, so per-expert
    modules (...experts.7.down_proj) and expert-shared modules
    (...shared_experts.down_proj) each get their own exact edit.
    """
    layers = set(parse_layers(layers_spec, num_layers))
    suffixes = [s.strip() for s in modules_spec.split(",") if s.strip()]
    if not suffixes:
        raise ValueError("empty --modules spec")
    targets: list[str] = []
    for tensor_name in sorted(weight_names):
        if not tensor_name.endswith(".weight"):
            continue
        module = module_from_weight(tensor_name)
        if match_suffix(module, suffixes) is None:
            continue
        idx = layer_of(module)
        if idx is None or idx not in layers:
            continue
        targets.append(module)
    return targets
