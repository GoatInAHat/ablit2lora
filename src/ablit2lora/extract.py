"""Contrastive activation capture: compute refusal direction(s) per layer.

Hooks the output of each selected decoder layer, runs harmful vs benign
prompt sets, and records the (normalized) mean-difference direction of the
residual stream at each layer boundary, plus a contrast score per layer.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from .directions import save_directions
from .names import parse_layers

log = logging.getLogger(__name__)


def read_examples(path: str) -> list[dict]:
    """JSONL lines of {"text": ...} | {"messages": [...]} | plain strings."""
    examples: list[dict] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if isinstance(item, str):
            item = {"text": item}
        examples.append(item)
    if not examples:
        raise ValueError(f"no examples found in {path}")
    return examples


def _to_text(item: dict, tokenizer, chat: bool) -> str:
    if chat and "messages" in item:
        return tokenizer.apply_chat_template(
            item["messages"], tokenize=False, add_generation_prompt=True
        )
    return item.get("text") or ""


def _decoder_layers(model):
    root = model
    for attr in ("model", "transformer", "gpt_neox"):
        nxt = getattr(root, attr, None)
        if nxt is not None:
            root = nxt
    dec = getattr(root, "decoder", None) or root
    layers = getattr(dec, "layers", None) or getattr(dec, "h", None)
    if layers is None:
        raise RuntimeError("could not locate decoder layers for this architecture")
    return list(layers)


def run(
    model: str,
    harmful: str,
    benign: str,
    layers: str = "all",
    position: str = "last",
    method: str = "diff",
    batch_size: int = 8,
    max_length: int = 512,
    device: str = "auto",
    dtype: str = "auto",
    chat: bool = False,
    out: str = "directions.safetensors",
) -> None:
    """Capture contrastive activations and save per-layer directions."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    kwargs = {"dtype": dtype} if dtype != "auto" else {}
    try:
        net = AutoModelForCausalLM.from_pretrained(model, **kwargs)
    except TypeError:
        kwargs = {"torch_dtype": dtype} if dtype != "auto" else {}
        net = AutoModelForCausalLM.from_pretrained(model, **kwargs)
    net = net.to(dev).eval()

    layer_objs = _decoder_layers(net)
    sel = parse_layers(layers, len(layer_objs))
    log.info("hooking %d/%d decoder layers", len(sel), len(layer_objs))

    state = {"mask": None}
    captured: dict[int, list[torch.Tensor]] = {i: [] for i in sel}

    def make_hook(idx: int):
        def hook(_module, _inputs, outputs):
            h = outputs[0] if isinstance(outputs, tuple) else outputs
            mask = state["mask"]
            if position == "last":
                last = mask.sum(dim=1).long() - 1
                rows = h[torch.arange(h.size(0), device=h.device), last]
            else:
                m = mask.unsqueeze(-1).to(h.dtype)
                rows = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
            captured[idx].append(rows.detach().float().cpu())

        return hook

    handles = [layer_objs[i].register_forward_hook(make_hook(i)) for i in sel]

    texts_h = [_to_text(x, tokenizer, chat) for x in read_examples(harmful)]
    texts_b = [_to_text(x, tokenizer, chat) for x in read_examples(benign)]

    @torch.inference_mode()
    def _forward_all(texts: list[str]) -> None:
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(dev)
            state["mask"] = enc["attention_mask"]
            try:
                net(**enc, logits_to_keep=1)
            except TypeError:
                net(**enc)

    log.info("capturing harmful activations (%d prompts)", len(texts_h))
    _forward_all(texts_h)
    split = {i: len(captured[i]) for i in sel}
    log.info("capturing benign activations (%d prompts)", len(texts_b))
    _forward_all(texts_b)
    for h in handles:
        h.remove()

    directions: dict[int, torch.Tensor] = {}
    scores: dict[int, float] = {}
    for i in sel:
        harm = torch.cat(captured[i][: split[i]])
        ben = torch.cat(captured[i][split[i]:])
        mean_h, mean_b = harm.mean(dim=0), ben.mean(dim=0)
        diff = mean_h - mean_b
        if method == "pca":
            X = torch.cat([harm, ben])
            Xc = X - X.mean(dim=0)
            _, _, vh = torch.linalg.svd(Xc, full_matrices=False)
            d = vh[0]
            if torch.dot(d, diff) < 0:
                d = -d
        else:
            d = diff
        norm = float(d.norm())
        if norm == 0.0:
            raise ValueError(f"layer {i}: zero direction (identical means?)")
        directions[i] = d / norm
        scores[i] = norm / (float(mean_h.norm()) + float(mean_b.norm()) + 1e-8)

    save_directions(
        out,
        model_id=model,
        directions=directions,
        scores=scores,
        meta={
            "position": position,
            "method": method,
            "num_harmful": len(texts_h),
            "num_benign": len(texts_b),
            "layers": sel,
        },
    )
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    print(f"saved {len(directions)} directions -> {out}")
    print("top layers by contrast score: " + ", ".join(f"{i}={s:.3f}" for i, s in ranked[:5]))
