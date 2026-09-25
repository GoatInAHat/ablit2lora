"""Shared fixtures: synthetic safetensors checkpoint pairs + tiny Llama.

Everything is generated locally: no network, no model downloads, CPU only.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

VOCAB_WORDS = [f"tok{i}" for i in range(120)]

# small block size so synthetic tests stay tiny; the dequant/quant math is
# block-size-agnostic and the real GLM-5.3-Flash format is 128x128
FP8_BLOCK = 16


def base_tensors(seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "model.embed_tokens.weight": torch.randn(100, 32, generator=g) * 0.1,
        "model.layers.0.self_attn.o_proj.weight": torch.randn(32, 32, generator=g),
        "model.layers.1.self_attn.o_proj.weight": torch.randn(32, 24, generator=g),
        "model.layers.2.mlp.down_proj.weight": torch.randn(24, 32, generator=g),
        "model.layers.3.mlp.down_proj.weight": torch.randn(24, 32, generator=g),
        "model.layers.0.input_layernorm.weight": torch.ones(32),
    }


def write_checkpoint(d: Path, tensors: dict[str, torch.Tensor]) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: v.contiguous() for k, v in tensors.items()},
        str(d / "model.safetensors"),
        metadata={"format": "pt"},
    )
    cfg: dict = {"num_hidden_layers": 4, "hidden_size": 32}
    has_fp8 = any(
        k.endswith(".weight") and "float8" in str(v.dtype).lower()
        for k, v in tensors.items()
    )
    if has_fp8:
        # tiny synthetic blocks; the real GLM-5.3-Flash format is 128x128
        cfg["quantization_config"] = {
            "weight_block_size": [FP8_BLOCK, FP8_BLOCK]
        }
    (d / "config.json").write_text(json.dumps(cfg))
    return d


def fp8_quantize_weight(w: torch.Tensor, block: int = FP8_BLOCK):
    """Quantize one float weight to (float8_e4m3fn, bf16 block scales)."""
    from ablit2lora.quant import quantize_blockwise

    return quantize_blockwise(
        w,
        (block, block),
        out_dtype=torch.float8_e4m3fn,
        scale_dtype=torch.bfloat16,
    )


def ref_dequant(w_fp8: torch.Tensor, scale: torch.Tensor, block: int = FP8_BLOCK):
    """Independent loop-based reference dequant (cross-check for quant.py)."""
    out_f, in_f = w_fp8.shape
    nb_r = (out_f + block - 1) // block
    nb_c = (in_f + block - 1) // block
    ref = torch.zeros(out_f, in_f, dtype=torch.float64)
    for bi in range(nb_r):
        for bj in range(nb_c):
            s = float(scale[bi, bj].float())
            rows = slice(bi * block, (bi + 1) * block)
            cols = slice(bj * block, (bj + 1) * block)
            ref[rows, cols] = w_fp8[rows, cols].float().double() * s
    return ref


def make_fp8_checkpoint(
    d: Path, base_tensors: dict[str, torch.Tensor], block: int = FP8_BLOCK
) -> Path:
    """Store 2-D weights as fp8 + block scales; keep other tensors as-is."""
    stored: dict[str, torch.Tensor] = {}
    for key, t in base_tensors.items():
        if key.endswith(".weight") and t.ndim == 2 and t.is_floating_point():
            q, s = fp8_quantize_weight(t, block)
            stored[key] = q
            stored[key[: -len(".weight")] + ".weight_scale_inv"] = s
        else:
            stored[key] = t
    return write_checkpoint(d, stored)


def make_pair(base_root: Path, ablit_root: Path, edit) -> SimpleNamespace:
    base = base_tensors()
    ablit = {k: v.clone() for k, v in base.items()}
    edit(ablit, base)
    return SimpleNamespace(
        base_dir=write_checkpoint(base_root, base),
        ablit_dir=write_checkpoint(ablit_root, ablit),
        base=base,
        ablit=ablit,
    )


@pytest.fixture()
def pair_factory(tmp_path):
    """Build a synthetic (base, abliterated) checkpoint pair from an edit fn."""

    def make(edit):
        return make_pair(tmp_path / "base", tmp_path / "ablit", edit)

    return make


@pytest.fixture(scope="session")
def mixed_pair(tmp_path_factory):
    """Rank-1 edit + rank-3 edit + fully retrained tensor + unchanged rest."""

    def edit(ablit, _base):
        g1 = torch.Generator().manual_seed(101)
        ablit["model.layers.0.self_attn.o_proj.weight"] += torch.outer(
            torch.randn(32, generator=g1), torch.randn(32, generator=g1)
        )
        g2 = torch.Generator().manual_seed(102)
        ablit["model.layers.1.self_attn.o_proj.weight"] += (
            torch.randn(32, 3, generator=g2) @ torch.randn(3, 24, generator=g2)
        )
        g3 = torch.Generator().manual_seed(103)
        ablit["model.layers.2.mlp.down_proj.weight"] += 0.25 * torch.randn(
            24, 32, generator=g3
        )

    return make_pair(
        tmp_path_factory.mktemp("pair-base"),
        tmp_path_factory.mktemp("pair-ablit"),
        edit,
    )


@pytest.fixture()
def fp8_pair_factory(tmp_path):
    """(base, abliterated) pair stored as FP8 block-scale checkpoints."""

    def make(edit):
        base = base_tensors()
        ablit = {k: v.clone() for k, v in base.items()}
        edit(ablit, base)
        return SimpleNamespace(
            base_dir=make_fp8_checkpoint(tmp_path / "fp8-base", base),
            ablit_dir=make_fp8_checkpoint(tmp_path / "fp8-ablit", ablit),
            base=base,
            ablit=ablit,
        )

    return make


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizerFast

    tok = Tokenizer(
        models.WordLevel(vocab={w: i for i, w in enumerate(VOCAB_WORDS)},
                         unk_token="tok0")
    )
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = LlamaTokenizerFast(
        tokenizer_object=tok,
        bos_token="tok1",
        eos_token="tok2",
        pad_token="tok2",
        model_max_length=64,
    )
    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg)
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() > 1:
                p.normal_(0.0, 0.5)
            else:
                p.zero_()
    d = tmp_path_factory.mktemp("tiny-model")
    model.save_pretrained(str(d))
    fast.save_pretrained(str(d))
    return d
