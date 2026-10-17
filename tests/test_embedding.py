"""Embedding LoRA layout: PEFT's nn.Embedding keys, round trips, tied warning."""

from __future__ import annotations

import json
import shutil
import warnings

import pytest
import torch
from safetensors.torch import load_file, save_file

from ablit2lora import bake, convert
from ablit2lora.adapter import (
    ADAPTER_PREFIX,
    EMBEDDING_SUFFIXES,
    is_embedding_module,
    read_adapter,
    write_adapter,
)
from conftest import base_tensors, write_checkpoint

EMBED = "model.embed_tokens"
O_PROJ = "model.layers.0.self_attn.o_proj"


def _edited_tiny_model(tiny_model_dir, tmp_path):
    """Abliterated copy of the tiny Llama: hidden-side projection of the
    embedding and of one o_proj, as in the mlabonne-style orthogonalization."""
    base_sd = load_file(str(tiny_model_dir / "model.safetensors"))
    edited = {k: v.clone() for k, v in base_sd.items()}
    g = torch.Generator().manual_seed(5)
    r = torch.randn(32, generator=g)
    r = r / r.norm()
    E = base_sd[f"{EMBED}.weight"].float()
    edited[f"{EMBED}.weight"] = (E - torch.outer(E @ r, r)).contiguous()
    W_o = base_sd[f"{O_PROJ}.weight"].float()
    edited[f"{O_PROJ}.weight"] = (W_o - torch.outer(r, r @ W_o)).contiguous()
    ablit_dir = tmp_path / "ablit"
    ablit_dir.mkdir()
    save_file(edited, str(ablit_dir / "model.safetensors"), metadata={"format": "pt"})
    for f in tiny_model_dir.iterdir():
        if f.suffix != ".safetensors":
            shutil.copy2(f, ablit_dir / f.name)
    return base_sd, edited, ablit_dir


def test_is_embedding_module_matches_documented_suffixes():
    assert EMBEDDING_SUFFIXES == (
        "embed_tokens", "wte", "word_embeddings", "embed_in", "tok_embeddings",
    )
    for s in EMBEDDING_SUFFIXES:
        assert is_embedding_module(s)
        assert is_embedding_module(f"model.language_model.{s}")
    assert not is_embedding_module("model.layers.0.self_attn.o_proj")
    assert not is_embedding_module("lm_head")
    assert not is_embedding_module("model.embed_tokens_extra")


def test_write_adapter_uses_peft_embedding_layout_and_read_adapter_inverts_it(tmp_path):
    vocab, hidden, k, rank = 40, 16, 2, 3
    g = torch.Generator().manual_seed(1)
    A_e = torch.randn(k, hidden, generator=g)
    B_e = torch.randn(vocab, k, generator=g)
    A_l = torch.randn(k, 8, generator=g)
    B_l = torch.randn(12, k, generator=g)
    out = write_adapter(
        tmp_path / "ad",
        base_model="x/y",
        entries={EMBED: (A_e, B_e), O_PROJ: (A_l, B_l)},
        rank=rank,
    )
    state = load_file(str(out / "adapter_model.safetensors"))
    assert sorted(state) == sorted([
        f"{ADAPTER_PREFIX}{EMBED}.lora_embedding_A",
        f"{ADAPTER_PREFIX}{EMBED}.lora_embedding_B",
        f"{ADAPTER_PREFIX}{O_PROJ}.lora_A.weight",
        f"{ADAPTER_PREFIX}{O_PROJ}.lora_B.weight",
    ])
    emb_a = state[f"{ADAPTER_PREFIX}{EMBED}.lora_embedding_A"]
    emb_b = state[f"{ADAPTER_PREFIX}{EMBED}.lora_embedding_B"]
    assert tuple(emb_a.shape) == (rank, vocab)
    assert tuple(emb_b.shape) == (hidden, rank)
    assert torch.equal(emb_a[:k], B_e.T) and not emb_a[k:].any()
    assert torch.equal(emb_b[:, :k], A_e.T) and not emb_b[:, k:].any()
    # PEFT applies (B @ A).T for embeddings; that must equal the delta B_e @ A_e
    assert torch.allclose((emb_b @ emb_a).T, B_e @ A_e, atol=1e-6)

    entries, _cfg = read_adapter(out)
    assert sorted(entries) == sorted([EMBED, O_PROJ])
    A_back, B_back = entries[EMBED]
    assert tuple(A_back.shape) == (rank, hidden) and tuple(B_back.shape) == (vocab, rank)
    assert torch.allclose(B_back @ A_back, B_e @ A_e, atol=1e-6)
    A_back, B_back = entries[O_PROJ]
    assert torch.allclose(B_back @ A_back, B_l @ A_l, atol=1e-6)


def test_embedding_delta_round_trips_through_peft_merge(tmp_path, tiny_model_dir):
    peft = pytest.importorskip("peft")
    from transformers import LlamaForCausalLM

    base_sd, edited, ablit_dir = _edited_tiny_model(tiny_model_dir, tmp_path)
    out = tmp_path / "adapter"
    convert.run(str(tiny_model_dir), str(ablit_dir), out=str(out))
    manifest = json.loads((out / "manifest.json").read_text())
    recs = {r["name"]: r for r in manifest["tensors"]}
    assert recs[f"{EMBED}.weight"]["status"] == "exact"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = peft.PeftModel.from_pretrained(
            LlamaForCausalLM.from_pretrained(str(tiny_model_dir)), str(out)
        )
    assert not [w for w in caught if "missing adapter keys" in str(w.message)]
    merged = model.merge_and_unload().state_dict()
    for key in (f"{EMBED}.weight", f"{O_PROJ}.weight"):
        edit_norm = (edited[key] - base_sd[key]).norm()
        assert edit_norm > 0
        assert float((merged[key] - edited[key]).norm() / edit_norm) < 1e-5, key


def test_embedding_delta_round_trips_through_bake(tmp_path, tiny_model_dir):
    base_sd, edited, ablit_dir = _edited_tiny_model(tiny_model_dir, tmp_path)
    out = tmp_path / "adapter"
    convert.run(str(tiny_model_dir), str(ablit_dir), out=str(out))
    baked_dir = tmp_path / "baked"
    bake.run(str(out), str(tiny_model_dir), str(baked_dir))
    baked = load_file(str(baked_dir / "model.safetensors"))
    for key in (f"{EMBED}.weight", f"{O_PROJ}.weight"):
        edit_norm = (edited[key] - base_sd[key]).norm()
        assert float((baked[key] - edited[key]).norm() / edit_norm) < 1e-5, key


def _embed_pair(tmp_path, *, tied: bool):
    base = base_tensors()
    ablit = {k: v.clone() for k, v in base.items()}
    g = torch.Generator().manual_seed(9)
    r = torch.randn(32, generator=g)
    r = r / r.norm()
    E = base["model.embed_tokens.weight"]
    ablit["model.embed_tokens.weight"] = E - torch.outer(E @ r, r)
    dirs = []
    for name, tensors in (("base", base), ("ablit", ablit)):
        d = write_checkpoint(tmp_path / name, tensors)
        (d / "config.json").write_text(json.dumps({"tie_word_embeddings": tied}))
        dirs.append(d)
    return dirs


def test_tied_embedding_edit_is_warned_and_no_lm_head_entry(tmp_path):
    base_dir, ablit_dir = _embed_pair(tmp_path, tied=True)
    out = tmp_path / "adapter"
    convert.run(str(base_dir), str(ablit_dir), out=str(out))
    manifest = json.loads((out / "manifest.json").read_text())
    assert len(manifest["warnings"]) == 1
    assert "model.embed_tokens" in manifest["warnings"][0]
    assert "tie_word_embeddings" in manifest["warnings"][0]
    state = load_file(str(out / "adapter_model.safetensors"))
    assert not [k for k in state if "lm_head" in k]
    assert f"{ADAPTER_PREFIX}model.embed_tokens.lora_embedding_A" in state


def test_untied_embedding_edit_has_no_warning(tmp_path):
    base_dir, ablit_dir = _embed_pair(tmp_path, tied=False)
    out = tmp_path / "adapter"
    convert.run(str(base_dir), str(ablit_dir), out=str(out))
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["warnings"] == []
