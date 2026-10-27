"""Real PEFT mixed-rank loading: scaling, embedding layout and application."""

from __future__ import annotations

import warnings

import pytest
import torch

from ablit2lora.adapter import read_adapter, write_adapter


def test_mixed_rank_peft_load_and_merge_exact(tmp_path):
    peft = pytest.importorskip("peft")
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(4)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        tie_word_embeddings=False,
    )).eval()
    # Binary-representable weights/factors let exact equality distinguish a
    # wrong alpha/r from a genuine mixed-rank round trip, without SVD noise.
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.round(p * 16) / 16)
    modules = {
        "model.layers.0.self_attn.o_proj": 1,
        "model.layers.0.mlp.down_proj": 3,
        "model.embed_tokens": 2,
    }
    entries, expected = {}, {}
    for name, k in modules.items():
        weight = model.get_submodule(name).weight.detach()
        a = torch.zeros(k, weight.shape[1])
        b = torch.zeros(weight.shape[0], k)
        for j in range(k):
            a[j, j] = (j + 1) / 8
            b[j, j] = 1 / 4
        entries[name] = a, b
        expected[f"{name}.weight"] = weight.clone() + b @ a
    out = write_adapter(tmp_path / "adapter", base_model="local/test",
                        entries=entries, rank=3)
    back, cfg = read_adapter(out)
    assert cfg["r"] == cfg["lora_alpha"] == 3
    for name, (a, b) in back.items():
        assert a.shape[0] == b.shape[1] == modules[name]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = peft.PeftModel.from_pretrained(model, out)
    assert not [w for w in caught if "missing adapter keys" in str(w.message)]
    for name, k in modules.items():
        layer = loaded.base_model.model.get_submodule(name)
        assert layer.r["default"] == k
        assert layer.scaling["default"] == 1
    merged = loaded.merge_and_unload().state_dict()
    for key, want in expected.items():
        assert torch.equal(merged[key], want), key
        assert torch.count_nonzero(entries[key.removesuffix(".weight")][1])


@pytest.mark.parametrize("a,b", [
    (torch.zeros(0, 4), torch.zeros(3, 0)),
    (torch.zeros(1, 4), torch.zeros(3, 2)),
    (torch.zeros(1, 2, 4), torch.zeros(3, 1)),
])
def test_invalid_factors_are_rejected(tmp_path, a, b):
    with pytest.raises(ValueError):
        write_adapter(tmp_path, base_model="local/test",
                      entries={"linear": (a, b)}, rank=2)


def test_shared_projection_factors_serialize_without_storage_aliases(tmp_path):
    # One refusal basis may be reused by many modules. safetensors must
    # receive independent factor storage, not duplicate views of one tensor.
    a, b = torch.ones(2, 4), torch.ones(8, 2)
    out = write_adapter(tmp_path / "shared", base_model="local/test",
                        entries={"first": (a, b), "second": (a, b)}, rank=2)
    back, _ = read_adapter(out)
    for name in ("first", "second"):
        assert torch.equal(back[name][1] @ back[name][0], b @ a)
