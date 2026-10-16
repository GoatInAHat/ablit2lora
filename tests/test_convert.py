"""Converter core: exact rank-1, rank escalation, retrained flagging (CPU)."""

from __future__ import annotations

import json
import math

import pytest
import torch
from safetensors.torch import load_file

from ablit2lora import convert

O1 = "model.layers.0.self_attn.o_proj.weight"
O2 = "model.layers.1.self_attn.o_proj.weight"
D2 = "model.layers.2.mlp.down_proj.weight"
LN0 = "model.layers.0.input_layernorm.weight"
MOD1 = "model.layers.0.self_attn.o_proj"
MOD2 = "model.layers.1.self_attn.o_proj"
MODD2 = "model.layers.2.mlp.down_proj"


def _rank1(seed: int, out: int = 32, inn: int = 32) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.outer(torch.randn(out, generator=g), torch.randn(inn, generator=g))


def _convert(pair, tmp_path, **kw):
    out = tmp_path / "adapter"
    convert.run(str(pair.base_dir), str(pair.ablit_dir), out=str(out), **kw)
    manifest = json.loads((out / "manifest.json").read_text())
    state = load_file(str(out / "adapter_model.safetensors"))
    cfg = json.loads((out / "adapter_config.json").read_text())
    return out, manifest, state, cfg


def _rec(manifest, name):
    return {r["name"]: r for r in manifest["tensors"]}[name]


def test_fit_rank_residual_energy_does_not_overflow_float32():
    delta = torch.diag(torch.tensor([3e20, 2e20, 1e20], dtype=torch.float32))
    rank, residual, a, b = convert.fit_rank(delta, tol=0.61, max_rank=3)

    assert rank == 1
    assert math.isfinite(residual)
    assert residual == pytest.approx(math.sqrt(5 / 14), rel=1e-6)
    assert torch.isfinite(a).all() and torch.isfinite(b).all()


def test_exact_rank1(pair_factory, tmp_path):
    def edit(ablit, _base):
        ablit[O1] += _rank1(7)

    pair = pair_factory(edit)
    out, manifest, state, cfg = _convert(pair, tmp_path)
    rec = _rec(manifest, O1)
    assert rec["status"] == "exact"
    assert rec["rank"] == 1
    assert rec["residual"] < 1e-5
    assert cfg["r"] == 1 and cfg["lora_alpha"] == 1
    assert cfg["target_modules"] == [MOD1]
    A = state[f"base_model.model.{MOD1}.lora_A.weight"]
    B = state[f"base_model.model.{MOD1}.lora_B.weight"]
    assert torch.allclose(pair.base[O1] + B @ A, pair.ablit[O1], atol=1e-4)
    assert manifest["counts"]["unchanged"] == 5
    assert manifest["counts"]["exact"] == 1


def test_rank3_escalation(pair_factory, tmp_path):
    def edit(ablit, _base):
        g = torch.Generator().manual_seed(8)
        ablit[O2] += torch.randn(32, 3, generator=g) @ torch.randn(3, 24, generator=g)

    pair = pair_factory(edit)
    out, manifest, state, cfg = _convert(pair, tmp_path)
    rec = _rec(manifest, O2)
    assert rec["status"] == "exact" and rec["rank"] == 3
    assert rec["residual"] < 1e-5
    assert cfg["r"] == 3 and cfg["lora_alpha"] == 3
    A = state[f"base_model.model.{MOD2}.lora_A.weight"]
    B = state[f"base_model.model.{MOD2}.lora_B.weight"]
    assert torch.allclose(pair.base[O2] + B @ A, pair.ablit[O2], atol=1e-4)

    # capping below the true rank flags the tensor instead of approximating
    out2 = tmp_path / "adapter2"
    with pytest.raises(ValueError, match="no tensor met tol"):
        convert.run(
            str(pair.base_dir), str(pair.ablit_dir), max_rank=2, out=str(out2)
        )
    m2 = json.loads((out2 / "manifest.json").read_text())
    rec2 = _rec(m2, O2)
    assert rec2["status"] == "flagged"
    assert rec2["residual"] >= m2["tol"]
    assert not (out2 / "adapter_model.safetensors").exists()


def test_retrained_flagged_not_approximated(mixed_pair, tmp_path):
    out, manifest, state, cfg = _convert(mixed_pair, tmp_path)
    assert _rec(manifest, O1)["status"] == "exact"
    assert _rec(manifest, O1)["rank"] == 1
    assert _rec(manifest, O2)["status"] == "exact"
    assert _rec(manifest, O2)["rank"] == 3
    retrained = _rec(manifest, D2)
    assert retrained["status"] == "flagged"
    assert retrained["residual"] >= manifest["tol"]
    assert MODD2 not in cfg["target_modules"]
    assert not any(MODD2 in k for k in state)
    # zero-padded to the max included rank; padding contributes exactly 0
    assert cfg["r"] == cfg["lora_alpha"] == 3
    A = state[f"base_model.model.{MOD1}.lora_A.weight"]
    B = state[f"base_model.model.{MOD1}.lora_B.weight"]
    assert A.shape[0] == 3
    assert torch.allclose(
        mixed_pair.base[O1] + B @ A, mixed_pair.ablit[O1], atol=1e-4
    )

    # opt-in best-effort inclusion still records the residual
    out3, manifest3, state3, cfg3 = _convert(
        mixed_pair, tmp_path / "a3", include_flagged=True
    )
    recs3 = {r["name"]: r for r in manifest3["tensors"]}
    assert recs3[D2]["status"] == "included_best_effort"
    assert recs3[D2]["residual"] >= manifest3["tol"]
    assert MODD2 in cfg3["target_modules"]
    assert manifest3["include_flagged"] is True


def test_changed_1d_unrepresentable(pair_factory, tmp_path):
    def edit(ablit, _base):
        ablit[LN0] += 0.5
        ablit[O1] += _rank1(9)

    pair = pair_factory(edit)
    out, manifest, state, cfg = _convert(pair, tmp_path)
    rec = _rec(manifest, LN0)
    assert rec["status"] == "unrepresentable"
    assert rec["delta_norm"] > 0
    assert "model.layers.0.input_layernorm" not in cfg["target_modules"]
    assert _rec(manifest, O1)["status"] == "exact"


def test_manifest_audit_fields(mixed_pair, tmp_path):
    out, manifest, state, cfg = _convert(mixed_pair, tmp_path)
    rec = _rec(manifest, O1)
    assert rec["sha_base"] == convert.tensor_sha256(mixed_pair.base[O1])
    assert rec["sha_ablit"] == convert.tensor_sha256(mixed_pair.ablit[O1])
    expected = float((mixed_pair.ablit[O1] - mixed_pair.base[O1]).norm())
    assert abs(rec["delta_norm"] - expected) < 1e-4
    assert manifest["counts"]["exact"] == 2
    assert manifest["counts"]["flagged"] == 1
    assert manifest["counts"]["unchanged"] == 3
    assert manifest["adapter"]["uniform_rank"] == 3
    assert str(mixed_pair.base_dir) in manifest["vllm_command"]
    assert "--enable-lora" in manifest["vllm_command"]


def _parse_front_matter(text: str) -> dict:
    """Parse a ``---``-delimited YAML front-matter block.

    Uses PyYAML when it is importable; otherwise falls back to a minimal
    ``key: value`` line split so this test still exercises the real
    front-matter content without hard-requiring the dependency.
    """
    front_matter = text.split("---", 2)[1]
    try:
        import yaml

        return yaml.safe_load(front_matter)
    except ImportError:
        data: dict = {}
        for line in front_matter.strip().splitlines():
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            data[key.strip()] = value.strip()
        return data


def test_write_model_card_front_matter(mixed_pair, tmp_path):
    out, _manifest, _state, _cfg = _convert(mixed_pair, tmp_path)
    readme = out / "README.md"
    assert readme.exists()

    text = readme.read_text()
    assert text.startswith("---\n")

    front_matter = _parse_front_matter(text)
    assert front_matter["library_name"] == "peft"
    assert "base_model" in front_matter
    assert front_matter["base_model_relation"] == "adapter"
    tags = front_matter["tags"]
    assert "lora" in tags  # works for a parsed list or the raw "[lora, ...]" string


def test_peft_merge_reproduces_abliterated(tmp_path, tiny_model_dir):
    peft = pytest.importorskip("peft")
    import shutil

    from safetensors.torch import load_file, save_file
    from transformers import LlamaForCausalLM

    base_sd = load_file(str(tiny_model_dir / "model.safetensors"))
    edited = {k: v.clone() for k, v in base_sd.items()}
    g = torch.Generator().manual_seed(5)
    d = torch.randn(32, generator=g)
    d = d / d.norm()
    for li in range(3):
        key = f"model.layers.{li}.self_attn.o_proj.weight"
        W = base_sd[key].float()
        edited[key] = (W - torch.outer(d, d @ W)).contiguous()
    ablit_dir = tmp_path / "ablit"
    ablit_dir.mkdir()
    save_file(edited, str(ablit_dir / "model.safetensors"),
              metadata={"format": "pt"})
    for f in tiny_model_dir.iterdir():
        if f.suffix != ".safetensors":
            shutil.copy2(f, ablit_dir / f.name)

    out = tmp_path / "adapter"
    convert.run(str(tiny_model_dir), str(ablit_dir), out=str(out))

    model = peft.PeftModel.from_pretrained(
        LlamaForCausalLM.from_pretrained(str(tiny_model_dir)), str(out)
    )
    merged = model.merge_and_unload()
    for li in range(3):
        key = f"model.layers.{li}.self_attn.o_proj.weight"
        got = merged.model.layers[li].self_attn.o_proj.weight
        assert torch.allclose(got.float(), edited[key], atol=1e-4), li


def test_convert_moe_expert_tensors(moe_pair, tmp_path):
    """Backs the architecture-agnostic/MoE-aware claim: convert must target
    only the actually-edited expert's own module path, not every expert
    sharing a down_proj suffix, and leave untouched siblings out entirely.
    """
    edited_expert = "model.layers.0.mlp.experts.1.down_proj.weight"
    shared = "model.layers.0.mlp.shared_experts.down_proj.weight"
    edited_expert_mod = edited_expert.removesuffix(".weight")
    shared_mod = shared.removesuffix(".weight")

    _out, manifest, state, cfg = _convert(moe_pair, tmp_path)
    recs = {r["name"]: r for r in manifest["tensors"]}

    assert recs[edited_expert]["status"] == "exact"
    assert recs[edited_expert]["rank"] == 1
    assert recs[shared]["status"] == "exact"
    assert recs[shared]["rank"] == 2

    # untouched sibling experts (0 and 2) are unchanged and skipped entirely --
    # not merely "not edited", but absent from the manifest's tensor list
    for expert in (0, 2):
        sibling = f"model.layers.0.mlp.experts.{expert}.down_proj.weight"
        assert sibling not in recs
        assert sibling.removesuffix(".weight") not in cfg["target_modules"]

    assert sorted(cfg["target_modules"]) == sorted([edited_expert_mod, shared_mod])
    assert cfg["r"] == cfg["lora_alpha"] == 2  # zero-padded to the larger true rank

    A1 = state[f"base_model.model.{edited_expert_mod}.lora_A.weight"]
    B1 = state[f"base_model.model.{edited_expert_mod}.lora_B.weight"]
    assert torch.allclose(
        moe_pair.base[edited_expert] + B1 @ A1, moe_pair.ablit[edited_expert], atol=1e-4
    )
    As = state[f"base_model.model.{shared_mod}.lora_A.weight"]
    Bs = state[f"base_model.model.{shared_mod}.lora_B.weight"]
    assert torch.allclose(
        moe_pair.base[shared] + Bs @ As, moe_pair.ablit[shared], atol=1e-4
    )


def test_convert_differently_named_tensors_and_generalized_tie_detection(
    differently_named_pair, tmp_path
):
    """Backs the architecture-agnostic claim for a non-default naming prefix,
    and the generalized tie-pair fix: a tied lm_head under a prefix the old
    hardcoded _TIE_PAIRS never covered (language_model.*) must resolve to
    tied_unchanged, not a spurious orphan.
    """
    edited = "language_model.layers.0.self_attn.o_proj.weight"
    edited_mod = edited.removesuffix(".weight")
    orphan_lm_head = "language_model.lm_head.weight"

    _out, manifest, state, cfg = _convert(differently_named_pair, tmp_path)
    recs = {r["name"]: r for r in manifest["tensors"]}

    assert recs[edited]["status"] == "exact"
    assert recs[edited]["rank"] == 1
    assert cfg["target_modules"] == [edited_mod]

    assert recs[orphan_lm_head]["status"] == "tied_unchanged"
    assert "sha_ablit" not in recs[orphan_lm_head]  # tied, not a real orphan

    A = state[f"base_model.model.{edited_mod}.lora_A.weight"]
    B = state[f"base_model.model.{edited_mod}.lora_B.weight"]
    assert torch.allclose(
        differently_named_pair.base[edited] + B @ A,
        differently_named_pair.ablit[edited],
        atol=1e-4,
    )


def test_convert_respects_explicit_tie_word_embeddings_false(tmp_path):
    """An explicit tie_word_embeddings=False must still block the shortcut
    even when the orphan tensor happens to be byte-identical to a same-named
    candidate partner -- the sha check alone is not what gates this; the
    config flag can override the shortcut away from "tied".
    """
    base = {
        "model.embed_tokens.weight": torch.randn(50, 16),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(16, 16),
    }
    ablit = {k: v.clone() for k, v in base.items()}
    ablit["model.layers.0.self_attn.o_proj.weight"] += _rank1(321, out=16, inn=16)
    # byte-identical to embed_tokens, but config explicitly says NOT tied
    ablit["model.lm_head.weight"] = base["model.embed_tokens.weight"].clone()

    from conftest import write_checkpoint

    base_dir = write_checkpoint(tmp_path / "base", base)
    ablit_dir = write_checkpoint(tmp_path / "ablit", ablit)
    (base_dir / "config.json").write_text(json.dumps({"tie_word_embeddings": False}))

    out = tmp_path / "adapter"
    convert.run(str(base_dir), str(ablit_dir), out=str(out))
    manifest = json.loads((out / "manifest.json").read_text())
    rec = _rec(manifest, "model.lm_head.weight")
    assert rec["status"] == "orphan"
    assert "sha_ablit" in rec
