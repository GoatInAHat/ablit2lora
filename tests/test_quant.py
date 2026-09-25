"""FP8 block-scale dequant + quantized-pair convert/bake coverage (CPU)."""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import load_file

from ablit2lora import bake, convert
from ablit2lora.quant import (
    dequant_blockwise,
    dtype_family,
    find_scale_key,
    requant_fused,
)
from conftest import FP8_BLOCK, ref_dequant

O1 = "model.layers.0.self_attn.o_proj.weight"
O1_SCALE = "model.layers.0.self_attn.o_proj.weight_scale_inv"
O2 = "model.layers.1.self_attn.o_proj.weight"
MOD1 = "model.layers.0.self_attn.o_proj"
MOD2 = "model.layers.1.self_attn.o_proj"


def test_find_scale_key_and_family():
    names = {"m.weight", "m.weight_scale_inv", "m.bias"}
    assert find_scale_key("m.weight", names) == "m.weight_scale_inv"
    assert find_scale_key("m.bias", names) is None
    names2 = {"m.weight", "m.weight_scale"}
    assert find_scale_key("m.weight", names2) == "m.weight_scale"
    assert dtype_family("float8_e4m3fn") == "fp8"
    assert dtype_family("bfloat16") == "plain"


def test_dequant_matches_independent_reference(fp8_pair_factory):
    pair = fp8_pair_factory(lambda ablit, base: None)
    state_b = load_file(str(pair.base_dir / "model.safetensors"))
    ref = ref_dequant(state_b[O1], state_b[O1_SCALE], FP8_BLOCK)
    got = dequant_blockwise(state_b[O1], state_b[O1_SCALE], (FP8_BLOCK, FP8_BLOCK))
    assert torch.allclose(got.double(), ref, atol=1e-6)


def test_quant_roundtrip_error_is_tiny(fp8_pair_factory):
    pair = fp8_pair_factory(lambda ablit, base: None)
    state = load_file(str(pair.base_dir / "model.safetensors"))
    deq = dequant_blockwise(state[O1], state[O1_SCALE], (FP8_BLOCK, FP8_BLOCK))
    orig = pair.base[O1].double()
    rel = (deq.double() - orig).norm() / orig.norm()
    # e4m3 + bf16 scales: RMS error ~2.7% of the weight norm, well under 5%
    assert rel < 0.05, rel


def test_requant_fused_fresh_scales(fp8_pair_factory):
    pair = fp8_pair_factory(lambda ablit, base: None)
    state = load_file(str(pair.base_dir / "model.safetensors"))
    g = torch.Generator().manual_seed(3)
    delta = torch.outer(torch.randn(32, generator=g), torch.randn(32, generator=g))
    q, s = requant_fused(
        state[O1], state[O1_SCALE], delta, (FP8_BLOCK, FP8_BLOCK)
    )
    assert q.dtype == state[O1].dtype
    assert s.dtype == state[O1_SCALE].dtype
    fused_deq = dequant_blockwise(q, s, (FP8_BLOCK, FP8_BLOCK)).double()
    want = dequant_blockwise(
        state[O1], state[O1_SCALE], (FP8_BLOCK, FP8_BLOCK)
    ).double() + delta.double()
    rel = (fused_deq - want).norm() / want.norm()
    assert rel < 0.05, rel


def test_convert_fp8_pair_dequant_diff(fp8_pair_factory, tmp_path):
    # large rank-1 edit (rel delta ~1.0) so the FP8 grid noise of the
    # requantized abliterated side stays under the tolerance
    def edit(ablit, base):
        g = torch.Generator().manual_seed(11)
        d = torch.outer(torch.randn(32, generator=g), torch.randn(32, generator=g))
        d = d / d.norm() * base[O1].norm()
        ablit[O1] += d

    pair = fp8_pair_factory(edit)
    out = tmp_path / "adapter"
    # FP8-vs-FP8 diffs carry quant-grid noise (~3% RMS): the tolerance must
    # be looser than a BF16 pair's -- that is the honest measurement
    convert.run(str(pair.base_dir), str(pair.ablit_dir), out=str(out), tol=0.05)
    manifest = json.loads((out / "manifest.json").read_text())
    recs = {r["name"]: r for r in manifest["tensors"]}
    assert recs[O1]["status"] == "exact"
    assert recs[O1]["rank"] == 1
    assert recs[O1]["quant"].startswith("fp8_dequant_block")
    assert manifest["quantization"]["dequant"].startswith("fp8_dequant_block")
    assert manifest["quantization"]["block_size"] == [FP8_BLOCK, FP8_BLOCK]
    # scale tensors are consumed by the dequant, never converted
    assert recs[O1_SCALE]["status"] == "scale_tensor"
    assert manifest["counts"]["scale_tensor"] >= 1
    # the adapter reproduces the dequantized abliterated weights
    state_a = load_file(str(pair.ablit_dir / "model.safetensors"))
    adapter = load_file(str(out / "adapter_model.safetensors"))
    A = adapter[f"base_model.model.{MOD1}.lora_A.weight"]
    B = adapter[f"base_model.model.{MOD1}.lora_B.weight"]
    deq_b = dequant_blockwise(
        load_file(str(pair.base_dir / "model.safetensors"))[O1],
        load_file(str(pair.base_dir / "model.safetensors"))[O1_SCALE],
        (FP8_BLOCK, FP8_BLOCK),
    )
    got = deq_b + (B @ A).float()
    want = dequant_blockwise(state_a[O1], state_a[O1_SCALE], (FP8_BLOCK, FP8_BLOCK))
    rel = (got.double() - want.double()).norm() / want.double().norm()
    assert rel < 0.05, rel


def test_convert_precision_mismatch_flagged(fp8_pair_factory, tmp_path):
    # FP8 base vs plain BF16 abliterated: not diffable; the only changed
    # tensor is precision_mismatch, so convert refuses an adapter but the
    # manifest it wrote records the mismatch explicitly
    from conftest import base_tensors, make_fp8_checkpoint, write_checkpoint

    def edit(ablit, base):
        ablit[O1] += 0.1

    base = base_tensors()
    ablit = {k: v.clone() for k, v in base.items()}
    edit(ablit, base)
    base_dir = make_fp8_checkpoint(tmp_path / "m-base", base)
    ablit_dir = write_checkpoint(tmp_path / "m-ablit", ablit)
    out = tmp_path / "adapter"
    with pytest.raises(ValueError, match="no tensor met tol"):
        convert.run(str(base_dir), str(ablit_dir), out=str(out))
    manifest = json.loads((out / "manifest.json").read_text())
    recs = {r["name"]: r for r in manifest["tensors"]}
    assert recs[O1]["status"] == "precision_mismatch"
    assert recs[O1]["quant_base"] == "fp8"
    assert recs[O1]["quant_ablit"] == "plain"


def test_convert_rejects_nvfp4(tmp_path):
    from conftest import base_tensors, write_checkpoint

    base = write_checkpoint(tmp_path / "n-base", base_tensors())
    ablit = write_checkpoint(tmp_path / "n-ablit", base_tensors())
    (ablit / "hf_quant_config.json").write_text(
        json.dumps({"quantization": {"quant_algo": "NVFP4"}})
    )
    with pytest.raises(ValueError, match="NVFP4"):
        convert.run(str(base), str(ablit), out=str(tmp_path / "a"))


def test_bake_fp8_requant_round_trip(fp8_pair_factory, tmp_path):
    def edit(ablit, base):
        g = torch.Generator().manual_seed(11)
        d = torch.outer(torch.randn(32, generator=g), torch.randn(32, generator=g))
        d = d / d.norm() * base[O1].norm()
        ablit[O1] += d

    pair = fp8_pair_factory(edit)
    out = tmp_path / "adapter"
    convert.run(str(pair.base_dir), str(pair.ablit_dir), out=str(out), tol=0.05)
    fused = tmp_path / "fused"
    bake.run(str(out), str(pair.base_dir), str(fused))
    fb = load_file(str(fused / "model.safetensors"))
    # edited tensor: fp8 + fresh bf16 scales, dequantized == dequant(base)+delta
    assert fb[O1].dtype == torch.float8_e4m3fn
    assert fb[O1_SCALE].dtype == torch.bfloat16
    deq = dequant_blockwise(fb[O1], fb[O1_SCALE], (FP8_BLOCK, FP8_BLOCK)).double()
    base_deq = dequant_blockwise(
        load_file(str(pair.base_dir / "model.safetensors"))[O1],
        load_file(str(pair.base_dir / "model.safetensors"))[O1_SCALE],
        (FP8_BLOCK, FP8_BLOCK),
    ).double()
    want = base_deq + (pair.ablit[O1].double() - pair.base[O1].double())
    rel = (deq - want).norm() / want.norm()
    assert rel < 0.05, rel
    # unedited fp8 tensors keep their exact original bytes
    orig = load_file(str(pair.base_dir / "model.safetensors"))
    assert torch.equal(fb[O2], orig[O2])
    assert torch.equal(
        fb["model.layers.1.self_attn.o_proj.weight_scale_inv"],
        orig["model.layers.1.self_attn.o_proj.weight_scale_inv"],
    )
    # config stays a valid fp8 config (quantization_config preserved)
    cfg = json.loads((fused / "config.json").read_text())
    assert cfg["quantization_config"]["weight_block_size"] == [FP8_BLOCK, FP8_BLOCK]


def test_bake_fp8_bfloat16_output(fp8_pair_factory, tmp_path):
    def edit(ablit, base):
        g = torch.Generator().manual_seed(11)
        d = torch.outer(torch.randn(32, generator=g), torch.randn(32, generator=g))
        d = d / d.norm() * base[O1].norm()
        ablit[O1] += d

    pair = fp8_pair_factory(edit)
    out = tmp_path / "adapter"
    convert.run(str(pair.base_dir), str(pair.ablit_dir), out=str(out), tol=0.05)
    fused = tmp_path / "fused-bf16"
    bake.run(str(out), str(pair.base_dir), str(fused), output_dtype="bfloat16")
    fb = load_file(str(fused / "model.safetensors"))
    assert fb[O1].dtype == torch.bfloat16
    # no scale tensors survive; all weights bf16
    assert not any(".weight_scale" in k for k in fb)
    assert all(not str(t.dtype).startswith("float8") for t in fb.values())
    # fused weight == dequant(base) + measured delta (no FP8 re-rounding;
    # the adapter's delta carries the FP8-pair quant noise, ~3% rel)
    orig = load_file(str(pair.base_dir / "model.safetensors"))
    base_deq = dequant_blockwise(orig[O1], orig[O1_SCALE], (FP8_BLOCK, FP8_BLOCK))
    want = base_deq + (pair.ablit[O1].double() - pair.base[O1].double()).float()
    rel = (fb[O1].float().double() - want.double()).norm() / want.double().norm()
    assert rel < 0.05, rel
    # config.json loses its quantization_config
    cfg = json.loads((fused / "config.json").read_text())
    assert "quantization_config" not in cfg
