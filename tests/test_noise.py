"""Storage-noise-aware acceptance in convert (CPU, synthetic, no downloads)."""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import load_file

from ablit2lora import convert
from ablit2lora.noise import (
    NoiseModel,
    block_grid_spacing,
    coarser,
    effective_dtype,
    grid_spacing,
)
from conftest import FP8_BLOCK, fp8_quantize_weight, make_fp8_checkpoint, write_checkpoint

NAME = "model.layers.0.self_attn.o_proj.weight"
MOD = "model.layers.0.self_attn.o_proj"
OUT_F, IN_F = 128, 96


def _weight(seed: int = 0, sigma: float = 0.02) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(OUT_F, IN_F, generator=g) * sigma


def _abliterate(w: torch.Tensor, frac: float = 0.02, seed: int = 1):
    """w - alpha r (r^T w): a rank-1 output-side edit of frac * ||w||."""
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(w.shape[0], generator=g)
    r = r / r.norm()
    proj = r @ w
    alpha = frac * w.norm() / proj.norm()
    return w - alpha * torch.outer(r, proj), r


def _bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16)


def _write_pair(tmp_path, base: dict, ablit: dict, *, fp8: bool = False):
    if fp8:
        return (
            make_fp8_checkpoint(tmp_path / "base", base),
            make_fp8_checkpoint(tmp_path / "ablit", ablit),
        )
    return write_checkpoint(tmp_path / "base", base), write_checkpoint(tmp_path / "ablit", ablit)


def _run(base_dir, ablit_dir, tmp_path, **kw):
    out = tmp_path / "adapter"
    convert.run(str(base_dir), str(ablit_dir), out=str(out), **kw)
    return out, json.loads((out / "manifest.json").read_text())


def _manifest_only(base_dir, ablit_dir, tmp_path, **kw):
    out = tmp_path / "adapter"
    with pytest.raises(ValueError, match="no tensor met tol"):
        convert.run(str(base_dir), str(ablit_dir), out=str(out), **kw)
    return json.loads((out / "manifest.json").read_text())


def _rec(manifest, name=NAME):
    return {r["name"]: r for r in manifest["tensors"]}[name]


def _delta_from_adapter(out, module=MOD) -> torch.Tensor:
    state = load_file(str(out / "adapter_model.safetensors"))
    a = state[f"base_model.model.{module}.lora_A.weight"]
    b = state[f"base_model.model.{module}.lora_B.weight"]
    return b @ a


def _bf16_rank1_pair(tmp_path, *, frac: float = 0.02):
    base = _bf16(_weight())
    edited, _ = _abliterate(base.float(), frac)
    ablit = _bf16(edited)
    true = edited - base.float()
    base_dir, ablit_dir = _write_pair(tmp_path, {NAME: base}, {NAME: ablit})
    return base_dir, ablit_dir, true


def _next_up(a: torch.Tensor) -> torch.Tensor:
    """Next representable value above positive finite ``a`` (float32 result)."""
    if a.dtype == torch.float32:
        return torch.nextafter(a, torch.full_like(a, float("inf")))
    bits = {2: torch.int16, 1: torch.uint8}[a.element_size()]
    return (a.view(bits) + 1).view(a.dtype).float()


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float8_e4m3fn]
)
def test_grid_spacing_matches_next_representable_value(dtype):
    g = torch.Generator().manual_seed(3)
    fi = torch.finfo(dtype)
    pow2 = 2.0 ** torch.arange(-4, 6, dtype=torch.float32)
    sub = torch.tensor([0.0, fi.tiny / 2, fi.tiny * fi.eps, fi.tiny, fi.tiny * 1.5])
    rand = (torch.rand(512, generator=g) * 2 - 1) * 8
    x = torch.cat([pow2, -pow2, sub, rand]).to(dtype)
    x = x[x.float().abs() < 0.9 * fi.max]
    a = x.float().abs().to(dtype)
    want = _next_up(a) - a.float()
    assert torch.equal(grid_spacing(x, dtype), want.float())


def test_grid_spacing_subnormal_and_zero_use_tiny_times_eps():
    fi = torch.finfo(torch.bfloat16)
    sp = grid_spacing(torch.tensor([0.0, fi.tiny / 4]), torch.bfloat16)
    assert torch.all(sp == fi.tiny * fi.eps)


def test_effective_dtype_detects_upcast_storage():
    w = _weight()
    assert effective_dtype(_bf16(w)) == torch.bfloat16
    assert effective_dtype(_bf16(w).float()) == torch.bfloat16
    assert effective_dtype(w.half().float()) == torch.float16
    assert effective_dtype(w) == torch.float32
    assert coarser(torch.bfloat16, torch.float32) == torch.bfloat16
    assert coarser(torch.float16, torch.bfloat16) == torch.bfloat16


def test_noise_model_uses_coarser_effective_grid_for_both_sides():
    base = _bf16(_weight())
    edited, _ = _abliterate(base.float())
    stored = NoiseModel.from_storage(base, _bf16(edited))
    upcast = NoiseModel.from_storage(base, _bf16(edited).float())
    assert upcast.energy() == pytest.approx(stored.energy(), rel=1e-12)
    off_grid = NoiseModel.from_storage(base, edited)
    assert off_grid.energy() == pytest.approx(stored.energy(), rel=0.02)
    expected = float(
        (
            grid_spacing(base, torch.bfloat16).double() ** 2
            + grid_spacing(edited, torch.bfloat16).double() ** 2
        ).sum()
        / 12
    )
    assert off_grid.energy() == pytest.approx(expected, rel=1e-12)


def test_fp8_spacing_is_code_grid_times_block_scale():
    q, scale = fp8_quantize_weight(_weight())
    sp = block_grid_spacing(q, scale, (FP8_BLOCK, FP8_BLOCK))
    assert sp.shape == q.shape
    s00 = float(scale[0, 0].float())
    code = q[0, 0].float().abs().to(torch.float8_e4m3fn)
    assert float(sp[0, 0]) == pytest.approx(
        float(_next_up(code) - code.float()) * s00, rel=1e-6
    )


def test_noise_ceiling_is_seeded_and_above_pure_noise():
    base = _bf16(_weight())
    edited, _ = _abliterate(base.float())
    nm = NoiseModel.from_storage(base, _bf16(edited))
    assert nm.sv_ceiling(7) == nm.sv_ceiling(7)
    real_noise = (_bf16(edited).float() - edited)
    top = float(torch.linalg.svdvals(real_noise)[0])
    assert nm.sv_ceiling(7) > top


def test_bf16_pair_default_settings_within_noise(tmp_path):
    base_dir, ablit_dir, true = _bf16_rank1_pair(tmp_path)
    out, manifest = _run(base_dir, ablit_dir, tmp_path)
    rec = _rec(manifest)
    assert rec["status"] == "within_noise"
    assert rec["rank"] == 1 and rec["signal_rank"] == 1
    assert rec["residual"] > manifest["tol"]
    assert 0 < rec["noise_floor"] < 0.3
    assert 0 < rec["noise_sv_ceiling"] < 0.5
    assert rec["spectral_gap"] > 5
    assert manifest["noise_margin"] == 2.0
    assert manifest["counts"]["within_noise"] == 1
    fit = _delta_from_adapter(out)
    cos = float((fit * true).sum() / (fit.norm() * true.norm()))
    assert cos > 0.999
    assert float((fit - true).norm() / true.norm()) < 0.03


def test_bf16_pair_strict_margin_zero_flags(tmp_path):
    base_dir, ablit_dir, _ = _bf16_rank1_pair(tmp_path)
    manifest = _manifest_only(base_dir, ablit_dir, tmp_path, noise_margin=0)
    rec = _rec(manifest)
    assert rec["status"] == "flagged"
    assert "signal_rank" not in rec and "reason" not in rec
    assert manifest["noise_margin"] == 0
    assert manifest["counts"]["within_noise"] == 0


def test_bf16_pair_stored_as_float32_uses_effective_grid(tmp_path):
    base = _bf16(_weight())
    edited, _ = _abliterate(base.float())
    base_dir, ablit_dir = _write_pair(
        tmp_path, {NAME: base}, {NAME: _bf16(edited).float()}
    )
    _, manifest = _run(base_dir, ablit_dir, tmp_path)
    rec = _rec(manifest)
    assert rec["status"] == "within_noise" and rec["rank"] == 1
    assert manifest["dtype_mismatch"] == [NAME]


def test_fp8_independent_quant_pair_within_noise(tmp_path):
    w = _weight()
    edited, _ = _abliterate(w, frac=0.15)
    base_dir, ablit_dir = _write_pair(tmp_path, {NAME: w}, {NAME: edited}, fp8=True)
    out, manifest = _run(base_dir, ablit_dir, tmp_path)
    rec = _rec(manifest)
    assert rec["status"] == "within_noise"
    assert rec["rank"] == 1
    assert rec["quant"].startswith("fp8_dequant_block")
    assert rec["residual"] > 0.05
    assert rec["noise_floor"] > 0.05
    fit = _delta_from_adapter(out)
    true = edited - w
    cos = float((fit * true).sum() / (fit.norm() * true.norm()))
    assert cos > 0.99


def test_noise_only_tensor_is_skipped(tmp_path):
    base = _bf16(_weight())
    ulp = grid_spacing(base, torch.bfloat16)
    g = torch.Generator().manual_seed(9)
    jittered = base.float() + 0.25 * ulp * torch.randn(OUT_F, IN_F, generator=g)
    other, _ = _abliterate(_bf16(_weight(5)).float())
    tensors_base = {NAME: base, "model.layers.1.self_attn.o_proj.weight": _bf16(_weight(5))}
    tensors_ablit = {
        NAME: _bf16(jittered),
        "model.layers.1.self_attn.o_proj.weight": _bf16(other),
    }
    base_dir, ablit_dir = _write_pair(tmp_path, tensors_base, tensors_ablit)
    out, manifest = _run(base_dir, ablit_dir, tmp_path)
    rec = _rec(manifest)
    assert rec["status"] == "noise_only"
    assert "rank" not in rec
    assert manifest["counts"]["noise_only"] == 1
    state = load_file(str(out / "adapter_model.safetensors"))
    assert not any(MOD in k for k in state)
    assert any("layers.1.self_attn.o_proj" in k for k in state)


def test_retrained_tensor_stays_flagged_even_with_noise_allowance(tmp_path):
    base = _bf16(_weight())
    g = torch.Generator().manual_seed(11)
    ablit = _bf16(base.float() + 0.5 * torch.randn(OUT_F, IN_F, generator=g) * 0.02)
    base_dir, ablit_dir = _write_pair(tmp_path, {NAME: base}, {NAME: ablit})
    manifest = _manifest_only(base_dir, ablit_dir, tmp_path)
    rec = _rec(manifest)
    assert rec["status"] == "flagged"
    assert rec["reason"] == "rank"
    assert rec["signal_rank"] > manifest["max_rank"]


def test_heteroscedastic_noise_does_not_inflate_rank(tmp_path):
    w = _weight()
    w[:, 7] *= 20
    base = _bf16(w)
    edited, _ = _abliterate(base.float())
    base_dir, ablit_dir = _write_pair(tmp_path, {NAME: base}, {NAME: _bf16(edited)})
    _, manifest = _run(base_dir, ablit_dir, tmp_path)
    rec = _rec(manifest)
    assert rec["status"] == "within_noise"
    assert rec["rank"] == 1


def test_select_rank_diffuse_and_residual_reasons():
    sp = torch.full((OUT_F, IN_F), 1e-3)
    nm = NoiseModel(sp, sp)
    ceiling = nm.sv_ceiling(0)
    energy = nm.energy()
    q, _ = torch.linalg.qr(torch.randn(OUT_F, IN_F, generator=torch.Generator().manual_seed(2)))
    flat = q * (0.95 * ceiling)
    assert float((flat**2).sum()) > 2 * energy
    sel = convert.select_rank(flat, 1e-3, 8, nm, noise_margin=2.0, seed=0)
    assert (sel.status, sel.reason, sel.signal_rank) == ("flagged", "diffuse", 0)
    assert sel.A.shape[0] == 8

    small = q * (0.1 * (energy / IN_F) ** 0.5)
    sel = convert.select_rank(small, 1e-3, 8, nm, noise_margin=2.0, seed=0)
    assert sel.status == "noise_only" and sel.A is None and sel.rank == 0

    qv, _ = torch.linalg.qr(
        torch.randn(IN_F, IN_F, generator=torch.Generator().manual_seed(4))
    )
    spectrum = torch.zeros(IN_F)
    spectrum[:2] = torch.tensor([50.0, 20.0]) * ceiling
    spectrum[2:82] = 0.95 * ceiling
    spread = (q * spectrum) @ qv.T
    assert float((spectrum[2:] ** 2).sum()) > 2 * energy
    sel = convert.select_rank(spread, 1e-3, 8, nm, noise_margin=2.0, seed=0)
    assert (sel.status, sel.reason, sel.signal_rank) == ("flagged", "residual", 2)


def test_exact_rule_still_wins_on_clean_pair():
    g = torch.Generator().manual_seed(5)
    delta = torch.randn(OUT_F, 2, generator=g) @ torch.randn(2, IN_F, generator=g)
    sp = torch.full((OUT_F, IN_F), 1e-9)
    sel = convert.select_rank(delta, 1e-3, 8, NoiseModel(sp, sp), noise_margin=2.0, seed=0)
    assert sel.status == "exact" and sel.rank == 2 and sel.reason is None
    k, resid, a, b = convert.fit_rank(delta, 1e-3, 8)
    assert (k, resid) == (sel.rank, sel.residual)
    assert torch.equal(a, sel.A) and torch.equal(b, sel.B)


def test_include_flagged_best_effort_counted_separately(tmp_path, capsys):
    base = _bf16(_weight())
    g = torch.Generator().manual_seed(11)
    ablit = _bf16(base.float() + 0.5 * torch.randn(OUT_F, IN_F, generator=g) * 0.02)
    base_dir, ablit_dir = _write_pair(tmp_path, {NAME: base}, {NAME: ablit})
    out, manifest = _run(base_dir, ablit_dir, tmp_path, include_flagged=True)
    text = capsys.readouterr().out
    assert _rec(manifest)["status"] == "included_best_effort"
    assert "exact 0 | within-noise 0 | best-effort 1 | flagged 0" in text
    assert "best-effort tensors ARE in the adapter" in text
    assert "included_best_effort" in text


def test_no_entries_error_mentions_noise_aware_path(tmp_path):
    base_dir, ablit_dir, _ = _bf16_rank1_pair(tmp_path)
    with pytest.raises(ValueError, match=r"no tensor met tol=.*noise-aware acceptance off"):
        convert.run(
            str(base_dir), str(ablit_dir), out=str(tmp_path / "a"), noise_margin=0
        )
    with pytest.raises(ValueError, match="--noise-margin must be"):
        convert.run(str(base_dir), str(ablit_dir), out=str(tmp_path / "b"), noise_margin=-1)


def test_one_d_zero_diff_across_dtypes_is_unchanged_not_unrepresentable(tmp_path):
    ln = "model.layers.0.input_layernorm.weight"
    base = {NAME: _bf16(_weight()), ln: torch.ones(32, dtype=torch.bfloat16)}
    edited, _ = _abliterate(base[NAME].float())
    ablit = {NAME: _bf16(edited), ln: torch.ones(32, dtype=torch.float32)}
    base_dir, ablit_dir = _write_pair(tmp_path, base, ablit)
    _, manifest = _run(base_dir, ablit_dir, tmp_path)
    assert ln not in {r["name"] for r in manifest["tensors"]}
    assert manifest["counts"]["unrepresentable"] == 0
    assert manifest["counts"]["unchanged"] == 1


def test_one_d_change_within_rounding_is_noise_only_else_unrepresentable(tmp_path):
    ln = "model.layers.0.input_layernorm.weight"
    base = {NAME: _bf16(_weight()), ln: torch.ones(32, dtype=torch.bfloat16)}
    edited, _ = _abliterate(base[NAME].float())
    ablit = {NAME: _bf16(edited), ln: torch.full((32,), 1.0001, dtype=torch.float32)}
    base_dir, ablit_dir = _write_pair(tmp_path, base, ablit)
    _, manifest = _run(base_dir, ablit_dir, tmp_path)
    assert _rec(manifest, ln)["status"] == "noise_only"
    strict = _manifest_only(base_dir, ablit_dir, tmp_path / "strict", noise_margin=0)
    assert _rec(strict, ln)["status"] == "unrepresentable"
    ablit[ln] = torch.full((32,), 1.5, dtype=torch.float32)
    base_dir2, ablit_dir2 = _write_pair(tmp_path / "big", base, ablit)
    _, big = _run(base_dir2, ablit_dir2, tmp_path / "big")
    assert _rec(big, ln)["status"] == "unrepresentable"


def test_tied_lm_head_only_in_base_is_tied_unchanged(tmp_path):
    emb = "model.embed_tokens.weight"
    head = "lm_head.weight"
    table = _bf16(torch.randn(100, 32, generator=torch.Generator().manual_seed(6)) * 0.1)
    base = {NAME: _bf16(_weight()), emb: table, head: table.clone()}
    edited, _ = _abliterate(base[NAME].float())
    ablit = {NAME: _bf16(edited), emb: table.float()}
    base_dir, ablit_dir = _write_pair(tmp_path, base, ablit)
    for d in (base_dir, ablit_dir):
        (d / "config.json").write_text(json.dumps({"tie_word_embeddings": True}))
    _, manifest = _run(base_dir, ablit_dir, tmp_path)
    assert _rec(manifest, head)["status"] == "tied_unchanged"
    assert manifest["counts"]["missing"] == 0
    assert manifest["counts"]["tied_unchanged"] == 1

    ablit[emb] = (table.float() + 0.5)
    base_dir2, ablit_dir2 = _write_pair(tmp_path / "x", base, ablit)
    for d in (base_dir2, ablit_dir2):
        (d / "config.json").write_text(json.dumps({"tie_word_embeddings": True}))
    _, edited_manifest = _run(base_dir2, ablit_dir2, tmp_path / "x")
    assert _rec(edited_manifest, head)["status"] == "missing"
