"""Bit-exactness of the LoRA algebra (CPU, torch-only)."""

from __future__ import annotations

import torch

from ablit2lora.directions import load_directions, save_directions
from ablit2lora.lora import lora_delta


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm()


def test_output_side_exact_fp32():
    g = torch.Generator().manual_seed(0)
    W = torch.randn(16, 8, generator=g)
    d = _unit(torch.randn(16, generator=g))
    A, B = lora_delta(W, d_out=d, alpha=1.0)
    assert A.shape == (1, 8) and B.shape == (16, 1)
    expected = W - torch.outer(d, d @ W)
    delta = (W + B @ A - expected).abs().max()
    assert delta < 1e-6, delta


def test_output_side_alpha_scales():
    g = torch.Generator().manual_seed(1)
    W = torch.randn(12, 10, generator=g)
    d = _unit(torch.randn(12, generator=g))
    A, B = lora_delta(W, d_out=d, alpha=0.37)
    expected = W - 0.37 * torch.outer(d, d @ W)
    assert (W + B @ A - expected).abs().max() < 1e-6


def test_input_side_exact_fp32():
    g = torch.Generator().manual_seed(2)
    W = torch.randn(12, 16, generator=g)
    d = _unit(torch.randn(16, generator=g))
    A, B = lora_delta(W, d_in=d, alpha=1.0)
    assert A.shape == (1, 16) and B.shape == (12, 1)
    expected = W - torch.outer(W @ d, d)
    assert (W + B @ A - expected).abs().max() < 1e-6


def test_both_side_exact_rank3():
    g = torch.Generator().manual_seed(3)
    W = torch.randn(12, 10, generator=g)
    do = _unit(torch.randn(12, generator=g))
    di = _unit(torch.randn(10, generator=g))
    A, B = lora_delta(W, d_out=do, d_in=di, alpha=1.0)
    assert A.shape == (3, 10) and B.shape == (12, 3)
    Po = torch.eye(12) - torch.outer(do, do)
    Pi = torch.eye(10) - torch.outer(di, di)
    expected = Po @ W @ Pi
    assert (W + B @ A - expected).abs().max() < 1e-5


def test_bf16_adapter_close():
    g = torch.Generator().manual_seed(4)
    W = torch.randn(16, 8, generator=g)
    d = _unit(torch.randn(16, generator=g))
    A, B = lora_delta(W, d_out=d, alpha=1.0, dtype=torch.bfloat16)
    got = W.to(torch.bfloat16).float() + (B.float() @ A.float())
    want = (W - torch.outer(d, d @ W)).to(torch.bfloat16).float()
    assert torch.allclose(got, want, atol=0.02, rtol=0.0)


def test_direction_file_roundtrip(tmp_path):
    dirs = {i: torch.randn(32) for i in (0, 2, 5)}
    p = save_directions(tmp_path / "dirs.safetensors", model_id="test-model",
                        directions=dirs, scores={0: 0.5, 2: 0.4, 5: 0.3},
                        meta={"position": "last"})
    loaded, meta = load_directions(p)
    assert set(loaded) == {0, 2, 5}
    for i in dirs:
        assert torch.allclose(loaded[i], dirs[i], atol=1e-6)
    assert meta["model"] == "test-model" and meta["position"] == "last"
