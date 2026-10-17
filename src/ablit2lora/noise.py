"""Storage-rounding noise model for noise-aware rank acceptance in convert.

A published (base, abliterated) pair is stored in BF16, FP16 or FP8, so
delta = W_ablit - W_base is the low-rank edit plus full-rank rounding noise
from both sides' storage grids. That noise puts a floor under the relative
residual that no rank can get below, so a fixed ``--tol`` cannot separate
"edit" from "rounding". This module estimates, from the stored values alone:

* the total noise energy ``E`` (sum of per-element rounding variances), and
* a ceiling ``tau`` on the singular values that rounding noise alone can
  produce, so singular values above it are signal.

The per-element model treats each stored value as rounded to a grid with
spacing ``s`` (the ulp at that value): the rounding error is uniform on
(-s/2, s/2) with variance s**2 / 12, and the two sides are independent, so
v = (s_base**2 + s_ablit**2) / 12.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .quant import expand_block_scale

_SAFETY = 1.1
_DRAWS = 2
_SKETCH = 6
_POWER_ITERS = 4


def grid_spacing(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Spacing (ulp) of the ``dtype`` grid at each value of ``x``, as float32.

    ``2**floor(log2(max(|x|, tiny))) * eps``. Clamping to ``tiny`` gives zero
    and subnormals the subnormal spacing ``tiny * eps``.
    """
    fi = torch.finfo(dtype)
    a = x.float().abs().clamp_min(fi.tiny)
    _, e = torch.frexp(a)
    return torch.ldexp(torch.full_like(a, fi.eps), e - 1)


def block_grid_spacing(
    q: torch.Tensor, scale: torch.Tensor, block: tuple[int, int]
) -> torch.Tensor:
    """Spacing of an FP8 block-scale weight on its dequantized values (float32).

    The grid is the FP8 grid of the raw codes ``q`` times the block scale.
    """
    s = expand_block_scale(scale, tuple(q.shape), block)
    return grid_spacing(q, q.dtype) * s


def effective_dtype(t: torch.Tensor) -> torch.dtype:
    """Coarsest float grid that holds every value of ``t`` exactly.

    A float32 tensor whose values all lie on the BF16 grid (low 16 bits zero)
    or are exactly representable in FP16 was computed at that precision and
    upcast on save; its rounding noise is that of the coarser grid.
    """
    if t.dtype != torch.float32:
        return t.dtype
    if not (t.contiguous().view(torch.int32) & 0xFFFF).any():
        return torch.bfloat16
    if t.half().float().eq(t).all():
        return torch.float16
    return torch.float32


def coarser(a: torch.dtype, b: torch.dtype) -> torch.dtype:
    return a if torch.finfo(a).eps >= torch.finfo(b).eps else b


def _sigma_max(m: torch.Tensor, gen: torch.Generator) -> float:
    """Top singular value by randomized subspace iteration (local RNG)."""
    rows, cols = m.shape
    q = min(_SKETCH, rows, cols)
    basis = torch.linalg.qr(m @ torch.randn(cols, q, generator=gen)).Q
    for _ in range(_POWER_ITERS):
        basis = torch.linalg.qr(m.T @ basis).Q
        basis = torch.linalg.qr(m @ basis).Q
    return float(torch.linalg.svdvals(basis.T @ m)[0])


@dataclass(frozen=True)
class NoiseModel:
    """Per-element grid spacings of the two stored tensors (float32, same shape)."""

    spacing_base: torch.Tensor
    spacing_ablit: torch.Tensor

    @classmethod
    def from_storage(cls, base: torch.Tensor, ablit: torch.Tensor) -> NoiseModel:
        """Plain-dtype pair, both sides on the coarser of their effective grids."""
        grid = coarser(effective_dtype(base), effective_dtype(ablit))
        return cls(grid_spacing(base, grid), grid_spacing(ablit, grid))

    @classmethod
    def from_blocks(
        cls,
        base_q: torch.Tensor,
        base_scale: torch.Tensor,
        ablit_q: torch.Tensor,
        ablit_scale: torch.Tensor,
        block: tuple[int, int],
    ) -> NoiseModel:
        return cls(
            block_grid_spacing(base_q, base_scale, block),
            block_grid_spacing(ablit_q, ablit_scale, block),
        )

    def energy(self) -> float:
        """Expected squared Frobenius norm of the rounding noise, in float64."""
        total = self.spacing_base.square().sum(dtype=torch.float64)
        total += self.spacing_ablit.square().sum(dtype=torch.float64)
        return float(total / 12)

    def sv_ceiling(self, seed: int) -> float:
        """Seeded Monte-Carlo ceiling on the top singular value of the noise.

        Simulates N = U_a * spacing_ablit + U_b * spacing_base with
        U ~ Uniform(-1/2, 1/2), takes the larger of two draws, and adds a
        10% safety factor. Following the actual spacing profile keeps
        outlier rows/columns and FP8 per-block scales from hiding a real
        edit, which a closed-form bound would.
        """
        if self.spacing_base.ndim != 2:
            raise ValueError("sv_ceiling needs 2-D spacings")
        gen = torch.Generator().manual_seed(seed)
        best = 0.0
        for _ in range(_DRAWS):
            u_a = torch.rand(self.spacing_ablit.shape, generator=gen) - 0.5
            u_b = torch.rand(self.spacing_base.shape, generator=gen) - 0.5
            sim = u_a * self.spacing_ablit + u_b * self.spacing_base
            best = max(best, _sigma_max(sim, gen))
        return _SAFETY * best
