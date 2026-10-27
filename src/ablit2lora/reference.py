"""Offline dense exports: measure the effective delta, not an ideal float edit.

No EXL3 decoder/quantizer or engine is implemented here. Callers must export
dequantized matrices with the verified quantizer/runtime, and supply their
provenance separately. The CPU forward is a simulation, not a kernel proof.
"""

from __future__ import annotations

import math

import torch


def _matrix(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim != 2 or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating 2-D matrix")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value.detach().to(device="cpu", dtype=torch.float64)


def _error(error: torch.Tensor, reference: torch.Tensor) -> dict:
    # Metric reductions must not overflow/underflow in simulated FP16/BF16.
    error = error.double()
    reference = reference.double()
    norm = float(error.norm())
    denom = float(reference.norm())
    return {
        "norm": norm,
        "relative": norm / denom if denom else None,
        "rms": float(error.square().mean().sqrt()),
        "max_abs": float(error.abs().max()),
    }


def audit_pair(
    base_live: torch.Tensor,
    reference: torch.Tensor,
    *,
    ranks: list[int],
    activations: torch.Tensor | None = None,
    factor_dtype: torch.dtype = torch.float32,
    arithmetic_dtype: torch.dtype = torch.float32,
    max_elements: int = 4_194_304,
) -> tuple[dict, dict[int, tuple[torch.Tensor, torch.Tensor]]]:
    """SVD rank sweep on Delta_eff = reference - base_live, without denoising.

    Returns a JSON-safe audit and cast CPU (A, B) factors for independent
    serialization/reload checks. Weight-optimal ranks need not be optimal for
    activation error. Activations are rows [tokens, input]; the staged CPU
    simulation rounds each GEMM result and the final sum to arithmetic_dtype.
    Neither this simulation nor its dense baseline emulates EXL3 kernels, TP
    reduction order, CUDA graphs, mHC, MTP, or whole-model logits.
    """
    if base_live.numel() > max_elements or reference.numel() > max_elements:
        raise ValueError("dense audit exceeds max_elements; use a bounded export")
    if factor_dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("unsupported factor dtype")
    if arithmetic_dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("unsupported arithmetic dtype")
    base = _matrix(base_live, "base_live")
    target = _matrix(reference, "reference")
    if base.shape != target.shape or min(base.shape) == 0:
        raise ValueError("base/reference must have equal, nonempty shapes")
    if not ranks or any(type(r) is not int or not 1 <= r <= min(base.shape) for r in ranks):
        raise ValueError("ranks must be nonempty integers within the matrix dimensions")
    x = None
    if activations is not None:
        if activations.numel() > max_elements:
            raise ValueError("activation audit exceeds max_elements")
        x = _matrix(activations, "activations")
        if x.shape[1] != base.shape[1] or x.shape[0] == 0:
            raise ValueError("activation shape must be [nonzero tokens, input]")
        if x.shape[0] * base.shape[0] > max_elements:
            raise ValueError("activation output audit exceeds max_elements")
    delta = target - base
    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    energy = float(delta.square().sum())
    report = {
        "definition": "Delta_eff = reference_dequant - base_live_dequant",
        "shape": list(base.shape),
        "factor_dtype": str(factor_dtype),
        "arithmetic_dtype": str(arithmetic_dtype),
        "delta_norm": math.sqrt(energy),
        "reference_norm": float(target.norm()),
        "singular_values": S.tolist(),
        "cpu_forward_only": True,
        "runtime_validated": False,
        "ranks": [],
    }
    factors = {}
    for rank in sorted(set(ranks)):
        A = (S[:rank, None] * Vh[:rank]).to(factor_dtype).contiguous()
        B = U[:, :rank].to(factor_dtype).contiguous()
        if not torch.isfinite(A).all() or not torch.isfinite(B).all():
            raise ValueError("factor cast overflowed; choose a wider factor dtype")
        ideal = U[:, :rank] @ (S[:rank, None] * Vh[:rank])
        serialized = B.double() @ A.double()
        remainder = delta - serialized
        entry = {
            "rank": rank,
            "factor_bytes": (A.numel() + B.numel()) * A.element_size(),
            "best_tail_norm": float(S[rank:].norm()),
            "best_tail_relative": float(S[rank:].norm()) / math.sqrt(energy) if energy else None,
            "serialized_residual_vs_delta": _error(remainder, delta),
            "serialized_residual_vs_reference": _error(remainder, target),
            "factor_cast_error_vs_ideal": _error(serialized - ideal, ideal),
            "recovered_energy_fraction": 1 - float(remainder.square().sum()) / energy
            if energy else None,
        }
        if x is not None:
            exact_y = x @ target.T
            cast_x = x.to(arithmetic_dtype)
            dense_y = (cast_x @ target.to(arithmetic_dtype).T).to(arithmetic_dtype)
            stock_y = (cast_x @ base.to(arithmetic_dtype).T).to(arithmetic_dtype)
            low_y = (cast_x @ A.to(arithmetic_dtype).T).to(arithmetic_dtype)
            low_y = (low_y @ B.to(arithmetic_dtype).T).to(arithmetic_dtype)
            candidate_y = (stock_y + low_y).to(arithmetic_dtype)
            if not all(torch.isfinite(y).all() for y in (dense_y, stock_y, candidate_y)):
                raise ValueError("simulated forward overflowed; choose a wider arithmetic dtype")
            entry["activation"] = {
                "tokens": x.shape[0],
                "real_arithmetic_residual": _error(x @ remainder.T, exact_y),
                "simulated_staged_vs_dense": _error(candidate_y.double() - dense_y, dense_y),
                "simulated_dense_vs_float64": _error(dense_y.double() - exact_y, exact_y),
                "simulated_dropped_vs_dense": _error(stock_y.double() - dense_y, dense_y),
            }
        factors[rank] = (A, B)
        report["ranks"].append(entry)
    return report, factors


def logit_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    """Paired teacher-forced [positions, vocab] exports; no auto-pass threshold.

    Token IDs, positions, tokenizer and execution provenance must be matched by
    the caller. Top-1 agreement alone does not establish behavioral equivalence.
    """
    ref = _matrix(reference, "reference logits")
    actual = _matrix(candidate, "candidate logits")
    if ref.shape != actual.shape or min(ref.shape) == 0:
        raise ValueError("logit shapes must match and be nonempty")
    lp = ref.log_softmax(-1)
    lq = actual.log_softmax(-1)
    kl = (lp.exp() * (lp - lq)).sum(-1).clamp_min(0)
    return {
        "positions": ref.shape[0],
        "mean_kl_ref_to_candidate": float(kl.mean()),
        "max_kl_ref_to_candidate": float(kl.max()),
        "top1_agreement": float((ref.argmax(-1) == actual.argmax(-1)).double().mean()),
        "logit_error": _error(actual - ref, ref),
        "logprob_max_abs": float((lq - lp).abs().max()),
    }
