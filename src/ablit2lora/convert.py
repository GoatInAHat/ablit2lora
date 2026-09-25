"""Convert an abliterated checkpoint into an exact PEFT LoRA adapter.

ablit2lora is a pure converter: the abliteration labs (audnai/penclaw,
orcarouter, dealignai, huihui-ai, ...) find refusal directions and publish
modified checkpoints; this tool diffs such a checkpoint against its base
and re-expresses the edit as a MB-scale LoRA adapter, so you keep one base
copy. It never finds directions itself.

Per matched tensor: delta = W_ablit - W_base, one exact SVD, and the
smallest rank r whose relative residual < --tol wins (rank escalation).
Unchanged tensors are skipped; tensors whose residual never drops below
tol (genuinely retrained) are flagged in manifest.json, never silently
approximated. Both checkpoints stream lazily: two tensors in RAM at a time,
never whole copies.

Precision: the diff runs in float32 on top of each side's storage dtype.
Compare checkpoints at matching precision (BF16 vs BF16, FP8 vs FP8):
quantized-vs-quantized or mixed-precision diffs carry quantization noise,
which surfaces as flagged tensors instead of a clean low-rank edit.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import torch

from . import __version__
from .adapter import write_adapter
from .quant import block_size_for, dequant_blockwise, dtype_family, find_scale_key
from .serve import build_lora_command
from .weights import LazyWeights

log = logging.getLogger(__name__)

_DTYPE_MAP = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

_TIE_PAIRS = (
    ("lm_head.weight", "model.embed_tokens.weight"),
    ("model.lm_head.weight", "model.embed_tokens.weight"),
)


def tensor_sha256(t: torch.Tensor) -> str:
    """SHA-256 over a tensor's raw storage bytes (provenance reference)."""
    raw = t.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def fit_rank(
    delta: torch.Tensor, tol: float, max_rank: int, device: str = "cpu"
) -> tuple[int, float, torch.Tensor, torch.Tensor]:
    """Best rank-k factors for delta from one exact SVD (float32).

    Returns (k, residual, A [k, in], B [out, k]) where k is the smallest
    rank with relative Frobenius residual ||delta - B A||_F / ||delta||_F
    < tol, escalating up to max_rank; if no k qualifies, k = max_rank
    (capped by min(delta.shape)) and its residual is returned. B @ A is
    the optimal rank-k approximation.
    """
    d = delta.detach().to(device=device, dtype=torch.float32)
    total = float(d.norm())
    if total == 0.0:
        raise ValueError("fit_rank called on a zero delta")
    U, S, Vh = torch.linalg.svd(d, full_matrices=False)
    tail_sq = torch.cumsum((S * S).flip(0), 0).flip(0)  # tail_sq[k] = sum_{i>=k} s_i^2

    def rel(k: int) -> float:
        return float(torch.sqrt(tail_sq[k])) / total if k < tail_sq.numel() else 0.0

    chosen = max_rank
    for cand in range(1, max_rank + 1):
        if rel(cand) < tol:
            chosen = cand
            break
    k = min(chosen, S.numel())
    A = (Vh[:k] * S[:k, None]).to(device="cpu")
    B = U[:, :k].to(device="cpu")
    return k, rel(k), A, B


def _nvfp4_blocker(path: str) -> str | None:
    """Reason string if a checkpoint dir is NVFP4-quantized (out of scope)."""
    cfg = Path(path) / "hf_quant_config.json"
    if not cfg.exists():
        return None
    try:
        data = json.loads(cfg.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    algo = str((data.get("quantization") or {}).get("quant_algo", ""))
    if "FP4" in algo.upper():
        return (
            f"{path} is an NVFP4 (modelopt) checkpoint: NVFP4 is out of scope "
            "for convert. The FP4 grid re-quantizes every 16 elements and the "
            "scales are split across shards (per-16 weight_scale plus "
            "per-tensor weight_scale_2), so a checkpoint-vs-checkpoint diff "
            "measures quantization-grid noise, not the published edit. "
            "Convert the BF16 pair instead (e.g. zai-org/GLM-5.3-Flash-BF16); "
            "the resulting adapter is precision-portable and 'bake' handles "
            "quantized serving copies."
        )
    return None


def _scale_keys(w: LazyWeights) -> set[str]:
    """Names of block-scale tensors that belong to fp8 weights."""
    names = set(w.keys())
    out: set[str] = set()
    for name in names:
        if dtype_family(w.tensors[name].dtype) == "fp8":
            sk = find_scale_key(name, names)
            if sk:
                out.add(sk)
    return out


def _tied_lm_head(name: str, orphan: LazyWeights, other: LazyWeights) -> bool:
    """True if an orphan tensor is just a tied lm_head equal to the base."""
    for a_name, b_name in _TIE_PAIRS:
        for want, partner in ((a_name, b_name), (b_name, a_name)):
            if name != want or partner not in other or want not in orphan:
                continue
            ta, tb = orphan.get(want), other.get(partner)
            return (
                ta.shape == tb.shape
                and ta.dtype == tb.dtype
                and tensor_sha256(ta) == tensor_sha256(tb)
            )
    return False


def run(
    base: str,
    abliterated: str,
    *,
    tol: float = 1e-3,
    max_rank: int = 8,
    out: str = "adapter",
    adapter_dtype: str = "float32",
    device: str = "cpu",
    include_flagged: bool = False,
) -> None:
    """Diff base vs abliterated; write adapter dir + manifest + serve cmd."""
    if max_rank < 1:
        raise ValueError("--max-rank must be >= 1")
    if adapter_dtype not in _DTYPE_MAP:
        raise ValueError(f"unknown adapter dtype {adapter_dtype!r}")
    bw = LazyWeights(base)
    aw = LazyWeights(abliterated)
    for side, path in (("base", bw.path), ("abliterated", aw.path)):
        blocker = _nvfp4_blocker(path)
        if blocker:
            raise ValueError(f"{side}: {blocker}")
    names_b = set(bw.keys())
    names_a = set(aw.keys())
    common = sorted(names_b & names_a)
    only_base = sorted(names_b - names_a)
    only_ablit = sorted(names_a - names_b)
    if not common:
        raise ValueError("checkpoints share no tensor names")
    scales_b = _scale_keys(bw)
    scales_a = _scale_keys(aw)
    block_b = block_size_for(bw.path)
    block_a = block_size_for(aw.path)

    records: list[dict] = []
    entries: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    n_unchanged = 0
    dtype_mismatch: list[str] = []

    any_fp8 = any(
        name not in scales_b and name not in scales_a
        and dtype_family(bw.tensors[name].dtype) == "fp8"
        for name in common
    )
    quant_method = (
        f"fp8_dequant_block{block_b[0]}x{block_b[1]}" if any_fp8 else "none"
    )

    for name in common:
        rb, ra = bw.tensors[name], aw.tensors[name]
        rec: dict = {
            "name": name,
            "shape": list(rb.shape),
            "dtype_base": rb.dtype,
            "dtype_ablit": ra.dtype,
        }
        if name in scales_b or name in scales_a:
            # block-scale tensors are consumed by the dequant of their
            # weight; the adapter edits the dequantized weight instead
            rec["status"] = "scale_tensor"
            records.append(rec)
            continue
        if tuple(rb.shape) != tuple(ra.shape):
            rec["status"] = "shape_mismatch"
            records.append(rec)
            continue
        fam_b, fam_a = dtype_family(rb.dtype), dtype_family(ra.dtype)
        if fam_b != fam_a:
            rec["status"] = "precision_mismatch"
            rec["quant_base"], rec["quant_ablit"] = fam_b, fam_a
            records.append(rec)
            continue
        if fam_b == "fp4":
            # NVFP4 repos are rejected up front via hf_quant_config.json;
            # this is the per-tensor fallback for unlabeled FP4 storage
            rec["status"] = "quant_unsupported"
            records.append(rec)
            continue
        if rb.dtype != ra.dtype:
            dtype_mismatch.append(name)
        Wb, Wa = bw.get(name), aw.get(name)
        sha_b, sha_a = tensor_sha256(Wb), tensor_sha256(Wa)
        rec["sha_base"], rec["sha_ablit"] = sha_b, sha_a
        same = sha_b == sha_a and rb.dtype == ra.dtype
        if fam_b == "fp8":
            sk_b = find_scale_key(name, names_b)
            sk_a = find_scale_key(name, names_a)
            if sk_b is None or sk_a is None:
                rec["status"] = "precision_mismatch"
                rec["reason"] = "fp8 weight without a block-scale tensor"
                records.append(rec)
                continue
            if block_b != block_a:
                raise ValueError(
                    f"FP8 block sizes differ: base {block_b} vs "
                    f"abliterated {block_a}"
                )
            sb, sa = bw.get(sk_b), aw.get(sk_a)
            rec["sha_scale_base"] = tensor_sha256(sb)
            rec["sha_scale_ablit"] = tensor_sha256(sa)
            if same and rec["sha_scale_base"] == rec["sha_scale_ablit"]:
                n_unchanged += 1
                continue
            if Wb.ndim != 2:
                rec["status"] = "precision_mismatch"
                rec["reason"] = "fp8 weight is not 2-D"
                records.append(rec)
                continue
            rec["quant"] = quant_method
            # diff the dequantized weights (float32); FP8-vs-FP8 pairs
            # carry quant-grid noise on changed blocks, which shows up as
            # elevated residuals/flags -- that is the measurement, and the
            # manifest records it
            Wb = dequant_blockwise(Wb, sb, block_b)
            Wa = dequant_blockwise(Wa, sa, block_a)
        elif same:
            n_unchanged += 1
            continue
        if Wb.ndim != 2 or not Wb.is_floating_point():
            d = Wa.float() - Wb.float()
            rec.update(
                status="unrepresentable",
                delta_norm=float(d.norm()),
                base_norm=float(Wb.float().norm()),
            )
            records.append(rec)
            continue
        delta = Wa.float() - Wb.float()
        dnorm = float(delta.norm())
        wnorm = float(Wb.float().norm())
        rec["delta_norm"] = dnorm
        rec["base_norm"] = wnorm
        rec["rel_delta"] = dnorm / wnorm if wnorm > 0 else None
        if dnorm == 0.0:
            n_unchanged += 1
            continue
        rank, resid, A, B = fit_rank(delta, tol, max_rank, device)
        rec["rank"], rec["residual"] = rank, resid
        module = name.removesuffix(".weight")
        if resid < tol:
            rec["status"] = "exact"
            entries[module] = (A, B)
            log.info("%s: rank %d, residual %.3g", name, rank, resid)
        else:
            rec["status"] = "flagged"
            log.info("%s: FLAGGED (best residual %.3g at rank %d)", name, resid, rank)
            if include_flagged:
                entries[module] = (A, B)
                rec["status"] = "included_best_effort"
        records.append(rec)

    for name in only_ablit:
        ra = aw.tensors[name]
        rec = {
            "name": name,
            "shape": list(ra.shape),
            "dtype_base": None,
            "dtype_ablit": ra.dtype,
        }
        if _tied_lm_head(name, aw, bw):
            rec["status"] = "tied_unchanged"
        else:
            rec["status"] = "orphan"
            rec["sha_ablit"] = tensor_sha256(aw.get(name))
        records.append(rec)
    for name in only_base:
        rb = bw.tensors[name]
        records.append(
            {
                "name": name,
                "shape": list(rb.shape),
                "dtype_base": rb.dtype,
                "dtype_ablit": None,
                "status": "missing",
            }
        )

    counts = {"common": len(common), "unchanged": n_unchanged}
    for status in (
        "exact",
        "flagged",
        "included_best_effort",
        "unrepresentable",
        "scale_tensor",
        "precision_mismatch",
        "quant_unsupported",
        "shape_mismatch",
        "orphan",
        "missing",
        "tied_unchanged",
    ):
        counts[status] = sum(1 for r in records if r["status"] == status)
    manifest = {
        "generator": "ablit2lora",
        "version": __version__,
        "base": base,
        "abliterated": abliterated,
        "base_path": bw.path,
        "abliterated_path": aw.path,
        "tol": tol,
        "max_rank": max_rank,
        "include_flagged": include_flagged,
        "dtype_mismatch": dtype_mismatch,
        "counts": counts,
        "tensors": records,
    }
    manifest["quantization"] = {
        "dequant": quant_method,
        "block_size": list(block_b) if quant_method != "none" else None,
        "nvfp4": "out of scope for convert; use the BF16 pair (see quant.py)",
    }
    manifest["deployment_note"] = (
        "when the engine's model class does not implement SupportsLoRA "
        "(e.g. glm5_next), hot-mounted LoRA is unavailable: bake the adapter "
        "into a full checkpoint (ablit2lora bake) and serve it as a second "
        "plain model (ablit2lora serve --abliterated ...)."
    )
    if dtype_mismatch:
        log.warning(
            "%d tensor(s) differ in storage dtype (first: %s): the diff carries "
            "quantization noise; prefer checkpoints at matching precision",
            len(dtype_mismatch),
            dtype_mismatch[0],
        )

    out_path = Path(out)
    if not entries:
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "manifest.json").write_text(json.dumps(manifest, indent=2))
        hint = ""
        if counts["precision_mismatch"] or counts["quant_unsupported"]:
            hint = (
                f" ({counts['precision_mismatch']} precision_mismatch + "
                f"{counts['quant_unsupported']} quant_unsupported tensors: "
                "compare checkpoints at matching precision -- dequantized FP8 "
                "vs FP8, or BF16 vs BF16)"
            )
        raise ValueError(
            f"no tensor met tol={tol} within max_rank={max_rank}: every changed "
            "tensor is flagged (retrained?), unrepresentable, or not diffable "
            "at matching precision; manifest.json records delta norms and "
            f"residuals{hint}"
        )

    uniform_rank = max(A.shape[0] for A, _ in entries.values())
    out_dir = write_adapter(
        out,
        base_model=base,
        entries=entries,
        rank=uniform_rank,
        dtype=_DTYPE_MAP[adapter_dtype],
    )
    adapter_bytes = (out_dir / "adapter_model.safetensors").stat().st_size
    cmd = build_lora_command(base, str(out_dir))
    manifest["adapter"] = {
        "path": str(out_dir),
        "uniform_rank": uniform_rank,
        "lora_alpha": uniform_rank,
        "dtype": adapter_dtype,
        "bytes": adapter_bytes,
        "target_modules": len(entries),
    }
    manifest["vllm_command"] = cmd
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(
        f"adapter: {out_dir} ({len(entries)} modules, uniform rank {uniform_rank}, "
        f"lora_alpha={uniform_rank}, dtype={adapter_dtype}, "
        f"{adapter_bytes / 1e6:.1f} MB)"
    )
    print(
        "  exact {x} | flagged {f} | unrepresentable {u} | unchanged {c} | "
        "orphan {o} | missing {m}".format(
            x=counts["exact"] + counts["included_best_effort"],
            f=counts["flagged"],
            u=counts["unrepresentable"],
            c=n_unchanged,
            o=counts["orphan"],
            m=counts["missing"],
        )
    )
    changed = [r for r in records if r.get("status") in ("flagged", "unrepresentable")]
    for r in sorted(changed, key=lambda r: -r.get("delta_norm", 0.0))[:5]:
        if r["status"] == "unrepresentable":
            print(f"  unrepresentable {r['name']}: delta_norm={r['delta_norm']:.4g}")
        else:
            print(
                f"  flagged {r['name']}: rel_delta={r.get('rel_delta'):.3g} "
                f"residual={r['residual']:.3g} at rank {r['rank']}"
            )
    if counts["flagged"] or counts["unrepresentable"]:
        print(
            "  flagged/unrepresentable tensors are NOT in the adapter "
            "(unless --include-flagged); manifest.json has the full audit trail"
        )
    print()
    print(cmd)
