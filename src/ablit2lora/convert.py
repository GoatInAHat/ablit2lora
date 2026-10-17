"""Convert an abliterated checkpoint into an exact PEFT LoRA adapter.

ablit2lora is a pure converter: the abliteration labs (audnai/penclaw,
orcarouter, dealignai, huihui-ai, ...) find refusal directions and publish
modified checkpoints; this tool diffs such a checkpoint against its base
and re-expresses the edit as a MB-scale LoRA adapter, so you keep one base
copy. It never finds directions itself.

Per matched tensor: delta = W_ablit - W_base, one exact SVD, and the
smallest rank r whose relative residual < --tol wins (rank escalation,
status "exact"). A BF16 or FP8 pair also stores full-rank rounding noise
from both sides' grids, which keeps the residual above any strict --tol at
every rank. For those tensors the noise model in noise.py estimates the
rounding-noise energy and a singular-value ceiling from the stored values;
the singular values above the ceiling are the edit, and the tensor is
accepted at that rank ("within_noise") when what remains is within
--noise-margin times the noise energy. Tensors that are only noise are
skipped ("noise_only"); tensors that are not a low-rank edit (retrained,
diffuse, or rank above --max-rank) are flagged in manifest.json with a
reason, never silently approximated. --noise-margin 0 restores the strict
--tol rule alone. Both checkpoints stream lazily: two tensors in RAM at a
time, never whole copies.

Precision: the diff runs in float32 on top of each side's storage dtype.
Compare checkpoints at matching precision (BF16 vs BF16, FP8 vs FP8). The
adapter reproduces the edit, not the rounding noise, so serving it over the
base differs from the abliterated checkpoint by about that noise.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shlex
import zlib
from dataclasses import dataclass
from pathlib import Path

import torch

from . import __version__
from .adapter import is_embedding_module, write_adapter, write_model_card
from .noise import NoiseModel
from .quant import block_size_for, dequant_blockwise, dtype_family, find_scale_key
from .weights import LazyWeights

log = logging.getLogger(__name__)

_DTYPE_MAP = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

_LM_HEAD_SUFFIX = "lm_head.weight"
_EMBED_SUFFIX = "embed_tokens.weight"


def tensor_sha256(t: torch.Tensor) -> str:
    """SHA-256 over a tensor's raw storage bytes (provenance reference)."""
    raw = t.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _decompose(delta: torch.Tensor, tol: float, max_rank: int, device: str):
    """Validate inputs and run the one exact SVD (float32) with float64 energies."""
    d = delta.detach().to(device=device, dtype=torch.float32)
    if not torch.isfinite(d).all():
        raise ValueError("fit_rank requires a finite delta")
    if not 0 < tol < 1 or not torch.isfinite(torch.tensor(tol)):
        raise ValueError("tol must be finite and between 0 and 1")
    if max_rank < 1:
        raise ValueError("max_rank must be >= 1")
    if not torch.count_nonzero(d):
        raise ValueError("fit_rank called on a zero delta")
    U, S, Vh = torch.linalg.svd(d, full_matrices=False)
    # Singular vectors/factors remain float32, but residual energy is reduced in
    # float64. Squaring large finite float32 singular values in float32 can
    # overflow and turn rank selection into inf/inf -> NaN.
    singular_sq = S.to(torch.float64).square()
    total_sq = singular_sq.sum()
    tail_sq = torch.cumsum(singular_sq.flip(0), 0).flip(0)
    return U, S, Vh, total_sq, tail_sq


def _rel(tail_sq: torch.Tensor, total_sq: torch.Tensor, k: int) -> float:
    return float(torch.sqrt(tail_sq[k] / total_sq)) if k < tail_sq.numel() else 0.0


def _factors(
    U: torch.Tensor, S: torch.Tensor, Vh: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    return (Vh[:k] * S[:k, None]).to(device="cpu"), U[:, :k].to(device="cpu")


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
    U, S, Vh, total_sq, tail_sq = _decompose(delta, tol, max_rank, device)
    chosen = max_rank
    for cand in range(1, max_rank + 1):
        if _rel(tail_sq, total_sq, cand) < tol:
            chosen = cand
            break
    k = min(chosen, S.numel())
    A, B = _factors(U, S, Vh, k)
    return k, _rel(tail_sq, total_sq, k), A, B


@dataclass
class RankSelection:
    """Outcome of noise-aware rank selection for one changed 2-D tensor.

    status is "exact", "within_noise", "noise_only" or "flagged". A, B are the
    factors to include: the accepted rank for exact/within_noise, the legacy
    best-effort rank (max_rank) for flagged, None for noise_only.
    """

    status: str
    reason: str | None
    rank: int
    residual: float
    A: torch.Tensor | None
    B: torch.Tensor | None
    signal_rank: int
    noise_floor: float
    noise_sv_ceiling: float
    spectral_gap: float | None


def select_rank(
    delta: torch.Tensor,
    tol: float,
    max_rank: int,
    noise: NoiseModel,
    *,
    noise_margin: float,
    seed: int,
    device: str = "cpu",
) -> RankSelection:
    """Rank selection that allows for storage-rounding noise in delta.

    One exact SVD, then (S singular values, total = sum S**2, E the noise
    energy from ``noise``, tau the larger of the numerical-rank floor and the
    noise singular-value ceiling, k_sig = count(S > tau)):

    1. smallest k <= max_rank with relative residual < tol: "exact" (the
       strict rule, so clean pairs behave as without the noise model);
    2. k_sig == 0: "noise_only" if total <= budget, else flagged "diffuse";
    3. k_sig > max_rank: flagged "rank";
    4. k = k_sig: "within_noise" if tail(k) <= budget, else flagged
       "residual".

    where budget = tol**2 * total + noise_margin * E.
    """
    U, S, Vh, total_sq, tail_sq = _decompose(delta, tol, max_rank, device)
    n_sv = S.numel()
    total = float(total_sq)
    energy = noise.energy()
    s0 = float(S[0])
    tau = max(
        max(delta.shape) * torch.finfo(torch.float32).eps * s0,
        noise.sv_ceiling(seed),
    )
    k_sig = int((S > tau).sum())
    budget = tol**2 * total + noise_margin * energy

    def tail(k: int) -> float:
        return float(tail_sq[k]) if k < n_sv else 0.0

    def gap(k: int) -> float | None:
        if not 0 < k < n_sv or float(S[k]) == 0.0:
            return None
        return float(S[k - 1] / S[k])

    common = dict(
        signal_rank=k_sig,
        noise_floor=(energy / total) ** 0.5,
        noise_sv_ceiling=tau / s0,
    )

    def accepted(status: str, k: int) -> RankSelection:
        A, B = _factors(U, S, Vh, k)
        return RankSelection(
            status, None, k, _rel(tail_sq, total_sq, k), A, B,
            spectral_gap=gap(k), **common,
        )

    for cand in range(1, max_rank + 1):
        if _rel(tail_sq, total_sq, cand) < tol:
            return accepted("exact", min(cand, n_sv))
    if k_sig == 0 and total <= budget:
        return RankSelection(
            "noise_only", None, 0, 1.0, None, None, spectral_gap=None, **common
        )
    if k_sig == 0:
        reason = "diffuse"
    elif k_sig > max_rank:
        reason = "rank"
    elif tail(k_sig) <= budget:
        return accepted("within_noise", k_sig)
    else:
        reason = "residual"
    k = min(max_rank, n_sv)
    A, B = _factors(U, S, Vh, k)
    return RankSelection(
        "flagged", reason, k, _rel(tail_sq, total_sq, k), A, B,
        spectral_gap=gap(k), **common,
    )


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


def _lora_serve_hint(base: str, adapter_dir: str, rank: int) -> str:
    """A ready-to-run vLLM hot-mount command for this adapter.

    A minimal, self-contained render: convert must not depend on the serving
    CLI (ablit2lora.serve). See ablit2lora.serve.build_multi_lora_command (or
    `ablit2lora serve --adapter`) for the full-featured, independently
    invocable command builder, including multi-adapter and (via
    build_sglang_multi_lora_command) SGLang variants.
    """
    adapter_path = Path(adapter_dir).resolve()
    return (
        f"vllm serve {shlex.quote(str(base))} --enable-lora "
        f"--lora-modules abliterated={shlex.quote(str(adapter_path))} "
        f"--max-lora-rank {max(8, rank)}"
    )


def _tie_word_embeddings(*paths: str) -> bool | None:
    """``tie_word_embeddings`` from the first checkpoint config.json that has it.

    Checks top-level config.json first, then a nested "text_config" (as used
    by some multimodal wrappers). None if neither checkpoint's config says
    either way.
    """
    for path in paths:
        cfg_file = Path(path) / "config.json"
        if not cfg_file.exists():
            continue
        try:
            data = json.loads(cfg_file.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "tie_word_embeddings" in data:
            return bool(data["tie_word_embeddings"])
        text_cfg = data.get("text_config")
        if isinstance(text_cfg, dict) and "tie_word_embeddings" in text_cfg:
            return bool(text_cfg["tie_word_embeddings"])
    return None


def _tied_lm_head(
    name: str, orphan: LazyWeights, other: LazyWeights, tied: bool | None
) -> bool:
    """True if an orphan tensor is just a tied lm_head equal to the other side.

    Symmetric: the orphan may be on either side. Generic over the naming
    prefix (``lm_head.weight``, ``model.lm_head.weight``,
    ``language_model.lm_head.weight``, ...): whatever precedes
    ``lm_head.weight``/``embed_tokens.weight`` must match on both tensors.
    ``tied is False`` short-circuits (the checkpoint says they are not tied);
    unknown config does not block the check, since the value-equality test
    below is already conservative. Values are compared in float32, so the
    two sides may be stored in different dtypes.
    """
    if tied is False:
        return False
    for partner in _tie_partners(name):
        if name not in orphan or partner not in other:
            continue
        ta, tb = orphan.get(name), other.get(partner)
        if ta.shape == tb.shape and torch.equal(ta.float(), tb.float()):
            return True
    return False


def _tie_partners(name: str) -> list[str]:
    """Names a tied lm_head/embedding tensor can pair with.

    HF checkpoints name the pair ``lm_head.weight`` / ``model.embed_tokens.weight``
    (the head sits outside ``model.``); some wrappers keep both under one prefix
    (``language_model.lm_head.weight`` / ``language_model.embed_tokens.weight``).
    """
    if name.endswith(_LM_HEAD_SUFFIX):
        prefix = name[: -len(_LM_HEAD_SUFFIX)]
        return [prefix + _EMBED_SUFFIX, prefix + "model." + _EMBED_SUFFIX]
    if name.endswith(_EMBED_SUFFIX):
        prefix = name[: -len(_EMBED_SUFFIX)]
        partners = [prefix + _LM_HEAD_SUFFIX]
        if prefix.endswith("model."):
            partners.append(prefix[: -len("model.")] + _LM_HEAD_SUFFIX)
        return partners
    return []


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
    noise_margin: float = 2.0,
) -> None:
    """Diff base vs abliterated; write adapter dir + manifest + serve cmd."""
    if max_rank < 1:
        raise ValueError("--max-rank must be >= 1")
    if not 0 <= noise_margin < float("inf"):
        raise ValueError("--noise-margin must be finite and >= 0")
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
    tied_word_embeddings = _tie_word_embeddings(bw.path, aw.path)

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
        noise = None
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
            # diff the dequantized weights (float32); FP8-vs-FP8 pairs carry
            # block-grid rounding noise, which the noise model accounts for
            # from the raw codes and scales
            if noise_margin > 0:
                noise = NoiseModel.from_blocks(Wb, sb, Wa, sa, block_b)
            Wb = dequant_blockwise(Wb, sb, block_b)
            Wa = dequant_blockwise(Wa, sa, block_a)
        elif same:
            n_unchanged += 1
            continue
        if Wb.ndim != 2 or not Wb.is_floating_point():
            d = Wa.float() - Wb.float()
            if not torch.count_nonzero(d):
                n_unchanged += 1
                continue
            sq = float(d.square().sum(dtype=torch.float64))
            rec.update(
                delta_norm=float(d.norm()),
                base_norm=float(Wb.float().norm()),
            )
            if (
                noise_margin > 0
                and Wb.is_floating_point()
                and Wa.is_floating_point()
                and sq <= noise_margin * NoiseModel.from_storage(Wb, Wa).energy()
            ):
                rec["status"] = "noise_only"
            else:
                rec["status"] = "unrepresentable"
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
        module = name.removesuffix(".weight")
        if noise_margin > 0:
            if noise is None:
                noise = NoiseModel.from_storage(Wb, Wa)
            sel = select_rank(
                delta, tol, max_rank, noise,
                noise_margin=noise_margin, seed=zlib.crc32(name.encode()),
                device=device,
            )
            rec["status"] = sel.status
            rec["residual"] = sel.residual
            if sel.status != "noise_only":
                rec["rank"] = sel.rank
            rec["signal_rank"] = sel.signal_rank
            rec["noise_floor"] = sel.noise_floor
            rec["noise_sv_ceiling"] = sel.noise_sv_ceiling
            if sel.reason:
                rec["reason"] = sel.reason
            if sel.spectral_gap is not None:
                rec["spectral_gap"] = sel.spectral_gap
            rank, resid, A, B = sel.rank, sel.residual, sel.A, sel.B
        else:
            rank, resid, A, B = fit_rank(delta, tol, max_rank, device)
            rec["rank"], rec["residual"] = rank, resid
            rec["status"] = "exact" if resid < tol else "flagged"
        status = rec["status"]
        if status in ("exact", "within_noise"):
            entries[module] = (A, B)
            log.info("%s: %s, rank %d, residual %.3g", name, status, rank, resid)
        elif status == "noise_only":
            log.info("%s: noise only (no edit above the storage rounding noise)", name)
        else:
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
        if _tied_lm_head(name, aw, bw, tied_word_embeddings):
            rec["status"] = "tied_unchanged"
        else:
            rec["status"] = "orphan"
            rec["sha_ablit"] = tensor_sha256(aw.get(name))
        records.append(rec)
    for name in only_base:
        rb = bw.tensors[name]
        tied = _tied_lm_head(name, bw, aw, tied_word_embeddings)
        records.append(
            {
                "name": name,
                "shape": list(rb.shape),
                "dtype_base": rb.dtype,
                "dtype_ablit": None,
                "status": "tied_unchanged" if tied else "missing",
            }
        )

    counts = {"common": len(common), "unchanged": n_unchanged}
    for status in (
        "exact",
        "within_noise",
        "noise_only",
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
        "noise_margin": noise_margin,
        "dtype_mismatch": dtype_mismatch,
        "counts": counts,
        "tensors": records,
    }
    warnings: list[str] = []
    if tied_word_embeddings:
        for module in sorted(m for m in entries if is_embedding_module(m)):
            warnings.append(
                f"{module} is edited and tie_word_embeddings is true: a runtime "
                "LoRA changes the input embedding only, so the tied lm_head "
                "output path is NOT edited. `ablit2lora bake` writes the shared "
                "weight and is exact."
            )
    for w in warnings:
        log.warning(w)
    manifest["warnings"] = warnings
    manifest["quantization"] = {
        "dequant": quant_method,
        "block_size": list(block_b) if quant_method != "none" else None,
        "nvfp4": "out of scope for convert; use the BF16 pair (see quant.py)",
    }
    manifest["deployment_note"] = (
        "when the engine's model class does not implement SupportsLoRA "
        "(e.g. glm5_next), hot-mounted LoRA is unavailable: bake the adapter "
        "into a full checkpoint (ablit2lora bake) and use the explicit "
        "two-engine fallback (ablit2lora serve --abliterated ...) if resources "
        "permit."
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
        gate = (
            f"noise-aware acceptance on (--noise-margin {noise_margin:g})"
            if noise_margin > 0
            else "noise-aware acceptance off (--noise-margin 0)"
        )
        raise ValueError(
            f"no tensor met tol={tol} within max_rank={max_rank} ({gate}): every "
            "changed tensor is flagged (retrained, or edit rank above "
            "--max-rank), noise-only, unrepresentable, or not diffable at "
            "matching precision; manifest.json records delta norms, residuals "
            f"and flag reasons{hint}"
        )

    uniform_rank = max(A.shape[0] for A, _ in entries.values())
    out_dir = write_adapter(
        out,
        base_model=base,
        entries=entries,
        rank=uniform_rank,
        dtype=_DTYPE_MAP[adapter_dtype],
    )
    write_model_card(out_dir, base_model=base)
    adapter_bytes = (out_dir / "adapter_model.safetensors").stat().st_size
    cmd = _lora_serve_hint(base, str(out_dir), uniform_rank)
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
        "  exact {x} | within-noise {w} | best-effort {b} | flagged {f} | "
        "noise-only {n} | unrepresentable {u}".format(
            x=counts["exact"],
            w=counts["within_noise"],
            b=counts["included_best_effort"],
            f=counts["flagged"],
            n=counts["noise_only"],
            u=counts["unrepresentable"],
        )
    )
    print(
        "  unchanged {c} | tied {t} | orphan {o} | missing {m}".format(
            c=n_unchanged,
            t=counts["tied_unchanged"],
            o=counts["orphan"],
            m=counts["missing"],
        )
    )
    changed = [
        r
        for r in records
        if r.get("status") in ("flagged", "included_best_effort", "unrepresentable")
    ]
    for r in sorted(changed, key=lambda r: -r.get("delta_norm", 0.0))[:5]:
        if r["status"] == "unrepresentable":
            print(f"  unrepresentable {r['name']}: delta_norm={r['delta_norm']:.4g}")
        else:
            why = f" ({r['reason']})" if r.get("reason") else ""
            print(
                f"  {r['status']} {r['name']}{why}: "
                f"rel_delta={r.get('rel_delta'):.3g} "
                f"residual={r['residual']:.3g} at rank {r['rank']}"
            )
    if counts["flagged"] or counts["unrepresentable"]:
        print(
            "  flagged/unrepresentable tensors are NOT in the adapter "
            "(unless --include-flagged); manifest.json has the full audit trail"
        )
    if counts["included_best_effort"]:
        print(
            "  best-effort tensors ARE in the adapter but did not meet the "
            "acceptance rule: their residual is recorded in manifest.json"
        )
    print()
    print(cmd)
