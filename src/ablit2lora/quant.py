"""FP8 block-scale quantization helpers (safetensors-level, no model load).

GLM-5.3-Flash-style FP8 checkpoints store each 2-D weight as
`float8_e4m3fn` plus a sibling `<module>.weight_scale_inv` tensor: one
BF16 scale per weight block (128x128 for GLM-5.3-Flash), with

    W_dequant = fp8_weight * scale           (elementwise, blockwise scale)

convert dequantizes both sides of a matching-precision FP8 pair and diffs
the results in float32; bake re-fuses the adapter delta and either
requantizes the edited tensors with fresh block scales (FP8 round trip) or
writes a plain BF16 checkpoint (--output-dtype bfloat16, the
accuracy-preserving path).

NVFP4 (modelopt) checkpoints are deliberately OUT OF SCOPE for convert:
their scales are split across shards (a per-16-element FP8 `weight_scale`
plus a per-tensor `weight_scale_2`), and the FP4 grid re-quantizes every
16 elements independently, so a checkpoint-vs-checkpoint diff is dominated
by quantization-grid noise rather than by the published edit. Convert the
BF16 pair instead (zai-org/GLM-5.3-Flash-BF16); adapters stay
precision-portable and bake handles the quantized serving copy.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

E4M3_MAX = 448.0
DEFAULT_BLOCK = (128, 128)
# torch dtype names AND safetensors dtype strings (safetensors reports
# "F8_E4M3" / "BF16" / ... from get_slice().get_dtype(); torch reports
# "torch.float8_e4m3fn")
FP8_DTYPES = ("float8_e4m3fn", "float8_e5m2", "F8_E4M3", "F8_E5M2")
FP4_DTYPES = (
    "float4_e2m1fn_x2",
    "float4_e2m1fn_x2_e2m1",
    "float4_e3mfn",
    "F4_E2M1",
    "F4_E2M1FN",
    "F4_E2M1FN_X2",
    "F4_E3M",
)

_SCALE_SUFFIXES = (".weight_scale_inv", ".weight_scale")


def find_scale_key(weight_key: str, available) -> str | None:
    """Scale tensor name for a weight key, if that checkpoint ships one."""
    if not weight_key.endswith(".weight"):
        return None
    stem = weight_key[: -len(".weight")]
    for suffix in _SCALE_SUFFIXES:
        key = stem + suffix
        if key in available:
            return key
    return None


def dtype_family(dtype: str) -> str:
    """'fp8' | 'fp4' | 'plain' from a safetensors or torch dtype string."""
    d = str(dtype)
    if d in FP8_DTYPES or d.upper() in ("F8_E4M3", "F8_E5M2"):
        return "fp8"
    if any(d.startswith(p) or d.upper().startswith(p) for p in FP4_DTYPES):
        return "fp4"
    if "float8" in d.lower() or d.upper().startswith("F8_"):
        return "fp8"
    if "float4" in d.lower() or d.upper().startswith("F4_"):
        return "fp4"
    return "plain"


def block_size_for(path: str | Path) -> tuple[int, int]:
    """Weight block size for a checkpoint dir: config override or 128x128."""
    cfg = Path(path) / "config.json"
    if cfg.exists():
        try:
            data = json.loads(cfg.read_text())
        except (OSError, json.JSONDecodeError):
            return DEFAULT_BLOCK
        qc = data.get("quantization_config") or {}
        wbs = qc.get("weight_block_size")
        if isinstance(wbs, (list, tuple)) and len(wbs) == 2:
            return int(wbs[0]), int(wbs[1])
    return DEFAULT_BLOCK


def dequant_blockwise(
    w_fp8: torch.Tensor, scale: torch.Tensor, block: tuple[int, int] = DEFAULT_BLOCK
) -> torch.Tensor:
    """Dequantize an fp8 weight + blockwise scales to float32."""
    if w_fp8.ndim != 2:
        raise ValueError(
            f"fp8 dequant needs a 2-D weight, got shape {tuple(w_fp8.shape)}"
        )
    out_f, in_f = w_fp8.shape
    rb, cb = block
    nb_r = (out_f + rb - 1) // rb
    nb_c = (in_f + cb - 1) // cb
    if tuple(scale.shape) != (nb_r, nb_c):
        raise ValueError(
            f"scale shape {tuple(scale.shape)} does not match block grid "
            f"{(nb_r, nb_c)} for weight {tuple(w_fp8.shape)} with block {block}"
        )
    s = scale.float().repeat_interleave(rb, dim=0)[:out_f]
    s = s.repeat_interleave(cb, dim=1)[:, :in_f]
    return w_fp8.float() * s


def quantize_blockwise(
    w: torch.Tensor,
    block: tuple[int, int] = DEFAULT_BLOCK,
    *,
    out_dtype: torch.dtype = torch.float8_e4m3fn,
    scale_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a float weight to fp8 e4m3 + blockwise scales (absmax/448).

    Returns (fp8_weight [out, in], scale [ceil(out/rb), ceil(in/cb)]).
    """
    if w.ndim != 2:
        raise ValueError(f"fp8 quant needs a 2-D weight, got shape {tuple(w.shape)}")
    out_f, in_f = w.shape
    rb, cb = block
    nb_r = (out_f + rb - 1) // rb
    nb_c = (in_f + cb - 1) // cb
    padded = torch.zeros(nb_r * rb, nb_c * cb, dtype=torch.float32)
    padded[:out_f, :in_f] = w.float()
    blocks = padded.reshape(nb_r, rb, nb_c, cb)
    absmax = blocks.abs().amax(dim=(1, 3))
    scale = torch.where(absmax > 0, absmax / E4M3_MAX, torch.ones_like(absmax))
    scale_full = (
        scale.repeat_interleave(rb, dim=0)
        .repeat_interleave(cb, dim=1)
        .reshape(nb_r, rb, nb_c, cb)
    )
    q = (padded.reshape(nb_r, rb, nb_c, cb) / scale_full).clamp(-E4M3_MAX, E4M3_MAX)
    q = q.reshape(nb_r * rb, nb_c * cb)[:out_f, :in_f].to(out_dtype)
    return q, scale.to(scale_dtype)


def requant_fused(
    w_fp8: torch.Tensor,
    scale: torch.Tensor,
    delta: torch.Tensor,
    block: tuple[int, int] = DEFAULT_BLOCK,
    scale_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse delta into a quantized weight and requantize with fresh scales.

    delta [out, in] is added to the dequantized weight, then the sum is
    requantized to the same fp8 dtype with recomputed block scales. The
    requantization re-rounds the fused value to the FP8 grid: the result
    carries quantization noise on the edited blocks (unchanged tensors keep
    their original bytes).
    """
    fused = dequant_blockwise(w_fp8, scale, block) + delta.float()
    return quantize_blockwise(
        fused,
        block,
        out_dtype=w_fp8.dtype,
        scale_dtype=scale.dtype if scale_dtype is None else scale_dtype,
    )
