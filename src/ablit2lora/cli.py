"""ablit2lora command-line interface."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from . import __version__, bake, convert, serve
from . import eval as eval_mod

PRECISION_NOTE = (
    "Precision: the per-tensor diff runs in float32 on each side's stored "
    "values. A BF16 or FP8 pair stores the edit plus rounding noise from both "
    "storage grids, so the diff is low-rank signal plus a full-rank noise "
    "floor that no rank can fit. convert estimates that floor from the "
    "stored values (per-element grid spacing; FP8 pairs are dequantized "
    "first and use the block-scaled grid) and accepts a tensor at the rank "
    "of its singular values above the noise ceiling when what remains is "
    "within the noise budget (status within_noise); --noise-margin 0 turns "
    "this off. The adapter reproduces the edit, not the rounding noise, so "
    "serving it over the base can differ from the abliterated checkpoint by "
    "about that noise. NVFP4 is out of scope for convert (FP4 grid noise "
    "dominates the diff; use the BF16 pair). Diff at the highest precision "
    "both sides share; quantize after converting."
)


def _add_convert(sub):
    p = sub.add_parser(
        "convert",
        help="diff an abliterated checkpoint against its base; emit a LoRA adapter",
        description=(
            "Diff a published abliterated checkpoint against its base and emit "
            "a PEFT LoRA adapter (lora_alpha == r, scale exactly 1.0) plus "
            "manifest.json and a ready-to-run vLLM serve command. Unchanged "
            "tensors are skipped; tensors whose change is not a low-rank edit "
            "(retrained, or rank above --max-rank) are flagged in the manifest, "
            "never silently approximated."
        ),
        epilog=PRECISION_NOTE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--base", required=True, help="HF id or local path of the base model")
    p.add_argument(
        "--abliterated",
        required=True,
        help="HF id or local path of the published abliterated checkpoint",
    )
    p.add_argument(
        "--tol",
        type=float,
        default=1e-3,
        help="strict relative Frobenius residual threshold (default 1e-3): a "
             "tensor whose rank-k residual is below it is accepted as exact. "
             "On BF16/FP8 pairs storage rounding noise keeps the residual above "
             "this at every rank (4-15%% of the diff for BF16, 50-150%% for "
             "FP8 in synthetic pairs); those tensors go through the "
             "noise-aware rule instead (see --noise-margin). Loosening --tol "
             "is not a substitute: it does not separate the edit from the "
             "noise",
    )
    p.add_argument(
        "--noise-margin",
        type=float,
        default=2.0,
        help="multiple of the estimated storage rounding-noise energy allowed "
             "as residual when a tensor fails --tol (default 2.0). The "
             "tensor is accepted as within_noise at the number of singular "
             "values above the noise ceiling, if the remaining energy is "
             "within tol^2 * total + margin * noise energy. 0 disables the "
             "noise allowance: strict --tol only, as in v0.4.0",
    )
    p.add_argument(
        "--max-rank",
        type=int,
        default=8,
        help="rank escalation cap per tensor (default 8)",
    )
    p.add_argument(
        "--adapter-dtype",
        choices=["float32", "bfloat16", "float16"],
        default="float32",
        help="storage dtype of the LoRA factors (default float32; bfloat16 "
             "halves the size but can miss small tolerances)",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="device for the SVDs: cpu | cuda | cuda:N | mps",
    )
    p.add_argument(
        "--include-flagged",
        action="store_true",
        help="also emit best-effort factors (rank --max-rank) for flagged "
             "tensors; they stay marked included_best_effort in manifest.json",
    )
    p.add_argument("--out", default="adapter", help="output adapter directory")
    p.set_defaults(func=convert.run)


def _add_serve(sub):
    p = sub.add_parser(
        "serve",
        help="emit a native multi-LoRA vLLM/SGLang command, or a two-engine fallback",
        description=(
            "Standard mode: one engine, one loaded base, one or more hot-mounted "
            "PEFT adapters (the stock-equivalent, abliterated, and any other "
            "variants for the same base), selected per request via the OpenAI "
            "model field. Requires the engine's model class to implement "
            "SupportsLoRA -- see 'Supported base families' in the README. A "
            "baked checkpoint fallback starts two independent engines and two "
            "full model allocations."
        ),
    )
    p.add_argument("--base", required=True, help="HF id or local path of the base model")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--adapter",
        action="append",
        metavar="NAME=PATH",
        default=None,
        help="PEFT adapter from convert, as name=path; repeat for multiple "
             "adapters on one engine (requires --enable-lora)",
    )
    mode.add_argument(
        "--abliterated",
        default=None,
        help="full/baked checkpoint dir (two independent engine fallback)",
    )
    p.add_argument(
        "--enable-lora",
        action="store_true",
        help="emit the base+adapter(s) hot-mount command (requires --adapter)",
    )
    p.add_argument(
        "--engine",
        choices=["vllm", "sglang"],
        default="vllm",
        help="serving engine for the --adapter command (default: vllm)",
    )
    p.add_argument(
        "--max-loras",
        type=int,
        default=None,
        help="override the adapter slot count (default: number of --adapter)",
    )
    p.add_argument(
        "--max-lora-rank",
        type=int,
        default=None,
        help="override the max LoRA rank (default: the largest r across "
             "each adapter_config.json)",
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--extra", default=None, help="extra engine flags, parsed as shell words"
    )
    p.add_argument("--script", default=None, help="also write an executable script")
    p.set_defaults(func=serve.run)


def _add_eval(sub):
    p = sub.add_parser(
        "eval",
        help="verify base+adapter reproduces the abliterated model's behavior",
    )
    p.add_argument("--base", required=True)
    p.add_argument("--harmful", required=True, help="JSONL of harmful prompts")
    p.add_argument(
        "--abliterated",
        default=None,
        help="optional abliterated reference checkpoint to evaluate alongside",
    )
    p.add_argument("--adapter", default=None, help="adapter dir from convert")
    p.add_argument("--ppl-file", default=None, help="JSONL of benign text for PPL")
    p.add_argument("--limit", type=int, default=32)
    p.add_argument("--out", default="eval.json")
    p.add_argument(
        "--backend", choices=["auto", "vllm", "transformers"], default="auto"
    )
    p.add_argument("--device", default="auto")
    p.set_defaults(func=eval_mod.run)


def _add_bake(sub):
    p = sub.add_parser(
        "bake",
        help="fuse the adapter into a full serving checkpoint (the "
             "deployment path when the engine cannot hot-mount LoRA)",
        description=(
            "Fuse W <- W + B @ A shard by shard into a full checkpoint at "
            "the base precision. On an FP8 block-scale base, edited tensors "
            "are requantized with fresh block scales by default "
            "(--output-dtype auto); --output-dtype bfloat16 writes a plain "
            "BF16 checkpoint instead (no re-rounding, ~2x disk)."
        ),
    )
    p.add_argument("--adapter", required=True, help="adapter directory from convert")
    p.add_argument("--model", required=True, help="HF id or local path of the base")
    p.add_argument("--output", required=True, help="output model directory")
    p.add_argument(
        "--output-dtype",
        choices=["auto", "bfloat16"],
        default="auto",
        help="auto: keep the base precision (FP8 base -> requantized FP8); "
             "bfloat16: dequantize everything to a plain BF16 checkpoint "
             "(accuracy-preserving, ~2x disk)",
    )
    p.set_defaults(func=bake.run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ablit2lora",
        description=(
            "Pure converter: turn published abliterated checkpoints into "
            "MB-scale LoRA adapters, and emit native multi-adapter vLLM/SGLang "
            "serving recipes. ablit2lora does not find refusal directions; "
            "the abliteration labs (audnai/penclaw, orcarouter, dealignai, "
            "huihui-ai, ...) make the checkpoints it converts."
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_convert(sub)
    _add_serve(sub)
    _add_eval(sub)
    _add_bake(sub)
    args = parser.parse_args(argv)
    vals = {k: v for k, v in vars(args).items() if k not in ("cmd", "func")}
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        args.func(**vals)
    except Exception as e:  # noqa: BLE001
        if os.environ.get("ABLIT2LORA_DEBUG"):
            raise
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0
