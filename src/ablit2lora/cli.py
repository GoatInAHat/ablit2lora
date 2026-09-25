"""ablit2lora command-line interface."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from . import __version__, bake, convert, serve
from . import eval as eval_mod

PRECISION_NOTE = (
    "Precision: the per-tensor diff runs in float32. BF16 pairs diff on "
    "their stored weights; FP8 pairs (float8_e4m3fn + block scales) are "
    "dequantized first, and the FP8 grid re-rounding on changed blocks "
    "surfaces as elevated residuals/flags -- that is the measurement, "
    "recorded in manifest.json. NVFP4 is out of scope for convert (FP4 "
    "grid noise dominates the diff; use the BF16 pair). Diff at the "
    "highest precision both sides share; quantize after converting."
)


def _add_convert(sub):
    p = sub.add_parser(
        "convert",
        help="diff an abliterated checkpoint against its base; emit a LoRA adapter",
        description=(
            "Diff a published abliterated checkpoint against its base and emit "
            "a PEFT LoRA adapter (lora_alpha == r, scale exactly 1.0) plus "
            "manifest.json and a ready-to-run vLLM serve command. Unchanged "
            "tensors are skipped; genuinely-retrained tensors (residual never "
            "below tol) are flagged in the manifest, never silently approximated."
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
        help="per-tensor relative Frobenius residual threshold (default 1e-3)",
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
        help="also emit best-effort factors for flagged (retrained) tensors; "
             "they stay marked in manifest.json",
    )
    p.add_argument("--out", default="adapter", help="output adapter directory")
    p.set_defaults(func=convert.run)


def _add_serve(sub):
    p = sub.add_parser(
        "serve",
        help="emit the serving config: two plain models by default, "
             "base+adapter LoRA behind --enable-lora",
        description=(
            "Default: two-model config -- base and a full (baked) "
            "abliterated checkpoint served as two plain models on the same "
            "engine, ports P and P+1. No LoRA hot-mount: engines whose "
            "model class lacks SupportsLoRA (verified: glm5_next / "
            "GLM-5.3-Flash) cannot hot-mount adapters, so the edit ships "
            "via 'ablit2lora bake' as a full checkpoint. "
            "--adapter requires --enable-lora and emits the legacy "
            "hot-mount command with a warning."
        ),
    )
    p.add_argument("--base", required=True, help="HF id or local path of the base model")
    p.add_argument(
        "--abliterated",
        default=None,
        help="full/baked abliterated checkpoint dir (two-model default path)",
    )
    p.add_argument(
        "--adapter",
        default=None,
        help="adapter dir from convert (legacy; requires --enable-lora)",
    )
    p.add_argument(
        "--enable-lora",
        action="store_true",
        help="emit the base+adapter hot-mount command (warned: glm5_next "
             "does not implement SupportsLoRA; kept for engines that do)",
    )
    p.add_argument("--name", default="abliterated", help="LoRA module name")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--extra", default=None, help="extra vLLM flags, appended verbatim")
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
            "MB-scale LoRA adapters -- one base copy, hot-swappable in vLLM. "
            "ablit2lora does not find refusal directions; the abliteration "
            "labs (audnai/penclaw, orcarouter, dealignai, huihui-ai, ...) "
            "make the checkpoints it converts."
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
