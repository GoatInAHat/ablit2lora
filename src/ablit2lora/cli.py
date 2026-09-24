"""ablit2lora command-line interface."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from . import __version__, bake, emit, extract, serve
from . import eval as eval_mod


def _add_extract(sub):
    p = sub.add_parser(
        "extract",
        help="capture contrastive activations, save refusal direction(s)",
    )
    p.add_argument("--model", required=True, help="HF id or local path of the base model")
    p.add_argument("--harmful", required=True, help="JSONL of harmful prompts")
    p.add_argument("--benign", required=True, help="JSONL of benign prompts")
    p.add_argument("--layers", default="all",
                   help="all | even | odd | 0,3,7 | 8-24 | 8-24:2 | 5-")
    p.add_argument("--position", choices=["last", "mean"], default="last",
                   help="where to read the residual stream per prompt")
    p.add_argument("--method", choices=["diff", "pca"], default="diff")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    p.add_argument("--dtype", default="auto",
                   help="auto | float32 | bfloat16 | float16")
    p.add_argument("--chat", action="store_true",
                   help="apply chat template to {messages: ...} examples")
    p.add_argument("--out", default="directions.safetensors")
    p.set_defaults(func=extract.run)


def _add_emit(sub):
    p = sub.add_parser(
        "emit", help="write the exact rank-1/rank-3 PEFT LoRA adapter")
    p.add_argument("--direction", required=True,
                   help="directions.safetensors produced by extract")
    p.add_argument("--model", required=True, help="HF id or local path of the base")
    p.add_argument("--layers", default="all")
    p.add_argument("--modules", default="o_proj,down_proj",
                   help="module suffixes; defaults cover MoE experts + shared")
    p.add_argument("--side", choices=["output", "input", "both"], default="output",
                   help="which space to orthogonalize; both => exact rank-3")
    p.add_argument("--alpha", "--scale", type=float, default=1.0,
                   help="1.0 = exact orthogonalization; >1 over-abliterates "
                        "(alias: --scale)")
    p.add_argument("--adapter-dtype", choices=["auto", "float32", "bfloat16",
                                               "float16"], default="auto")
    p.add_argument("--shared-direction", type=int, default=None,
                   help="use layer N's direction for every target (classic "
                        "single-direction abliteration)")
    p.add_argument("--out", default="adapter")
    p.set_defaults(func=emit.run)


def _add_bake(sub):
    p = sub.add_parser(
        "bake", help="fuse the adapter into full weights (quant fallback)")
    p.add_argument("--adapter", required=True, help="adapter directory from emit")
    p.add_argument("--model", required=True, help="HF id or local path of the base")
    p.add_argument("--output", required=True, help="output model directory")
    p.add_argument("--fix-bias", action="store_true",
                   help="also orthogonalize biases (rank-1 adapters only)")
    p.set_defaults(func=bake.run)


def _add_serve(sub):
    p = sub.add_parser(
        "serve", help="print the vLLM base+adapter command + compat notes")
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--name", default="abliterated", help="LoRA module name")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--extra", default=None, help="extra vLLM flags, appended verbatim")
    p.add_argument("--script", default=None, help="also write an executable script")
    p.set_defaults(func=serve.run)


def _add_eval(sub):
    p = sub.add_parser(
        "eval", help="quick refusal-rate vs perplexity harness")
    p.add_argument("--base", required=True)
    p.add_argument("--harmful", required=True, help="JSONL of harmful prompts")
    p.add_argument("--adapter", default=None, help="adapter dir from emit")
    p.add_argument("--ppl-file", default=None, help="JSONL of benign text for PPL")
    p.add_argument("--limit", type=int, default=32)
    p.add_argument("--out", default="eval.json")
    p.add_argument("--backend", choices=["auto", "vllm", "transformers"],
                   default="auto")
    p.add_argument("--device", default="auto")
    p.set_defaults(func=eval_mod.run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ablit2lora",
        description=(
            "Abliteration as a rank-1 LoRA: orthogonalize a model against its "
            "refusal direction without keeping a second copy of the weights."
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_extract(sub)
    _add_emit(sub)
    _add_bake(sub)
    _add_serve(sub)
    _add_eval(sub)
    args = parser.parse_args(argv)
    vals = {k: v for k, v in vars(args).items() if k not in ("cmd", "func")}
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        args.func(**vals)
    except Exception as e:  # noqa: BLE001
        if os.environ.get("ABLIT2LORA_DEBUG"):
            raise
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0
