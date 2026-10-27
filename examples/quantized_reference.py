"""Audit bounded, dense exports from an independently quantized reference.

This script does not quantize EXL3, read a live model or prove engine support.
See docs/quantized-reference.md for the upstream reference prerequisites.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

from ablit2lora.adapter import read_adapter, write_adapter
from ablit2lora.reference import audit_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", required=True,
                        help="safetensors: base_live, reference, activations")
    parser.add_argument("--provenance", required=True,
                        help="JSON export/recipe provenance; not certified")
    parser.add_argument("--out", required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--factor-dtype", choices=["float32", "bfloat16", "float16"],
                        default="float32")
    parser.add_argument("--arithmetic-dtype", choices=["float32", "bfloat16", "float16"],
                        default="float32")
    parser.add_argument("--max-elements", type=int, default=4_194_304)
    args = parser.parse_args()
    with safe_open(args.pair, framework="pt", device="cpu") as f:
        values = {}
        for name in ("base_live", "reference", "activations"):
            if name not in f.keys() and name == "activations":
                continue
            shape = f.get_slice(name).get_shape()
            if len(shape) != 2 or shape[0] * shape[1] > args.max_elements:
                raise ValueError(f"{name}: input exceeds dense 2-D audit budget")
            values[name] = f.get_tensor(name)
    provenance_bytes = Path(args.provenance).read_bytes()
    report, factors = audit_pair(**values, ranks=args.ranks,
                                 factor_dtype=getattr(torch, args.factor_dtype),
                                 arithmetic_dtype=getattr(torch, args.arithmetic_dtype),
                                 max_elements=args.max_elements)
    report["provenance"] = json.loads(provenance_bytes)
    report["provenance_sha256"] = hashlib.sha256(provenance_bytes).hexdigest()
    report["provenance_verified"] = False
    # Hash the bounded exported tensors, not an unbounded/possibly sparse input file.
    report["export_tensor_sha256"] = {
        key: hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        for key, t in values.items()
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    for entry in report["ranks"]:
        rank = entry["rank"]
        path = write_adapter(out / f"rank-{rank}", base_model="offline-export-not-deployable",
                             entries={"exported_writer": factors[rank]}, rank=rank,
                             dtype=getattr(torch, args.factor_dtype))
        restored, _ = read_adapter(path)
        A, B = restored["exported_writer"]
        expected_A, expected_B = factors[rank]
        if not torch.equal(A, expected_A) or not torch.equal(B, expected_B):
            raise ValueError("serialized factor reload differs from audited factors")
        residual = (values["reference"].double() - values["base_live"].double()
                    - B.double() @ A.double())
        entry["reloaded_residual_norm"] = float(residual.norm())
        entry["adapter_sha256"] = hashlib.sha256(
            (path / "adapter_model.safetensors").read_bytes()).hexdigest()
    (out / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(out / "audit.json")


if __name__ == "__main__":
    main()
