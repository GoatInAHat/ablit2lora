"""Print (or write) the vLLM command that serves base + abliteration adapter.

The base model stays bit-identical on disk; clients pick the behavior per
request via the LoRA module name, so removal is a per-request choice with no
restart and no second copy of the weights.
"""

from __future__ import annotations

import json
import shlex
import stat
from pathlib import Path

NOTES = (
    "vLLM / quantization compatibility notes\n"
    "- BF16 / FP16 base + LoRA: fully supported.\n"
    "- FP8 (W8A8) base + LoRA: supported for major architectures on recent "
    "vLLM;\n"
    "  if vllm serve rejects the adapter, upgrade vLLM or use ablit2lora "
    "bake.\n"
    "- NVFP4 / MXFP4 (ModelOpt) base + LoRA: newer and architecture-limited;\n"
    "  LoRA-over-NVFP4 is young -- check your vLLM release notes, and treat "
    "bake\n"
    "  as the reliable fallback on any quantized base.\n"
    "- Converted adapters use lora_alpha == r (LoRA scale exactly 1.0), so "
    "served\n"
    "  weights equal base + the measured delta in the serving dtype.\n"
    "- Hot swap: pass model=<name> per request to use the abliterated "
    "behavior;\n"
    "  omit it to use the base. No restart, no second copy.\n"
)


def build_command(
    base: str,
    adapter: str,
    name: str = "abliterated",
    host: str = "0.0.0.0",
    port: int = 8000,
    extra: str | None = None,
) -> str:
    adapter_path = Path(adapter).resolve()
    cfg_file = adapter_path / "adapter_config.json"
    if not cfg_file.exists():
        raise FileNotFoundError(f"{cfg_file} not found")
    rank = int(json.loads(cfg_file.read_text()).get("r", 1))
    cmd = [
        "vllm", "serve", str(base),
        "--enable-lora",
        "--lora-modules", f"{name}={adapter_path}",
        "--max-lora-rank", str(max(8, rank)),
        "--host", str(host), "--port", str(port),
    ]
    if extra:
        cmd.append(extra)
    return " \\\n  ".join(shlex.quote(c) for c in cmd)


def run(
    base: str,
    adapter: str,
    name: str = "abliterated",
    host: str = "0.0.0.0",
    port: int = 8000,
    extra: str | None = None,
    script: str | None = None,
) -> None:
    """Emit the serve command and compatibility notes."""
    cmd = build_command(base, adapter, name=name, host=host, port=port, extra=extra)
    print(cmd)
    print()
    print(NOTES)
    if script:
        p = Path(script)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n" + cmd + "\n")
        p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print(f"wrote {p}")
