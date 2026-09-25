"""Emit the serving config for the abliterated model.

Two modes:

- DEFAULT (two-model): serve the base and a full abliterated checkpoint as
  two plain models on the same engine -- no LoRA hot-mount involved. This
  is the deployment path for engines whose model class does not implement
  SupportsLoRA (verified for glm5_next, the GLM-5.3-Flash arch): produce
  the adapter, bake it into a full checkpoint, and serve that as a second
  plain model. Clients pick the behavior by pointing at the model/port.
- --enable-lora (legacy, warned): the vLLM base+adapter hot-mount command.
  Kept for architectures whose engine class DOES implement SupportsLoRA;
  on glm5_next this command will not work and the tool says so.
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
)

TWO_MODEL_NOTE = (
    "two-model deployment (default)\n"
    "- Same engine, two plain models: base on --port, the baked abliterated "
    "checkpoint on --port+1.\n"
    "- No LoRA hot-mount: engines whose model class lacks SupportsLoRA "
    "(verified: glm5_next / GLM-5.3-Flash) cannot hot-mount adapters, so "
    "the edit ships as a full checkpoint via 'ablit2lora bake'.\n"
    "- Per request: point the client (model name / base_url port) at the "
    "abliterated model to use the converted behavior; the base stays "
    "untouched.\n"
)

LORA_WARNING = (
    "warning: hot-mounted LoRA requires the engine's model class to "
    "implement SupportsLoRA. glm5_next (GLM-5.3-Flash) does NOT, so this "
    "command will not hot-mount the adapter on the stock engine -- it is "
    "emitted for architectures that do support it. The supported path for "
    "glm5_next is: bake the adapter into a full checkpoint and serve it as "
    "a second plain model (ablit2lora serve --abliterated ...)."
)


def build_lora_command(
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
        "vllm",
        "serve",
        str(base),
        "--enable-lora",
        "--lora-modules",
        f"{name}={adapter_path}",
        "--max-lora-rank",
        str(max(8, rank)),
        "--host",
        str(host),
        "--port",
        str(port),
    ]
    if extra:
        cmd.append(extra)
    return " \\\n  ".join(shlex.quote(c) for c in cmd)


def build_two_model_commands(
    base: str,
    abliterated: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    extra: str | None = None,
) -> tuple[str, str, int]:
    """Two plain-model serve commands: base and the baked abliterated copy."""
    abliterated_path = Path(abliterated)
    if not (abliterated_path / "config.json").exists():
        raise FileNotFoundError(
            f"{abliterated_path} does not look like a model directory "
            "(no config.json); pass the baked checkpoint from "
            "'ablit2lora bake'"
        )
    second_port = port + 1

    def one(model: str, p: int) -> str:
        cmd = [
            "vllm",
            "serve",
            str(model),
            "--host",
            str(host),
            "--port",
            str(p),
        ]
        if extra:
            cmd.append(extra)
        return " \\\n  ".join(shlex.quote(c) for c in cmd)

    return one(base, port), one(str(abliterated_path), second_port), second_port


def render_two_model(base_cmd: str, ablit_cmd: str, second_port: int) -> str:
    return (
        f"# two plain models, same engine -- no LoRA hot-mount\n"
        f"# base:\n{base_cmd}\n\n"
        f"# abliterated (baked), port {second_port}:\n{ablit_cmd}\n\n"
        f"# pick the behavior per request by pointing the client at the "
        f"model on port {second_port}.\n"
    )


def run(
    base: str,
    abliterated: str | None = None,
    adapter: str | None = None,
    enable_lora: bool = False,
    name: str = "abliterated",
    host: str = "0.0.0.0",
    port: int = 8000,
    extra: str | None = None,
    script: str | None = None,
) -> None:
    """Emit the serving config (two-model default, LoRA behind a flag)."""
    if adapter is not None and not enable_lora:
        raise ValueError(
            "--adapter needs --enable-lora to emit the hot-mount command. "
            "Note: glm5_next (GLM-5.3-Flash) does not implement SupportsLoRA, "
            "so hot-mounted LoRA is unavailable on that engine -- prefer the "
            "default two-model path: bake the adapter "
            "(ablit2lora bake) and pass --abliterated <baked-model-dir>."
        )
    if adapter is not None:
        cmd = build_lora_command(base, adapter, name=name, host=host, port=port, extra=extra)
        print(cmd)
        print()
        print(LORA_WARNING)
        print()
        print(NOTES)
    elif abliterated is not None:
        base_cmd, ablit_cmd, second_port = build_two_model_commands(
            base, abliterated, host=host, port=port, extra=extra
        )
        text = render_two_model(base_cmd, ablit_cmd, second_port)
        print(text)
        print(TWO_MODEL_NOTE)
    else:
        raise ValueError(
            "nothing to serve: pass --abliterated <full/baked checkpoint> "
            "(recommended) or --adapter <adapter-dir> --enable-lora"
        )

    if script:
        p = Path(script)
        p.parent.mkdir(parents=True, exist_ok=True)
        body = cmd if adapter is not None else text
        p.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n" + body + "\n")
        p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print(f"wrote {p}")
