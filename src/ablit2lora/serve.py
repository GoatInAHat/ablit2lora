"""Emit serving launch commands for abliterated-model adapters.

The standard interface: one engine loads the base model once and hot-mounts
one or more PEFT LoRA adapters (the stock-equivalent/abliterated variants and
any others for the same base), selected per request via the OpenAI ``model``
field. This module emits that command for vLLM and for SGLang, plus the
dynamic-REST runbook for hot-swapping adapters without a restart.

A fallback remains available: ``--abliterated`` launches two independent
engines for a baked checkpoint. This needs two full model allocations and is
not a shared-weight deployment.
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

TWO_ENGINE_NOTE = (
    "two-engine fallback\n"
    "- This launches two independent vLLM processes: the base on --port and "
    "the baked checkpoint on --port+1.\n"
    "- It loads two full model allocations. It does not share weights and may "
    "not fit on hardware sized for one model.\n"
)

LORA_WARNING = (
    "warning: hot-mounted LoRA requires the engine's model class to "
    "implement SupportsLoRA. glm5_next (GLM-5.3-Flash) does NOT, so this "
    "command will not hot-mount the adapter on the stock engine. Use the "
    "two-engine baked fallback when resources permit."
)


def _command(words: list[str], extra: str | None = None) -> str:
    """Render one shell command; parse ``extra`` as shell words first."""
    if extra:
        words.extend(shlex.split(extra))
    separator = " " + chr(92) + "\n  "
    return separator.join(shlex.quote(word) for word in words)


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
    return _command(
        [
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
        ],
        extra,
    )


def build_two_engine_commands(
    base: str,
    abliterated: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    extra: str | None = None,
) -> tuple[str, str, int]:
    """Two independent plain-model serve commands (not shared weights)."""
    abliterated_path = Path(abliterated)
    if not (abliterated_path / "config.json").exists():
        raise FileNotFoundError(
            f"{abliterated_path} does not look like a model directory "
            "(no config.json); pass the baked checkpoint from 'ablit2lora bake'"
        )
    second_port = port + 1

    def one(model: str, selected_port: int) -> str:
        return _command(
            [
                "vllm",
                "serve",
                str(model),
                "--host",
                str(host),
                "--port",
                str(selected_port),
            ],
            extra,
        )

    return one(base, port), one(str(abliterated_path), second_port), second_port


# Backward-compatible import name from 0.3.0. The implementation is correctly
# documented as two engines from 0.3.1 onward.
build_two_model_commands = build_two_engine_commands


def render_two_engine(base_cmd: str, ablit_cmd: str, second_port: int) -> str:
    return (
        "# two independent engines; two full model allocations\n"
        f"# base:\n{base_cmd} &\nbase_pid=$!\n\n"
        f"# abliterated (baked), port {second_port}:\n"
        f"{ablit_cmd} &\nabliterated_pid=$!\n\n"
        "trap 'kill \"$base_pid\" \"$abliterated_pid\" 2>/dev/null || true' "
        "EXIT INT TERM\n"
        "wait \"$base_pid\" \"$abliterated_pid\"\n"
    )


# Backward-compatible import name from 0.3.0.
render_two_model = render_two_engine


def run(
    base: str,
    abliterated: str | None = None,
    adapter: str | None = None,
    enable_lora: bool = False,
    name: str | None = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    extra: str | None = None,
    script: str | None = None,
) -> None:
    """Emit one of the two explicit serving modes."""
    selected = sum(value is not None for value in (abliterated, adapter))
    if selected != 1:
        raise ValueError("choose exactly one of --abliterated or --adapter")
    if enable_lora and adapter is None:
        raise ValueError("--enable-lora requires --adapter")
    if adapter is not None and not enable_lora:
        raise ValueError("--adapter requires --enable-lora")

    if adapter is not None:
        text = build_lora_command(
            base, adapter, name=name or "abliterated", host=host, port=port, extra=extra
        )
        note = f"{LORA_WARNING}\n\n{NOTES}"
    else:
        base_cmd, ablit_cmd, second_port = build_two_engine_commands(
            base, abliterated or "", host=host, port=port, extra=extra
        )
        text = render_two_engine(base_cmd, ablit_cmd, second_port)
        note = TWO_ENGINE_NOTE

    print(text)
    print()
    print(note)

    if script:
        path = Path(script)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n" + text + "\n")
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print(f"wrote {path}")
