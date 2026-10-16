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

def _is_glm53_flash(base: str) -> bool:
    """True if ``base`` names the GLM-5.3-Flash family (arch ``glm5_next``).

    Matched case-insensitively against common spellings seen in HF ids and
    local paths. Scopes the GLM-specific LoRA caveat below so it is not
    printed for the Llama / Qwen / GLM-4-MoE bases that DO work with native
    ``--enable-lora`` today (see "Supported base families" in the README);
    GLM-5.3-Flash is the one known gap.
    """
    low = base.lower()
    return any(marker in low for marker in ("glm-5.3-flash", "glm5next", "glm5_next"))


GLM53_FLASH_LORA_WARNING = (
    "warning: glm5_next (GLM-5.3-Flash) does not declare upstream SupportsLoRA "
    "yet (tracking vllm-project/vllm#56327), so the stock vLLM engine will not "
    "hot-mount this adapter on this base. ablit2lora ships an experimental, "
    "UNVERIFIED registration plugin (ablit2lora.vllm_plugins.glm5next_lora) "
    "that attempts to add the declaration out-of-tree -- see its module "
    "docstring before relying on it. 'ablit2lora bake' plus the two-engine "
    "fallback (--abliterated) is the reliable path on this base meanwhile."
)


def _command(words: list[str], extra: str | None = None) -> str:
    """Render one shell command; parse ``extra`` as shell words first."""
    if extra:
        words.extend(shlex.split(extra))
    separator = " " + chr(92) + "\n  "
    return separator.join(shlex.quote(word) for word in words)


AdapterSpec = tuple[str, Path]


def _parse_adapter_specs(adapters: list[str] | list[tuple[str, str]]) -> list[AdapterSpec]:
    """Normalize ``name=path`` strings (or ``(name, path)`` pairs); validate names."""
    specs: list[AdapterSpec] = []
    for item in adapters:
        if isinstance(item, str):
            name, sep, raw_path = item.partition("=")
            if not sep:
                raise ValueError(f"adapter spec {item!r} must be name=path")
        else:
            name, raw_path = item
        name = name.strip()
        if not name:
            raise ValueError(f"adapter spec {item!r} has an empty name")
        specs.append((name, Path(raw_path).expanduser().resolve()))
    if not specs:
        raise ValueError("at least one --adapter is required")
    names = [n for n, _ in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate adapter names: {names}")
    return specs


def _adapter_rank(adapter_path: Path) -> int:
    cfg_file = adapter_path / "adapter_config.json"
    if not cfg_file.exists():
        raise FileNotFoundError(f"{cfg_file} not found")
    return int(json.loads(cfg_file.read_text()).get("r", 1))


def _resolved_max_lora_rank(specs: list[AdapterSpec], max_lora_rank: int | None) -> int:
    if max_lora_rank is not None:
        return max_lora_rank
    return max(8, max(_adapter_rank(path) for _, path in specs))


def _validate_multi_lora_limits(
    specs: list[AdapterSpec], max_loras: int | None, max_lora_rank: int | None
) -> None:
    """Reject an explicit --max-loras/--max-lora-rank too small for ``specs``.

    Without this, an under-sized explicit override would only surface as the
    serving engine refusing to start once the emitted command is actually run.
    """
    if max_loras is not None and max_loras < len(specs):
        raise ValueError(
            f"--max-loras {max_loras} is less than the {len(specs)} adapter(s) "
            "supplied"
        )
    if max_lora_rank is not None:
        largest = max(_adapter_rank(path) for _, path in specs)
        if max_lora_rank < largest:
            raise ValueError(
                f"--max-lora-rank {max_lora_rank} is less than the largest "
                f"adapter rank ({largest}) found in the supplied "
                "adapter_config.json files"
            )


def build_multi_lora_command(
    base: str,
    adapters: list[str] | list[tuple[str, str]],
    *,
    max_loras: int | None = None,
    max_lora_rank: int | None = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    extra: str | None = None,
) -> str:
    """Static multi-adapter vLLM launch: one engine, N hot-mounted adapters.

    ``adapters`` is one or more ``name=path`` specs (the stock-equivalent,
    abliterated, and any other variants for the same base). ``--max-loras``
    defaults to the adapter count; ``--max-lora-rank`` defaults to the
    largest ``r`` across their ``adapter_config.json`` files.
    """
    specs = _parse_adapter_specs(adapters)
    _validate_multi_lora_limits(specs, max_loras, max_lora_rank)
    rank = _resolved_max_lora_rank(specs, max_lora_rank)
    n = max_loras if max_loras is not None else len(specs)
    return _command(
        [
            "vllm",
            "serve",
            str(base),
            "--enable-lora",
            "--lora-modules",
            *(f"{name}={path}" for name, path in specs),
            "--max-loras",
            str(n),
            "--max-lora-rank",
            str(rank),
            "--host",
            str(host),
            "--port",
            str(port),
        ],
        extra,
    )


def build_dynamic_lora_runbook(
    host: str = "0.0.0.0",
    port: int = 8000,
    example_name: str = "abliterated",
    example_path: str = "/path/to/adapter",
) -> str:
    """vLLM's runtime load/unload REST runbook (no engine restart)."""
    base_url = f"http://{host}:{port}"
    load = _command(
        ["curl", "-s", "-X", "POST", f"{base_url}/v1/load_lora_adapter",
         "-H", "Content-Type: application/json",
         "-d", json.dumps({"lora_name": example_name, "lora_path": example_path})]
    )
    unload = _command(
        ["curl", "-s", "-X", "POST", f"{base_url}/v1/unload_lora_adapter",
         "-H", "Content-Type: application/json",
         "-d", json.dumps({"lora_name": example_name})]
    )
    switch = _command(
        ["curl", "-s", f"{base_url}/v1/completions",
         "-H", "Content-Type: application/json",
         "-d", json.dumps({"model": example_name, "prompt": "..."})]
    )
    return (
        "dynamic LoRA reload (vLLM, no restart)\n"
        "- start the engine with VLLM_ALLOW_RUNTIME_LORA_UPDATING=True (export "
        "it in the same shell as the launch command above).\n"
        "- load (or replace) an adapter at runtime:\n"
        f"{load}\n"
        "- unload one that is no longer needed:\n"
        f"{unload}\n"
        "- per-request routing: the OpenAI `model` field selects the adapter "
        "by the exact name passed to --lora-modules / load_lora_adapter; omit "
        "it, or pass the base model's own served name, to use the untouched "
        "base:\n"
        f"{switch}\n"
    )


def build_sglang_multi_lora_command(
    base: str,
    adapters: list[str] | list[tuple[str, str]],
    *,
    max_loras: int | None = None,
    max_lora_rank: int | None = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    extra: str | None = None,
) -> str:
    """Static multi-adapter SGLang launch: same PEFT adapters, a second engine.

    Proves the adapter is a standard, engine-portable artifact rather than a
    vLLM-specific one: identical ``name=path`` specs, SGLang's own flag names.
    """
    specs = _parse_adapter_specs(adapters)
    _validate_multi_lora_limits(specs, max_loras, max_lora_rank)
    rank = _resolved_max_lora_rank(specs, max_lora_rank)
    n = max_loras if max_loras is not None else len(specs)
    return _command(
        [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            str(base),
            "--enable-lora",
            "--lora-paths",
            *(f"{name}={path}" for name, path in specs),
            "--max-loras-per-batch",
            str(n),
            "--max-lora-rank",
            str(rank),
            "--host",
            str(host),
            "--port",
            str(port),
        ],
        extra,
    )


def build_sglang_dynamic_lora_runbook(
    base: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    example_name: str = "abliterated",
    example_path: str = "/path/to/adapter",
) -> str:
    """SGLang's runtime load/unload REST runbook (no engine restart)."""
    base_url = f"http://{host}:{port}"
    load = _command(
        ["curl", "-s", "-X", "POST", f"{base_url}/load_lora_adapter",
         "-H", "Content-Type: application/json",
         "-d", json.dumps({"lora_name": example_name, "lora_path": example_path})]
    )
    unload = _command(
        ["curl", "-s", "-X", "POST", f"{base_url}/unload_lora_adapter",
         "-H", "Content-Type: application/json",
         "-d", json.dumps({"lora_name": example_name})]
    )
    switch = _command(
        ["curl", "-s", f"{base_url}/v1/completions",
         "-H", "Content-Type: application/json",
         "-d", json.dumps({"model": f"{base}:{example_name}", "prompt": "..."})]
    )
    return (
        "dynamic LoRA reload (SGLang, no restart)\n"
        "- load (or replace) an adapter at runtime (note: no /v1 prefix):\n"
        f"{load}\n"
        "- unload one that is no longer needed:\n"
        f"{unload}\n"
        "- per-request routing on the OpenAI-compatible endpoints: the "
        "`model` field uses base-model:adapter-name syntax; the native "
        "/generate endpoint instead takes a top-level lora_path field:\n"
        f"{switch}\n"
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
    adapter: list[str] | None = None,
    enable_lora: bool = False,
    engine: str = "vllm",
    max_loras: int | None = None,
    max_lora_rank: int | None = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    extra: str | None = None,
    script: str | None = None,
) -> None:
    """Emit one of the two explicit serving modes."""
    selected = sum(bool(value) for value in (abliterated, adapter))
    if selected != 1:
        raise ValueError("choose exactly one of --abliterated or --adapter")
    if enable_lora and not adapter:
        raise ValueError("--enable-lora requires --adapter")
    if adapter and not enable_lora:
        raise ValueError("--adapter requires --enable-lora")
    if engine not in ("vllm", "sglang"):
        raise ValueError(f"unknown --engine {engine!r}: choose 'vllm' or 'sglang'")

    if adapter:
        specs = _parse_adapter_specs(adapter)
        if engine == "vllm":
            text = build_multi_lora_command(
                base, specs, max_loras=max_loras, max_lora_rank=max_lora_rank,
                host=host, port=port, extra=extra,
            )
            runbook = build_dynamic_lora_runbook(
                host=host, port=port,
                example_name=specs[0][0], example_path=str(specs[0][1]),
            )
            if _is_glm53_flash(base):
                note = f"{GLM53_FLASH_LORA_WARNING}\n\n{NOTES}\n\n{runbook}"
            else:
                note = f"{NOTES}\n\n{runbook}"
        else:
            text = build_sglang_multi_lora_command(
                base, specs, max_loras=max_loras, max_lora_rank=max_lora_rank,
                host=host, port=port, extra=extra,
            )
            runbook = build_sglang_dynamic_lora_runbook(
                base, host=host, port=port,
                example_name=specs[0][0], example_path=str(specs[0][1]),
            )
            note = runbook
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
