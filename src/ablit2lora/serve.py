"""Emit vLLM launch commands for abliterated-model serving.

The preferred mode is a compatible ``vllm.general_plugins`` runtime that
projects selected writer outputs per request. It exposes a stock name and an
abliterated alias from one engine and one loaded base checkpoint. This module
does not implement or silently install that runtime plugin; it emits the exact
environment/served-name contract for an already installed compatible plugin.

Two fallbacks remain available:

* ``--enable-lora`` hot-mounts a PEFT adapter on architectures that implement
  vLLM ``SupportsLoRA``.
* ``--abliterated`` launches two independent engines for a baked checkpoint.
  This needs two model allocations and is not a shared-weight deployment.
"""
from __future__ import annotations

import json
import re
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

SHARED_WEIGHT_NOTE = (
    "shared-weight runtime deployment\n"
    "- One vLLM engine loads the base once and advertises the stock name plus "
    "the abliterated alias.\n"
    "- A separately installed compatible vllm.general_plugins runtime must "
    "recognize ABL_SERVING_PROFILE, ABL_SERVING_PROOF, ABL_DIRS, "
    "ABL_DIRS_MANIFEST, and ABL_ALIAS and apply the request-local "
    "writer-output projection.\n"
    "- Direction artifacts are runtime-plugin inputs, not PEFT adapters. "
    "Their provenance, numerical fit, and live behavioral acceptance remain "
    "the runtime/deployment operator's responsibility.\n"
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
    "command will not hot-mount the adapter on the stock engine. Use a "
    "compatible shared-weight runtime plugin, or the two-engine baked "
    "fallback when resources permit."
)


def _command(words: list[str], extra: str | None = None) -> str:
    """Render one shell command; parse ``extra`` as shell words first."""
    if extra:
        words.extend(shlex.split(extra))
    separator = " " + chr(92) + "\n  "
    return separator.join(shlex.quote(word) for word in words)


def _export(name: str, value: str) -> str:
    return f"export {name}={shlex.quote(value)}"


def build_shared_weight_command(
    base: str,
    directions: str,
    *,
    directions_manifest: str | None = None,
    serving_profile: str = "glm53-experimental-approx-output-v1",
    serving_proof: str | None = None,
    plugin_path: str | None = None,
    stock_name: str = "glm-5.3-flash",
    name: str = "glm-5.3-flash-abliterated",
    host: str = "0.0.0.0",
    port: int = 8000,
    extra: str | None = None,
) -> str:
    """One-engine launch contract for a compatible writer-projection plugin."""
    if not stock_name or not name:
        raise ValueError("--stock-name and --name must be non-empty")
    if stock_name == name:
        raise ValueError("--stock-name and --name must differ")
    if not serving_profile:
        raise ValueError("--serving-profile must be non-empty")
    if serving_profile == "glm53-experimental-approx-output-v1" and not re.fullmatch(
        r"glm-5\.3-flash-abliterated(?:-[a-z0-9-]{1,32})?", name
    ):
        raise ValueError(
            "the GLM-5.3 serving profile requires --name glm-5.3-flash-abliterated "
            "or that name followed by a lowercase alphanumeric/hyphen suffix"
        )
    if not serving_proof:
        raise ValueError("--serving-proof is required in shared-weight mode")

    directions_path = Path(directions).expanduser().resolve()
    if not directions_path.is_file():
        raise FileNotFoundError(f"direction artifact not found: {directions_path}")
    manifest_path = (
        Path(directions_manifest).expanduser().resolve()
        if directions_manifest
        else directions_path.with_suffix(".manifest.json")
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"direction manifest not found: {manifest_path}; pass --directions-manifest"
        )
    proof_path = Path(serving_proof).expanduser().resolve()
    if not proof_path.is_file():
        raise FileNotFoundError(f"serving proof not found: {proof_path}")

    exports: list[str] = []
    if plugin_path:
        resolved_plugin = Path(plugin_path).expanduser().resolve()
        if not resolved_plugin.is_dir():
            raise NotADirectoryError(f"runtime plugin path is not a directory: {resolved_plugin}")
        # Preserve an existing PYTHONPATH without expanding the generator's
        # environment into the emitted command.
        exports.append(
            f"export PYTHONPATH={shlex.quote(str(resolved_plugin))}"
            "${PYTHONPATH:+:$PYTHONPATH}"
        )
    exports.extend(
        (
            _export("ABL_SERVING_PROFILE", serving_profile),
            _export("ABL_SERVING_PROOF", str(proof_path)),
            _export("ABL_DIRS", str(directions_path)),
            _export("ABL_DIRS_MANIFEST", str(manifest_path)),
            _export("ABL_ALIAS", name),
        )
    )
    command = _command(
        [
            "vllm",
            "serve",
            str(base),
            "--served-model-name",
            stock_name,
            name,
            "--host",
            str(host),
            "--port",
            str(port),
        ],
        extra,
    )
    return "\n".join((*exports, "", command))


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
    directions: str | None = None,
    directions_manifest: str | None = None,
    serving_profile: str = "glm53-experimental-approx-output-v1",
    serving_proof: str | None = None,
    plugin_path: str | None = None,
    stock_name: str | None = None,
    abliterated: str | None = None,
    adapter: str | None = None,
    enable_lora: bool = False,
    name: str | None = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    extra: str | None = None,
    script: str | None = None,
) -> None:
    """Emit one of the three explicit serving modes."""
    selected = sum(value is not None for value in (directions, abliterated, adapter))
    if selected != 1:
        raise ValueError("choose exactly one of --directions, --abliterated, or --adapter")
    if enable_lora and adapter is None:
        raise ValueError("--enable-lora requires --adapter")
    if adapter is not None and not enable_lora:
        raise ValueError("--adapter requires --enable-lora")
    if directions is None and (directions_manifest is not None or plugin_path is not None):
        raise ValueError("--directions-manifest and --plugin-path require --directions")
    if directions is None and serving_proof is not None:
        raise ValueError("--serving-proof requires --directions")

    if directions is not None:
        text = build_shared_weight_command(
            base,
            directions,
            directions_manifest=directions_manifest,
            serving_profile=serving_profile,
            serving_proof=serving_proof,
            plugin_path=plugin_path,
            stock_name=stock_name or "glm-5.3-flash",
            name=name or "glm-5.3-flash-abliterated",
            host=host,
            port=port,
            extra=extra,
        )
        note = SHARED_WEIGHT_NOTE
    elif adapter is not None:
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
