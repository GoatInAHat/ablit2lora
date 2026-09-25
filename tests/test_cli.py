"""CLI smoke tests + end-to-end convert/serve/bake on synthetic checkpoints."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
import torch
from safetensors.torch import load_file

O1 = "model.layers.0.self_attn.o_proj.weight"
O2 = "model.layers.1.self_attn.o_proj.weight"
D2 = "model.layers.2.mlp.down_proj.weight"
D3 = "model.layers.3.mlp.down_proj.weight"


def _run(*args):
    return subprocess.run(
        [sys.executable, "-m", "ablit2lora", *args],
        capture_output=True,
        text=True,
        timeout=300,
    )


def _flat(s: str) -> str:
    """Collapse the serve command's backslash-newline continuations."""
    return " ".join(s.replace(chr(92) + "\n", " ").split())


@pytest.mark.parametrize("cmd", ["convert", "serve", "eval", "bake"])
def test_help(cmd):
    r = _run(cmd, "--help")
    assert r.returncode == 0, r.stderr
    assert "usage" in r.stdout


def test_version():
    r = _run("--version")
    assert r.returncode == 0 and "0.3.0" in r.stdout


def _convert_cli(tmp_path, pair, *extra):
    out = tmp_path / "adapter"
    r = _run(
        "convert",
        "--base", str(pair.base_dir),
        "--abliterated", str(pair.ablit_dir),
        "--out", str(out),
        *extra,
    )
    return r, out


def test_convert_end_to_end(tmp_path, mixed_pair):
    r, out = _convert_cli(tmp_path, mixed_pair)
    assert r.returncode == 0, r.stderr
    assert (out / "adapter_model.safetensors").exists()
    assert (out / "adapter_config.json").exists()
    assert (out / "manifest.json").exists()
    flat = _flat(r.stdout)
    assert "vllm serve" in flat and "--enable-lora" in flat
    cfg = json.loads((out / "adapter_config.json").read_text())
    assert cfg["r"] == cfg["lora_alpha"] == 3
    assert "model.layers.2.mlp.down_proj" not in cfg["target_modules"]
    assert len(cfg["target_modules"]) == 2


def test_serve_two_model_default(tmp_path, mixed_pair):
    _r, out = _convert_cli(tmp_path, mixed_pair)
    fused = tmp_path / "fused"
    r2 = _run("bake", "--adapter", str(out), "--model", str(mixed_pair.base_dir),
              "--output", str(fused))
    assert r2.returncode == 0, r2.stderr
    r = _run(
        "serve", "--base", str(mixed_pair.base_dir), "--abliterated", str(fused)
    )
    assert r.returncode == 0, r.stderr
    flat = _flat(r.stdout)
    assert flat.count("vllm serve") == 2
    assert "--port 8000" in flat and "--port 8001" in flat
    assert "--enable-lora" not in flat
    assert "SupportsLoRA" in flat  # the note explains why no hot-mount
    # script output carries both commands
    script = tmp_path / "serve.sh"
    r3 = _run("serve", "--base", str(mixed_pair.base_dir),
              "--abliterated", str(fused), "--script", str(script))
    assert r3.returncode == 0, r3.stderr
    text = _flat(script.read_text())
    assert text.count("vllm serve") == 2 and "--enable-lora" not in text


def test_serve_adapter_requires_enable_lora(tmp_path, mixed_pair):
    _r, out = _convert_cli(tmp_path, mixed_pair)
    r = _run("serve", "--base", str(mixed_pair.base_dir), "--adapter", str(out))
    assert r.returncode != 0
    assert "--enable-lora" in r.stderr


def test_serve_enable_lora_warns(tmp_path, mixed_pair):
    _r, out = _convert_cli(tmp_path, mixed_pair)
    r = _run(
        "serve", "--base", str(mixed_pair.base_dir),
        "--adapter", str(out), "--enable-lora",
    )
    assert r.returncode == 0, r.stderr
    flat = _flat(r.stdout)
    assert "vllm serve" in flat
    assert "--lora-modules abliterated=" in flat
    assert "--max-lora-rank 8" in flat
    assert "warning:" in flat and "SupportsLoRA" in flat


def test_bake_fuses_only_converted(tmp_path, mixed_pair):
    _r, out = _convert_cli(tmp_path, mixed_pair)
    fused = tmp_path / "fused"
    r = _run(
        "bake",
        "--adapter", str(out),
        "--model", str(mixed_pair.base_dir),
        "--output", str(fused),
    )
    assert r.returncode == 0, r.stderr
    W = load_file(str(fused / "model.safetensors"))
    b, a = mixed_pair.base, mixed_pair.ablit
    for name in (O1, O2):
        assert torch.allclose(W[name], a[name], atol=1e-4), name
    assert torch.equal(W[D2], b[D2])
    assert torch.equal(W[D3], b[D3])
    assert (fused / "config.json").exists()


def test_bake_output_dtype_flag(tmp_path, mixed_pair):
    _r, out = _convert_cli(tmp_path, mixed_pair)
    fused = tmp_path / "fused"
    r = _run(
        "bake", "--adapter", str(out), "--model", str(mixed_pair.base_dir),
        "--output", str(fused), "--output-dtype", "bfloat16",
    )
    assert r.returncode == 0, r.stderr
    W = load_file(str(fused / "model.safetensors"))
    b, a = mixed_pair.base, mixed_pair.ablit
    for name in (O1, O2):
        assert torch.allclose(W[name], a[name].to(torch.bfloat16), atol=1e-4), name
    assert torch.equal(W[D2], b[D2].to(torch.bfloat16))


def test_identical_checkpoints_no_adapter(tmp_path, mixed_pair):
    out = tmp_path / "ad"
    r = _run(
        "convert",
        "--base", str(mixed_pair.base_dir),
        "--abliterated", str(mixed_pair.base_dir),
        "--out", str(out),
    )
    assert r.returncode != 0
    assert (out / "manifest.json").exists()
