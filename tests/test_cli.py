"""CLI smoke tests + end-to-end: extract -> emit -> PEFT merge == orthogonal.

Runs a tiny random-weight LlamaForCausalLM on CPU; no downloads.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def _run(*args):
    return subprocess.run(
        [sys.executable, "-m", "ablit2lora", *args],
        capture_output=True,
        text=True,
        timeout=300,
    )


@pytest.mark.parametrize("cmd", ["extract", "emit", "bake", "serve", "eval"])
def test_help(cmd):
    r = _run(cmd, "--help")
    assert r.returncode == 0, r.stderr
    assert "usage" in r.stdout


def test_version():
    r = _run("--version")
    assert r.returncode == 0 and "0.1.0" in r.stdout


def test_end_to_end_tiny(tmp_path, tiny_model_dir, prompt_files):
    harmful, benign = prompt_files
    dirs = tmp_path / "dirs.safetensors"
    r = _run(
        "extract", "--model", str(tiny_model_dir),
        "--harmful", str(harmful), "--benign", str(benign),
        "--layers", "all", "--batch-size", "8",
        "--max-length", "64", "--out", str(dirs),
    )
    assert r.returncode == 0, r.stderr
    assert dirs.exists() and Path(str(dirs) + ".json").exists()

    adapter = tmp_path / "adapter"
    r = _run(
        "emit", "--direction", str(dirs), "--model", str(tiny_model_dir),
        "--layers", "0-2", "--modules", "o_proj,down_proj",
        "--alpha", "1.0", "--out", str(adapter),
    )
    assert r.returncode == 0, r.stderr
    assert (adapter / "adapter_model.safetensors").exists()
    assert (adapter / "adapter_config.json").exists()

    _verify_merge_matches_orthogonalization(tiny_model_dir, dirs, adapter)

    r = _run("serve", "--base", str(tiny_model_dir), "--adapter", str(adapter))
    assert r.returncode == 0, r.stderr
    assert "vllm serve" in r.stdout and "--enable-lora" in r.stdout


def _verify_merge_matches_orthogonalization(model_dir, dirs, adapter):
    peft = pytest.importorskip("peft")
    import torch
    from safetensors.torch import load_file
    from transformers import LlamaForCausalLM

    sd = load_file(str(dirs))
    base = LlamaForCausalLM.from_pretrained(str(model_dir))
    orig = {k: v.clone() for k, v in base.state_dict().items()}
    model = peft.PeftModel.from_pretrained(base, str(adapter))
    merged = model.merge_and_unload()
    for li in range(3):
        d = sd[f"layer.{li}"]
        o = merged.model.layers[li].self_attn.o_proj.weight
        wo = orig[f"model.layers.{li}.self_attn.o_proj.weight"]
        assert torch.allclose(o, wo - torch.outer(d, d @ wo), atol=1e-5), li
        dn = merged.model.layers[li].mlp.down_proj.weight
        wn = orig[f"model.layers.{li}.mlp.down_proj.weight"]
        assert torch.allclose(dn, wn - torch.outer(d, d @ wn), atol=1e-5), li
