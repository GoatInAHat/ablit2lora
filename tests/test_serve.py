"""Unit tests for the native multi-adapter vLLM/SGLang emitters in serve.py."""

from __future__ import annotations

import json

import pytest

from ablit2lora import serve


def _flat(s: str) -> str:
    """Collapse the serve command's backslash-newline continuations."""
    return " ".join(s.replace(chr(92) + "\n", " ").split())


def _adapter(tmp_path, name: str, rank: int):
    """A minimal adapter dir: just adapter_config.json (all these emitters read)."""
    d = tmp_path / name
    d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps({"r": rank}))
    return d


def test_build_multi_lora_command_defaults_from_adapter_configs(tmp_path):
    a = _adapter(tmp_path, "a", rank=4)
    b = _adapter(tmp_path, "b", rank=16)

    cmd = _flat(
        serve.build_multi_lora_command(
            "base-model", [f"stock-equivalent={a}", f"abliterated={b}"]
        )
    )

    assert "vllm serve base-model" in cmd
    assert "--enable-lora" in cmd
    assert f"--lora-modules stock-equivalent={a} abliterated={b}" in cmd
    # N defaults to the adapter count, R to the largest r across configs
    assert "--max-loras 2" in cmd
    assert "--max-lora-rank 16" in cmd
    assert "--host 0.0.0.0" in cmd and "--port 8000" in cmd


def test_build_multi_lora_command_overrides_and_rank_floor(tmp_path):
    a = _adapter(tmp_path, "a", rank=2)  # below the floor of 8

    cmd = _flat(
        serve.build_multi_lora_command(
            "base-model", [f"only={a}"], max_loras=4, max_lora_rank=32
        )
    )
    assert "--max-loras 4" in cmd  # explicit override wins over the adapter count
    assert "--max-lora-rank 32" in cmd  # explicit override wins over the auto rank

    cmd_no_override = _flat(serve.build_multi_lora_command("base-model", [f"only={a}"]))
    assert "--max-lora-rank 8" in cmd_no_override  # floored like build_lora_command


def test_build_multi_lora_command_rejects_duplicate_names_and_bad_specs(tmp_path):
    a = _adapter(tmp_path, "a", rank=4)
    with pytest.raises(ValueError, match="duplicate adapter names"):
        serve.build_multi_lora_command("base", [f"dup={a}", f"dup={a}"])
    with pytest.raises(ValueError, match="must be name=path"):
        serve.build_multi_lora_command("base", [str(a)])
    with pytest.raises(ValueError, match="at least one --adapter"):
        serve.build_multi_lora_command("base", [])


def test_build_dynamic_lora_runbook_vllm():
    note = serve.build_dynamic_lora_runbook(
        host="0.0.0.0", port=8000, example_name="abliterated", example_path="/a/b"
    )
    assert "VLLM_ALLOW_RUNTIME_LORA_UPDATING=True" in note
    assert "POST http://0.0.0.0:8000/v1/load_lora_adapter" in _flat(note)
    assert "POST http://0.0.0.0:8000/v1/unload_lora_adapter" in _flat(note)
    assert '"lora_name": "abliterated"' in note
    assert '"lora_path": "/a/b"' in note
    assert "http://0.0.0.0:8000/v1/completions" in _flat(note)
    assert '"model": "abliterated"' in note


def test_build_sglang_multi_lora_command(tmp_path):
    a = _adapter(tmp_path, "a", rank=8)
    b = _adapter(tmp_path, "b", rank=32)

    cmd = _flat(
        serve.build_sglang_multi_lora_command("base-model", [f"x={a}", f"y={b}"])
    )
    assert "python -m sglang.launch_server" in cmd
    assert "--model-path base-model" in cmd
    assert "--enable-lora" in cmd
    assert f"--lora-paths x={a} y={b}" in cmd
    assert "--max-loras-per-batch 2" in cmd
    assert "--max-lora-rank 32" in cmd


def test_build_sglang_dynamic_lora_runbook_has_no_v1_prefix_and_colon_model_syntax():
    note = serve.build_sglang_dynamic_lora_runbook(
        "base-model", host="0.0.0.0", port=30000,
        example_name="abliterated", example_path="/a/b",
    )
    flat = _flat(note)
    assert "POST http://0.0.0.0:30000/load_lora_adapter" in flat
    assert "/v1/load_lora_adapter" not in flat
    assert "POST http://0.0.0.0:30000/unload_lora_adapter" in flat
    assert '"lora_name": "abliterated"' in note
    assert '"model": "base-model:abliterated"' in note


def test_run_vllm_multi_adapter_dispatch(tmp_path, capsys):
    a = _adapter(tmp_path, "a", rank=4)
    b = _adapter(tmp_path, "b", rank=16)

    serve.run(
        "base-model",
        adapter=[f"stock-equivalent={a}", f"abliterated={b}"],
        enable_lora=True,
    )
    out = _flat(capsys.readouterr().out)
    assert "vllm serve base-model" in out
    assert "--max-loras 2" in out and "--max-lora-rank 16" in out
    assert "VLLM_ALLOW_RUNTIME_LORA_UPDATING" in out
    assert "warning:" in out and "SupportsLoRA" in out  # LORA_WARNING is kept


def test_run_sglang_multi_adapter_dispatch(tmp_path, capsys):
    a = _adapter(tmp_path, "a", rank=4)
    b = _adapter(tmp_path, "b", rank=16)

    serve.run(
        "base-model",
        adapter=[f"stock-equivalent={a}", f"abliterated={b}"],
        enable_lora=True,
        engine="sglang",
    )
    out = _flat(capsys.readouterr().out)
    assert "sglang.launch_server" in out
    assert "--max-loras-per-batch 2" in out and "--max-lora-rank 16" in out
    assert "/load_lora_adapter" in out
    assert "warning:" not in out  # the vLLM/glm5_next-specific warning is vLLM-only


def test_run_rejects_unknown_engine(tmp_path):
    a = _adapter(tmp_path, "a", rank=4)
    with pytest.raises(ValueError, match="unknown --engine"):
        serve.run("base-model", adapter=[f"a={a}"], enable_lora=True, engine="tgi")


def test_run_still_requires_exactly_one_mode(tmp_path):
    a = _adapter(tmp_path, "a", rank=4)
    with pytest.raises(ValueError, match="choose exactly one"):
        serve.run("base-model", adapter=[f"a={a}"], abliterated="x", enable_lora=True)
    with pytest.raises(ValueError, match="choose exactly one"):
        serve.run("base-model")
