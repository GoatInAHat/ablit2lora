"""vLLM plugin registration against a stubbed vllm package (no vLLM needed)."""

from __future__ import annotations

import importlib
import sys
import types

import pytest

PLUGIN = "ablit2lora.vllm_plugins.glm5next_lora"


class _SupportsLoRA:
    pass


class _StockCausalLM:
    pass


class _StockConditional:
    pass


class _Registry:
    def __init__(self, fail_on: str | None = None):
        self.calls: list[tuple[str, str]] = []
        self.fail_on = fail_on

    def register_model(self, arch: str, path: str) -> None:
        if arch == self.fail_on:
            raise RuntimeError("boom")
        self.calls.append((arch, path))


def _module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    return mod


def _install_fake_vllm(monkeypatch, registry, *, conditional=True, causal=True):
    glm = {}
    if causal:
        glm["Glm5NextForCausalLM"] = _StockCausalLM
    if conditional:
        glm["Glm5NextForConditionalGeneration"] = _StockConditional
    mods = {
        "vllm": _module("vllm", ModelRegistry=registry),
        "vllm.model_executor": _module("vllm.model_executor"),
        "vllm.model_executor.models": _module("vllm.model_executor.models"),
        "vllm.model_executor.models.interfaces": _module(
            "vllm.model_executor.models.interfaces", SupportsLoRA=_SupportsLoRA
        ),
        "vllm.models": _module("vllm.models"),
        "vllm.models.glm5next": _module("vllm.models.glm5next", **glm),
    }
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)


@pytest.fixture()
def load_plugin(monkeypatch):
    def load():
        sys.modules.pop(PLUGIN, None)
        return importlib.import_module(PLUGIN)

    yield load
    sys.modules.pop(PLUGIN, None)


def _resolve(path: str):
    module, cls = path.split(":")
    return getattr(importlib.import_module(module), cls)


def test_registers_both_architectures_with_lazy_strings(monkeypatch, load_plugin):
    registry = _Registry()
    _install_fake_vllm(monkeypatch, registry)
    plugin = load_plugin()
    plugin.register()
    assert dict(registry.calls).keys() == {
        "Glm5NextForCausalLM",
        "Glm5NextForConditionalGeneration",
    }
    assert len(registry.calls) == 2
    for arch, path in registry.calls:
        assert isinstance(path, str) and ":" in path
        cls = _resolve(path)
        assert issubclass(cls, _SupportsLoRA)
        stock = _StockCausalLM if arch == "Glm5NextForCausalLM" else _StockConditional
        assert issubclass(cls, stock)
    paths = dict(registry.calls)
    assert paths["Glm5NextForCausalLM"] != paths["Glm5NextForConditionalGeneration"]


def test_causal_override_keeps_gate_up_packed_mapping(monkeypatch, load_plugin):
    _install_fake_vllm(monkeypatch, _Registry())
    plugin = load_plugin()
    assert plugin.Glm5NextForCausalLMWithLoRA.packed_modules_mapping == {
        "gate_up_proj": ["gate_proj", "up_proj"]
    }


def test_registers_only_the_architecture_the_build_has(monkeypatch, load_plugin):
    registry = _Registry()
    _install_fake_vllm(monkeypatch, registry, conditional=False)
    plugin = load_plugin()
    plugin.register()
    assert [arch for arch, _ in registry.calls] == ["Glm5NextForCausalLM"]


def test_register_never_raises_when_vllm_is_absent(monkeypatch, load_plugin):
    monkeypatch.setitem(sys.modules, "vllm", None)
    plugin = load_plugin()
    assert not hasattr(plugin, "Glm5NextForCausalLMWithLoRA")
    plugin.register()


def test_register_never_raises_when_a_registration_fails(monkeypatch, load_plugin):
    registry = _Registry(fail_on="Glm5NextForCausalLM")
    _install_fake_vllm(monkeypatch, registry)
    plugin = load_plugin()
    plugin.register()
    assert [arch for arch, _ in registry.calls] == ["Glm5NextForConditionalGeneration"]


def test_entry_point_targets_register():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    assert '[project.entry-points."vllm.general_plugins"]' in text
    assert f'ablit2lora_glm5next_lora = "{PLUGIN}:register"' in text
