"""Layer-spec parsing and MoE-aware module targeting (pure string logic)."""

from __future__ import annotations

import pytest

from ablit2lora.names import (
    is_expert_module,
    layer_of,
    parse_layers,
    select_targets,
)


def test_parse_layers_forms():
    assert parse_layers("all", 4) == [0, 1, 2, 3]
    assert parse_layers("", 4) == [0, 1, 2, 3]
    assert parse_layers("even", 5) == [0, 2, 4]
    assert parse_layers("odd", 5) == [1, 3]
    assert parse_layers("0,3,1", 4) == [0, 1, 3]
    assert parse_layers("8-10", 12) == [8, 9, 10]
    assert parse_layers("8-12:2", 20) == [8, 10, 12]
    assert parse_layers("10-", 13) == [10, 11, 12]
    assert parse_layers("-2", 6) == [0, 1, 2]


def test_parse_layers_rejects_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        parse_layers("9", 4)


def test_layer_of():
    assert layer_of("model.layers.7.self_attn.o_proj") == 7
    assert layer_of("model.layers.12.mlp.experts.3.down_proj") == 12
    assert layer_of("model.embed_tokens") is None


GLM_MOE_WEIGHTS = [
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.experts.0.gate_proj.weight",
    "model.layers.0.experts.0.down_proj.weight",
    "model.layers.0.experts.7.down_proj.weight",
    "model.layers.0.shared_experts.down_proj.weight",
    "model.layers.1.mlp.down_proj.weight",
    "model.layers.2.self_attn.o_proj.weight",
    "lm_head.weight",
]


def test_select_targets_moe():
    targets = select_targets(GLM_MOE_WEIGHTS, num_layers=3, layers_spec="all",
                             modules_spec="o_proj,down_proj")
    assert targets == [
        "model.layers.0.experts.0.down_proj",
        "model.layers.0.experts.7.down_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.shared_experts.down_proj",
        "model.layers.1.mlp.down_proj",
        "model.layers.2.self_attn.o_proj",
    ]


def test_select_targets_layer_filter():
    targets = select_targets(GLM_MOE_WEIGHTS, num_layers=3, layers_spec="1",
                             modules_spec="o_proj,down_proj")
    assert targets == ["model.layers.1.mlp.down_proj"]


def test_expert_classification():
    assert is_expert_module("model.layers.0.experts.3.down_proj")
    assert not is_expert_module("model.layers.0.shared_experts.down_proj")
    assert not is_expert_module("model.layers.0.self_attn.o_proj")
