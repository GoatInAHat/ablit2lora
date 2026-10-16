"""Unit tests for eval.py: rank-reading fix, refusal classification, PPL math.

CPU-only. The end-to-end run() test uses the tiny_model_dir fixture (a real,
randomly-initialized LlamaForCausalLM) with backend="transformers" so it
never touches vllm or the network.
"""

from __future__ import annotations

import json
import math

import pytest
import torch
import torch.nn.functional as F

from ablit2lora import eval as eval_mod


def test_read_examples_accepts_strings_and_dicts(tmp_path):
    path = tmp_path / "examples.jsonl"
    path.write_text(
        '"plain string"\n'
        '{"text": "dict with text"}\n'
        '{"messages": [{"role": "user", "content": "hi"}]}\n'
        "\n"  # blank lines are skipped
    )
    examples = eval_mod.read_examples(str(path))
    assert examples == [
        {"text": "plain string"},
        {"text": "dict with text"},
        {"messages": [{"role": "user", "content": "hi"}]},
    ]


def test_read_examples_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n\n")
    with pytest.raises(ValueError, match="no examples"):
        eval_mod.read_examples(str(path))


@pytest.mark.parametrize(
    "text",
    [
        "I cannot help with that request.",
        "I'M SORRY, but I can't do that.",
        "As an AI, I must decline.",
        "I am unable to assist with this.",
    ],
)
def test_is_refusal_detects_known_markers_case_insensitively(text):
    assert eval_mod._is_refusal(text)


@pytest.mark.parametrize(
    "text",
    [
        "Sure, here's how you do that.",
        "The answer is 42.",
        "",
    ],
)
def test_is_refusal_negative_cases(text):
    assert not eval_mod._is_refusal(text)


def test_is_refusal_only_checks_the_first_400_characters():
    padding = "x" * 400
    assert eval_mod._is_refusal("I cannot help." + padding)  # marker up front
    assert not eval_mod._is_refusal(padding + "I cannot help.")  # marker past the cutoff


def test_lora_rank_reads_adapter_config_and_floors_at_8_like_build_lora_command(tmp_path):
    # the bug: eval.py used to hardcode max_lora_rank=8 for every adapter,
    # regardless of its real rank
    assert eval_mod._lora_rank(None) == 8

    missing_cfg = tmp_path / "no-config"
    missing_cfg.mkdir()
    assert eval_mod._lora_rank(str(missing_cfg)) == 8

    small_rank = tmp_path / "small-rank"
    small_rank.mkdir()
    (small_rank / "adapter_config.json").write_text(json.dumps({"r": 3}))
    assert eval_mod._lora_rank(str(small_rank)) == 8  # floored, like build_lora_command

    big_rank = tmp_path / "big-rank"
    big_rank.mkdir()
    (big_rank / "adapter_config.json").write_text(json.dumps({"r": 32}))
    assert eval_mod._lora_rank(str(big_rank)) == 32  # real rank: no flooring needed


def test_ppl_matches_an_independent_cross_entropy_computation(tiny_model_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tiny_model_dir))
    net = AutoModelForCausalLM.from_pretrained(str(tiny_model_dir)).eval()
    texts = ["tok3 tok4 tok5 tok6", "tok7 tok8 tok9"]

    got = eval_mod._ppl(net, tokenizer, texts, torch.device("cpu"))

    # Independent re-derivation (manual log_softmax + gather, not net(...).loss):
    # per-text next-token cross-entropy, aggregated as sum(nll)/sum(tokens).
    total_nll, total_tokens = 0.0, 0
    with torch.inference_mode():
        for t in texts:
            ids = tokenizer(t, return_tensors="pt").input_ids
            logits = net(ids).logits[0, :-1].float()
            targets = ids[0, 1:]
            logp = F.log_softmax(logits, dim=-1)
            nll = -logp.gather(1, targets.unsqueeze(1)).squeeze(1)
            total_nll += float(nll.sum())
            total_tokens += nll.numel()
    want = math.exp(total_nll / total_tokens)

    assert math.isfinite(got) and got > 0
    assert got == pytest.approx(want, rel=1e-5)


def test_ppl_raises_on_no_scorable_tokens(tiny_model_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tiny_model_dir))
    net = AutoModelForCausalLM.from_pretrained(str(tiny_model_dir)).eval()
    with pytest.raises(ValueError, match="no tokens"):
        # max_length=1 leaves every sequence with < 2 ids, so nothing is scored
        eval_mod._ppl(net, tokenizer, ["tok3 tok4 tok5"], torch.device("cpu"), max_length=1)


def test_run_end_to_end_transformers_backend(tmp_path, tiny_model_dir):
    harmful = tmp_path / "harmful.jsonl"
    harmful.write_text(
        "\n".join(
            json.dumps({"text": t})
            for t in ["tok3 tok4 tok5", "tok6 tok7 tok8", "tok3 tok6 tok9"]
        )
    )
    benign = tmp_path / "benign.jsonl"
    benign.write_text(json.dumps({"text": "tok3 tok4 tok5 tok6"}) + "\n")
    out = tmp_path / "eval.json"

    eval_mod.run(
        base=str(tiny_model_dir),
        harmful=str(harmful),
        ppl_file=str(benign),
        limit=10,
        out=str(out),
        backend="transformers",
        device="cpu",
    )

    result = json.loads(out.read_text())
    assert result["backend"] == "transformers"
    base_entry = result["base"]
    assert base_entry["n"] == 3
    # refusal_rate is exactly refused / n, the harness-level arithmetic eval.py
    # itself performs in _eval_config
    assert base_entry["refusal_rate"] == pytest.approx(base_entry["refused"] / 3)
    assert 0 <= base_entry["refused"] <= 3
    assert math.isfinite(base_entry["perplexity"]) and base_entry["perplexity"] > 0
    assert len(base_entry["sample_outputs"]) <= 5
