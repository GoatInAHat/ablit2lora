"""Shared fixtures: a tiny random-weight Llama + synthetic prompt files.

Everything is generated locally: no network, no model downloads, CPU only.
"""

from __future__ import annotations

import json
import random

import pytest
import torch

VOCAB_WORDS = [f"tok{i}" for i in range(120)]


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        LlamaConfig,
        LlamaForCausalLM,
        LlamaTokenizerFast,
    )

    tok = Tokenizer(
        models.WordLevel(vocab={w: i for i, w in enumerate(VOCAB_WORDS)},
                         unk_token="tok0")
    )
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = LlamaTokenizerFast(
        tokenizer_object=tok,
        bos_token="tok1",
        eos_token="tok2",
        pad_token="tok2",
        model_max_length=64,
    )
    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg)
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() > 1:
                p.normal_(0.0, 0.5)
            else:
                p.zero_()
    d = tmp_path_factory.mktemp("tiny-model")
    model.save_pretrained(str(d))
    fast.save_pretrained(str(d))
    return d


@pytest.fixture(scope="session")
def prompt_files(tmp_path_factory):
    rng = random.Random(0)
    d = tmp_path_factory.mktemp("prompts")

    def make(name: str, n: int = 24):
        items = []
        for _ in range(n):
            k = rng.randrange(3, 9)
            items.append({"text": " ".join(rng.choice(VOCAB_WORDS[3:]) for _ in range(k))})
        p = d / name
        p.write_text("\n".join(json.dumps(x) for x in items))
        return p

    return make("harmful.jsonl"), make("benign.jsonl")
