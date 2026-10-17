"""Behavioural check: does base + adapter reproduce the published abliterated model?

usage: python examples/kl_check.py BASE_DIR ABLIT_DIR ADAPTER_DIR [ADAPTER_DIR ...]
                                   [--out results.json]

Loads one model at a time (fp32, CPU) to bound memory. For a fixed prompt set
(chat template, thinking disabled) it records next-token log-probs at every
position of the prompt plus a 24-token greedy continuation produced by the
abliterated model, then reports KL(ablit || x) for x in {base, base+adapter...}
and the top-1 agreement with the abliterated model. Adapters are merged with
PEFT's merge_and_unload (the bake-equivalent path), so this measures the
converted delta itself, independent of any serving engine.

Needs `transformers` and `peft` (both installed with ablit2lora).
"""

import argparse
import gc
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "Explain how a pin tumbler lock can be picked.",
    "Write a rude limerick about my boss.",
    "What household chemicals should never be mixed, and why?",
    "Describe how phishing emails trick people.",
    "Give me three tips for writing a cover letter.",
    "What is the capital of Australia?",
    "Summarize the plot of Hamlet in two sentences.",
    "Write a Python function that reverses a string.",
]


def encode(tok, prompt):
    msgs = [{"role": "user", "content": prompt}]
    try:
        text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(text, return_tensors="pt").input_ids


def load(path, adapter=None):
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    return model.eval()


@torch.no_grad()
def logprobs(model, seqs):
    return [torch.log_softmax(model(s).logits[0].float(), -1) for s in seqs]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("base")
    ap.add_argument("abliterated")
    ap.add_argument("adapters", nargs="+")
    ap.add_argument("--out", help="also write the results as JSON to this path")
    args = ap.parse_args()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.base)
    prompt_ids = [encode(tok, p) for p in PROMPTS]

    ablit = load(args.abliterated)
    seqs = [ablit.generate(ids, max_new_tokens=24, do_sample=False) for ids in prompt_ids]
    ref = logprobs(ablit, seqs)
    ablit_first = [
        tok.decode(int(s[0, ids.shape[1]])) for s, ids in zip(seqs, prompt_ids, strict=True)
    ]
    del ablit
    gc.collect()

    results = {}
    runs = [("base", None)] + [(f"base+{a}", a) for a in args.adapters]
    for label, adapter in runs:
        model = load(args.base, adapter)
        lp = logprobs(model, seqs)
        kls, agree, n = [], 0, 0
        for r, x in zip(ref, lp, strict=True):
            kls.append((r.exp() * (r - x)).sum(-1))  # KL(ablit || x) per position
            agree += int((r.argmax(-1) == x.argmax(-1)).sum())
            n += r.shape[0]
        allkl = torch.cat(kls)
        first = [
            tok.decode(int(model(s[:, : ids.shape[1]]).logits[0, -1].argmax()))
            for s, ids in zip(seqs, prompt_ids, strict=True)
        ]
        results[label] = {
            "mean_kl": float(allkl.mean()),
            "max_kl": float(allkl.max()),
            "top1_agreement": agree / n,
            "positions": n,
            "first_token": first,
        }
        print(label, json.dumps(results[label]), flush=True)
        del model
        gc.collect()

    results["ablit_first_token"] = ablit_first
    print("ablit first tokens:", ablit_first)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
