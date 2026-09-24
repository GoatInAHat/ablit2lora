# ablit2lora

Abliteration without a second model copy. Refusal-direction
orthogonalization is **exactly rank 1**, so it ships as a standard PEFT LoRA
adapter: the base weights stay untouched on disk, and vLLM serves base +
adapter with `--enable-lora` — hot-swappable per request, one copy,
~MBs instead of ~GBs.

First-class target: **GLM-5.3-Flash** (arch `glm5_next`, MoE) — and any
Llama/GLM/Qwen-style decoder, dense or MoE.

## The math

Abliteration (in the style of Arditi et al., *Refusal in LLMs is mediated by
a single direction*) projects a unit refusal direction `d` out of the
residual stream at chosen module boundaries. For a linear module `y = W x`:

    W' = W - d d^T W          (output-side orthogonalization)

The removed term is an outer product, so the edit is exactly rank 1:

    W - d d^T W  ==  W + B @ A,   A = d^T W,  B = -d

PEFT applies `W + (lora_alpha / r) * B @ A`. With `r = 1` and
`lora_alpha = 1` the scale is exactly 1.0 and the adapter is the
orthogonalization — no training, no approximation. `--alpha 1.0` is the exact
projection; larger values over-abliterate.

`--side both` emits the exact double projection
`W' = P_out W P_in` as a **rank-3** adapter (each removed term plus the cross
term is rank <= 1):

    A = [d_o^T W ; d_in^T ; d_in^T],   B = [-d_o , -W d_in , (d_o^T W d_in) d_o]

### Extract

`ablit2lora extract` hooks every selected decoder layer, runs harmful vs
benign prompt sets, and saves the normalized mean-difference direction of the
residual stream at each layer boundary (plus a contrast score per layer and
an optional PCA estimate). Direction files are one small safetensors.

### Emit

`ablit2lora emit` reads the base weights lazily (one tensor at a time via
safetensors memory mapping — never the whole model in RAM) and writes the
adapter: for each target module `A = d^T W`, `B = -alpha * d` with PEFT
scaling 1. Defaults target every `o_proj` and `down_proj` in the selected
layers — the modules that *write into* the residual stream, which is where a
residual-space direction lives.

## MoE models (GLM-5.3-Flash / glm5_next)

Target selection is per fully-qualified module path, so MoE works unchanged:

every per-expert module (`model.layers.N.experts.k.down_proj`) and every
expert-shared module (`...shared_experts.down_proj`) receives its own exact
rank-1 edit keyed to its layer's direction. Expert-shared modules are edited
**once**, not per expert. Per-expert *directions* (rather than the shared
residual direction) would need expert-routed attribution and are intentionally
out of scope; the shared direction is the standard practice.

For `glm5_next` and similar MoE archs, extraction requires a transformers
build that can load the model (possibly `--trust-remote-code` or a newer
version). `emit` and `bake` are safetensors-level and architecture-agnostic —
they work on any shard layout.

## Quickstart

    # 1. directions from contrasting activations (GPU box with the weights)
    ablit2lora extract --model zai-org/GLM-5.3-Flash \\
      --harmful harmful.jsonl --benign benign.jsonl --chat \\
      --out glm53flash-dirs.safetensors

    # 2. exact rank-1 adapter (~MBs)
    ablit2lora emit --direction glm53flash-dirs.safetensors \\
      --model zai-org/GLM-5.3-Flash --layers 8-40:2 \\
      --shared-direction 24 --out adapter-glm53flash-ablit

    # 3. serve base + adapter (one copy on disk)
    ablit2lora serve --base zai-org/GLM-5.3-Flash \\
      --adapter adapter-glm53flash-ablit --script serve.sh

    # 4. sanity: refusal rate vs base, perplexity drift
    ablit2lora eval --base zai-org/GLM-5.3-Flash \\
      --adapter adapter-glm53flash-ablit --harmful harmful.jsonl \\
      --ppl-file benign.jsonl

Layer specs: `all | even | odd | 0,3,7 | 8-24 | 8-24:2 | 10-` (open end).
`--shared-direction N` applies layer N's direction everywhere (classic
single-direction abliteration); the default uses each layer's own direction.

## Disk savings

Adapter bytes = sum over targets of (in_dim + out_dim) * dtype_size * r.

| Artifact | Size (order of magnitude) |
|---|---|
| Full abliterated copy (e.g. 30B-class MoE, bf16) | ~60 GB |
| Same, FP8 quant variant | ~30 GB (a second full copy per quant) |
| **ablit2lora adapter (r=1, bf16, all o_proj+down_proj)** | **~10-50 MB** |
| Direction file | ~KB-MB |

Illustrative for a GLM-5.3-Flash-class MoE; `emit` prints the exact sizes for
your model. Every additional variant (different alpha, layers, dtype) is
another MB-scale adapter against the same base — vs a full re-release per
variant.

## vs full-copy abliterated releases

Community releases (audnai/penclaw, orcarouter, dealainai-style repos)
distribute complete modified model snapshots:

| | Full-copy releases | ablit2lora |
|---|---|---|
| Disk | one full copy per variant (plus per-quant re-releases) | one base + MB-scale adapters |
| Hot swap | no — swap model dirs, restart | per-request LoRA name, no restart |
| Variants (alpha/layers) | re-run full export each time | re-emit adapter in seconds |
| Provenance | opaque diff vs base | base untouched; edit is auditable algebra |

(Not affiliated with those projects; sizes/claims reflect their published
artifact layout, not benchmarks we ran.)

## vLLM snippet

```bash
vllm serve zai-org/GLM-5.3-Flash \\
  --enable-lora \\
  --lora-modules abliterated=/data/adapters/glm53flash-ablit \\
  --max-lora-rank 8
```

Then per request: `"model": "abliterated"` uses the orthogonalized behavior;
omitting it uses the untouched base. `ablit2lora serve` prints this command
plus compatibility notes:

- **BF16/FP16 base + LoRA**: fully supported.
- **FP8 (W8A8) base + LoRA**: supported for major architectures on recent
  vLLM; if startup rejects the adapter, upgrade vLLM or `bake`.
- **NVFP4/MXFP4 (ModelOpt) base + LoRA**: newer and architecture-limited;
  the reliable fallback on any quantized base is `bake`.
- Adapter scale is exactly 1.0 (`lora_alpha == r`), so served weights equal
  the algebraic orthogonalization in the serving dtype.

## bake (optional fused pass)

`ablit2lora bake --adapter adapter-glm53flash-ablit --model <base> \\
  --output glm53flash-ablit-bf16 [--fix-bias]`

Streams tensors shard-by-shard (flat RAM, preserved shard layout + configs)
so steady-state disk still holds only one serving copy; peak = base + baked
during the pass. `--fix-bias` also orthogonalizes biases (the one term LoRA
cannot express; usually negligible, reported by `emit`).

## Limitations

- Bias terms keep their `d`-component in LoRA mode (reported; `bake --fix-bias`
  removes them exactly).
- Direction quality depends on the prompt sets; check `emit`'s delta norms
  and the `eval` refusal/PPL harness before trusting an adapter.
- Abliteration reduces measured refusal, not all safety behavior; treat as a
  research/safety-evaluation tool, not a guarantee.

## Dev

    python -m venv .venv && .venv/bin/pip install -e '.[test]' ruff
    .venv/bin/ruff check src tests
    .venv/bin/python -m pytest -q

Tests are CPU-only with a tiny random-weight LlamaForCausalLM — no downloads.
The key test proves: emitted LoRA applied to base **==** directly
orthogonalized weights (fp32, max |diff| < 1e-5), end-to-end through PEFT.

MIT license. Python >= 3.10.
