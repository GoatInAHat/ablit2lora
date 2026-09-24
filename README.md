# ablit2lora

**Pure converter**: turn a *published* abliterated checkpoint into a
MB-scale LoRA adapter -- keep one base copy, hot-swap the edit in vLLM.

ablit2lora does **not** find refusal directions and does not compete with
the abliteration labs. They do the science -- contrastive direction search,
orthogonalization, retraining -- and publish full modified checkpoints.
This tool converts those artifacts into PEFT LoRA adapters so you never
keep a second full copy of the weights:

        published abliteration        ablit2lora convert        adapter (~MB)
        (full checkpoint,     ────►   diff + SVD rank    ────►  + manifest.json
         transient download)          diagnostics
                                                                  │
      base model (untouched)  ◄───────────────────────────────────┘
                │
                ▼
      vllm serve base --enable-lora --lora-modules abliterated=adapter
                          (no second copy)

**Transient workflow:** download the abliteration, convert it, *delete the
abliteration copy*, keep base + adapter. Disk: one base + MBs, not one
full checkpoint per variant.

First-class target: **GLM-5.3-Flash** (arch glm5_next, MoE) -- and any
Llama/GLM/Qwen-style decoder, dense or MoE. convert and bake operate at
the safetensors level, never load the model, and are architecture-agnostic.

## Credits

The checkpoints this tool consumes exist because of the abliteration labs,
building on the research lineage of Arditi et al., *Refusal in LLMs is
mediated by a single direction*:

- **audnai/penclaw** -- abliterated checkpoint releases
- **orcarouter** -- abliterated/decensored model family
- **dealignai** -- dealignment releases
- **huihui-ai** -- abliterated checkpoints across many architectures
- ...and everyone else publishing diffs of their work

Not affiliated with any of them. The science and the checkpoints are
theirs -- go read and star their repos. ablit2lora only re-expresses their
published edits as adapters.

## How convert works

    ablit2lora convert --base <hf-id-or-path> --abliterated <hf-id-or-path> \
      [--tol 1e-3] [--max-rank 8] [--out adapter]

1. **Stream both checkpoints lazily** (safetensors memory-mapping; two
   tensors in RAM at a time, never full copies) and match tensors by name.
2. **Per matched tensor**: delta = W_ablit - W_base, then one exact SVD:
   - rank-1 residual < tol -> exact r=1 LoRA factors (the classic
     abliteration case)
   - otherwise escalate rank until the residual < tol or --max-rank,
     recording the residual per tensor
   - unchanged tensors are skipped
   - genuinely-retrained tensors (residual never < tol) are **flagged in
     manifest.json, not silently approximated**; opt in with
     --include-flagged for best-effort factors
3. **Emit**: a PEFT-compatible adapter (factors zero-padded to a uniform
   rank, lora_alpha == r so the LoRA scale is exactly 1.0), a
   manifest.json (per-tensor sha refs, delta norm, chosen rank, residual),
   and the ready-to-run vLLM serve command.

**Precision guidance.** The per-tensor diff runs in float32 on top of each
side's storage dtype. Compare checkpoints at matching precision -- BF16
base vs BF16 abliteration, FP8 vs FP8: quantized-vs-quantized or
mixed-precision diffs carry quantization noise, which surfaces as flagged
tensors instead of a clean low-rank edit. Diff at the highest precision
both sides share (BF16 before FP8), and quantize after converting. Adapter
factors default to float32; bfloat16 halves the size but can miss small
tolerances.

## Quickstart (GLM-5.3-Flash)

    # transient: fetch the published abliteration at the same precision as the base
    ablit2lora convert \
      --base zai-org/GLM-5.3-Flash \
      --abliterated <lab>/GLM-5.3-Flash-abliterated \
      --tol 1e-3 --max-rank 8 \
      --out adapter-glm53flash-ablit
    # then delete the abliteration copy; serve base + adapter (one copy):
    ablit2lora serve --base zai-org/GLM-5.3-Flash \
      --adapter adapter-glm53flash-ablit --script serve.sh
    # verify fidelity: base+adapter should match the abliterated reference's
    # refusal behavior while perplexity stays near the plain base
    ablit2lora eval --base zai-org/GLM-5.3-Flash \
      --abliterated <lab>/GLM-5.3-Flash-abliterated \
      --adapter adapter-glm53flash-ablit \
      --harmful harmful.jsonl --ppl-file benign.jsonl

## eval

Side-by-side harness: refusal rate over a harmful-prompt set (greedy
generations, conservative keyword classifier -- harness-grade, not a
safety evaluation) and perplexity over benign text, for the plain base,
base+adapter, and optionally the abliterated reference checkpoint. vLLM
backend preferred; transformers fallback.

## bake (optional fused export)

For pipelines that cannot carry a LoRA at serve time (some quantized/edge
runtimes). Prefer serve; bake is the escape hatch:

    ablit2lora bake --adapter adapter-glm53flash-ablit --model <base> \
      --output glm53flash-ablit-fused

Streams tensors shard by shard (flat RAM, preserved shard layout +
configs). Steady-state disk keeps a single serving copy; peak = base +
fused during the pass -- verify, then delete the source copy and adapter.

## vLLM snippet + quantization notes

    vllm serve zai-org/GLM-5.3-Flash \
      --enable-lora \
      --lora-modules abliterated=/data/adapters/glm53flash-ablit \
      --max-lora-rank 8

Per request: "model": "abliterated" uses the converted behavior; omitting
it uses the untouched base.

- **BF16/FP16 base + LoRA**: fully supported.
- **FP8 (W8A8) base + LoRA**: supported for major architectures on recent
  vLLM; if startup rejects the adapter, upgrade vLLM or bake.
- **NVFP4/MXFP4 (ModelOpt) base + LoRA**: newer and architecture-limited;
  LoRA-over-NVFP4 is young -- check your vLLM release notes, and treat
  bake as the reliable fallback on any quantized base.
- Converted adapters use lora_alpha == r (LoRA scale exactly 1.0), so
  served weights equal base + the measured delta in the serving dtype.

## Disk savings

Adapter bytes = sum over converted modules of (in_dim + out_dim) x
dtype_size x rank.

| Artifact | Size (order of magnitude) |
|---|---|
| Full abliterated copy (30B-class MoE, bf16) | ~60 GB |
| Same, FP8 quant variant | ~30 GB (a second full copy per quant) |
| **ablit2lora adapter (r <= 8, fp32, changed modules only)** | **~10-100 MB** |
| manifest.json | KBs |

Illustrative for a GLM-5.3-Flash-class MoE; convert prints exact sizes for
your pair. Every additional variant is another MB-scale adapter against
the same base -- vs a full re-release per variant.

## Limitations

- LoRA cannot edit biases, norms, or other non-2D tensors: changes there
  are reported as unrepresentable in manifest.json, never applied.
- Flagged (retrained) tensors are excluded by default: base+adapter
  reproduces the abliterated model only up to the flagged deltas listed in
  manifest.json. Use eval to judge the gap.
- Refusal classification in eval is a keyword heuristic: harness-grade,
  not a safety evaluation. Abliteration reduces measured refusal, not all
  safety behavior; treat as a research/safety-evaluation tool.
- v0.1.0's direction finding (extract/emit) was removed by design: this
  tool is a pure converter and does not compete with the abliteration
  labs.

## Dev

    python -m venv .venv && .venv/bin/pip install -e '.[test]' ruff
    .venv/bin/ruff check src tests
    .venv/bin/python -m pytest -q

Tests are CPU-only with synthetic safetensors checkpoint pairs (exact
rank-1 edits, rank-3 edits, fully retrained tensors, changed 1-D tensors)
plus a tiny random-weight LlamaForCausalLM for the PEFT end-to-end -- no
downloads. Key tests: an exact rank-1 edit converts to r=1 with residual
~0; rank-3 edits escalate to the true rank; retrained tensors are flagged,
never approximated; and a PEFT merge of the converted adapter reproduces
the abliterated weights.

MIT license. Python >= 3.10.
