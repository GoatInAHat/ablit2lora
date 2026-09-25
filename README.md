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

**Deployment reality check (v0.3).** Hot-mounted LoRA requires the serving
engine's model class to implement `SupportsLoRA`. The vLLM `glm5_next`
class (GLM-5.3-Flash) does **not** — no `packed_modules_mapping`, no LoRA
plumbing — so on the stock engine the adapter cannot be hot-mounted. The
supported deployment path is therefore:

    convert (verify + compact artifact) ──► bake into a full checkpoint ──►
    serve as a SECOND PLAIN MODEL next to the base (same engine, no
    --enable-lora). 'ablit2lora serve' emits that two-model config by
    default; the old --enable-lora command stays behind a flag with a
    warning.

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
2. **Per matched tensor**: delta = W_ablit - W_base, then one exact SVD
   (BF16 pairs diff their stored weights; FP8 pairs are dequantized first,
   see below):
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

**Precision guidance.** The per-tensor diff runs in float32. Compare
checkpoints at matching precision:

- **BF16 pair**: weights are diffed as stored.
- **FP8 pair** (GLM-5.3-Flash-style `float8_e4m3fn` + block scales,
  `<module>.weight_scale_inv`, 128x128 blocks): both sides are dequantized
  automatically before the diff; manifest.json records the dequant method
  and per-tensor block-scale provenance. The FP8 grid re-rounds the edited
  blocks, so the measured delta carries ~2-3% quant noise -- use a looser
  `--tol` (e.g. 0.05) than a BF16 pair's; that residual is the honest
  measurement, not a failure.
- **NVFP4 (modelopt) is out of scope for convert**: the FP4 grid
  re-quantizes every 16 elements and its scales are split across shards
  (per-16 `weight_scale` + per-tensor `weight_scale_2`), so a
  checkpoint-vs-checkpoint diff measures quantization-grid noise, not the
  published edit. ablit2lora rejects NVFP4 checkpoints with that
  explanation; convert the BF16 pair
  (`zai-org/GLM-5.3-Flash-BF16`) instead.

Adapter factors default to float32; bfloat16 halves the size but can miss
small tolerances. The adapter is precision-portable: compute it once, then
bake it onto any quantization of the same base.

The adapter is precision-portable: compute it once from a matching-precision
pair (e.g. FP8 abliteration vs FP8 base), then serve the same adapter over
any quantization of the same base -- FP8, NVFP4, ... -- via --enable-lora;
the LoRA factors stay high-precision on top of the quantized base. If your
vLLM build rejects LoRA on a given quant (LoRA-over-NVFP4 is young), fall
back to bake.

## Quickstart (GLM-5.3-Flash)

    # transient: fetch the published abliteration at the same precision as the base
    ablit2lora convert \
      --base zai-org/GLM-5.3-Flash \
      --abliterated <lab>/GLM-5.3-Flash-abliterated \
      --tol 0.05 --max-rank 8 \
      --out adapter-glm53flash-ablit
    # glm5_next cannot hot-mount LoRA: bake the edit into a full checkpoint
    ablit2lora bake --adapter adapter-glm53flash-ablit \
      --model zai-org/GLM-5.3-Flash \
      --output glm53flash-ablit-fused
    # serve base + baked abliterated as two plain models (default):
    ablit2lora serve --base zai-org/GLM-5.3-Flash \
      --abliterated glm53flash-ablit-fused --script serve.sh
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

## bake (the deployment path)

Fuse W <- W + B @ A into a full checkpoint. This is the supported path for
engines that cannot hot-mount LoRA (glm5_next lacks SupportsLoRA) and for
quantized/edge runtimes:

    ablit2lora bake --adapter adapter-glm53flash-ablit --model <base> \
      --output glm53flash-ablit-fused [--output-dtype bfloat16]

Streams tensors shard by shard (flat RAM, preserved shard layout +
configs). Precision handling on an **FP8 block-scale base**:

- default (`--output-dtype auto`): edited tensors are dequantized, fused,
  and **requantized** with fresh block scales (absmax/448); untouched
  tensors keep their original bytes. The output stays a valid FP8
  checkpoint for the same serving stack; edited blocks carry FP8
  re-rounding noise.
- `--output-dtype bfloat16`: every tensor is dequantized to BF16, the
  delta is fused exactly, block scales are dropped, and config.json loses
  its `quantization_config`. The accuracy-preserving path at ~2x disk.

On a BF16 base the fused weights are written back in the base dtype
(exact LoRA merge, like PEFT merge_and_unload). Steady-state disk keeps a
single serving copy; peak = base + fused during the pass -- verify, then
delete the source copy and adapter.

## Serving

Default ('ablit2lora serve --base <base> --abliterated <baked-model-dir>')
emits two plain `vllm serve` commands -- base on port P, the baked
abliterated checkpoint on P+1, same engine, no `--enable-lora`. Pick the
behavior per request by pointing the client at the model/port.

The legacy hot-mount command stays behind a flag with a warning (glm5_next
does not implement SupportsLoRA; kept for engines that do):

    ablit2lora serve --base <base> --adapter <adapter-dir> --enable-lora

## vLLM quantization notes (adapter path, architectures that support LoRA)

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

- **glm5_next cannot hot-mount LoRA** (no SupportsLoRA in the stock vLLM
  engine class): use bake + the two-model serve config on GLM-5.3-Flash.
- **NVFP4 is out of scope for convert** (FP4 grid noise dominates a
  checkpoint-vs-checkpoint diff); convert the BF16 pair and bake instead.
- **FP8-pair diffs carry quant-grid noise** (~2-3% RMS): use a looser
  --tol (0.05) than a BF16 pair; flagged tensors in the manifest tell you
  when the edit does not clear the noise.
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
