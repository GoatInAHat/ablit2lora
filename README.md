# ablit2lora

Convert a *published* abliterated checkpoint into a standard, MB-scale PEFT
LoRA adapter, and emit native multi-adapter vLLM/SGLang serving recipes so
it runs alongside the untouched base on one engine.

ablit2lora does **not** find refusal directions and does not compete with
the abliteration labs. They do the science -- contrastive direction search,
orthogonalization, retraining -- and publish full modified checkpoints.
This tool converts those artifacts into PEFT LoRA adapters:

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
abliteration copy*, keep base + adapter. Disk: one base + MBs, not one full
checkpoint per variant. convert and bake operate at the safetensors level,
never load the model, and are architecture-agnostic (dense or MoE).

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
   (BF16 pairs diff their stored weights; FP8 pairs are dequantized first --
   see "Precision & quantization" below):
   - rank-1 residual < tol -> exact r=1 LoRA factors (the classic
     abliteration case)
   - otherwise escalate rank until the residual < tol or --max-rank,
     recording the residual per tensor
   - unchanged tensors are skipped
   - genuinely-retrained tensors (residual never < tol) are **flagged in
     manifest.json, not silently approximated**; opt in with
     --include-flagged for best-effort factors
3. **Emit**: a PEFT-compatible adapter directory (factors zero-padded to a
   uniform rank, lora_alpha == r so the LoRA scale is exactly 1.0), a
   manifest.json (per-tensor sha refs, delta norm, chosen rank, residual),
   a Hub-ready README.md model card, and the ready-to-run vLLM serve
   command.

## Quickstart

    ablit2lora convert --base <hf-id-or-path> --abliterated <hf-id-or-path> \
      --out adapter-abliterated

    vllm serve <base> --enable-lora \
      --lora-modules abliterated=adapter-abliterated --max-lora-rank 8

Per request, the OpenAI `model` field picks the variant: `"model":
"abliterated"` for the converted behavior; omit it (or name the base) for
the untouched base. Verify fidelity against the published checkpoint with
`ablit2lora eval` before relying on the adapter.

GLM-5.3-Flash specifically cannot hot-mount LoRA on today's stock vLLM yet
(see "Supported base families" below); `bake` + the two-engine fallback is
the path meanwhile:

    ablit2lora bake --adapter adapter-abliterated --model <base> \
      --output abliterated-fused
    ablit2lora serve --base <base> --abliterated abliterated-fused --script serve.sh

## Multiple adapters, one engine

Convert as many variants of the same base as you like -- each is another
MB-scale adapter, not another full checkpoint -- and hot-mount all of them
at once:

    ablit2lora serve --base <base> --enable-lora \
      --adapter stock-equivalent=<adapter-a> --adapter abliterated=<adapter-b>

This also prints the dynamic-reload runbook
(`VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` plus `/v1/load_lora_adapter` and
`/v1/unload_lora_adapter`) so adapters can be added or swapped without
restarting the engine, and the identical recipe for SGLang (`--engine
sglang`, same PEFT adapter files, SGLang's own flags and endpoints) --
proving the adapter is a standard, engine-portable artifact, not a
vLLM-specific one.

Multiple LoRA adapters -- stock-equivalent, abliterated, and any others for
the same base -- share one loaded copy of the base model's weights and one
KV-cache pool on a single vLLM engine; you select the variant per request
via the OpenAI `model` field. A prompt prefix already served under a given
adapter (or the base model) is reused from cache by later requests to that
same adapter without re-prefilling; switching to a different adapter for
the same prompt text triggers one fresh prefill for that adapter before its
cache is warm too -- no single in-flight response changes adapters
mid-generation. LoRA adds a small, batched set of extra matmuls per decode
step; serving several different adapters concurrently costs about the same
as serving one, but the per-step overhead versus no adapter at all is
small, not zero.

## Supported base families

| Base family | convert / bake | Native LoRA serving |
|---|---|---|
| Llama / Qwen / GLM-4-MoE-style dense and MoE decoders | architecture-agnostic safetensors-level diff/fuse | works wherever the engine's model class implements `SupportsLoRA` (most mainstream dense and MoE decoders do, on both vLLM and SGLang) |
| **GLM-5.3-Flash** (arch `glm5_next`, hybrid KDA+MLA+MoE) | converter target today | pending [vllm-project/vllm#56327](https://github.com/vllm-project/vllm/pull/56327) ("[LoRA][Model] Add LoRA support for GLM-5.3-Flash") -- experimental registration plugin included (`src/ablit2lora/vllm_plugins/glm5next_lora.py`, UNVERIFIED, see its docstring); use `bake` meanwhile |

An experimental GLM-EXL3 activation-projection runtime lives separately in
the `glm53-ablation-plugin` repo and is out of scope for this general tool.

## Serving interfaces

| Interface | Static launch | Dynamic reload |
|---|---|---|
| vLLM | `vllm serve <base> --enable-lora --lora-modules name=path [name2=path2 ...] --max-loras N --max-lora-rank R` | `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` + `POST /v1/load_lora_adapter` / `/v1/unload_lora_adapter` |
| SGLang | `python -m sglang.launch_server --model-path <base> --enable-lora --lora-paths name=path ... --max-loras-per-batch N --max-lora-rank R` | `POST /load_lora_adapter` / `/unload_lora_adapter` (no `/v1` prefix) |
| bake (fallback) | fuse into a full checkpoint, then two independent engines: `ablit2lora serve --abliterated <fused>` | none -- restart to change variants |

`ablit2lora serve` emits all three: `--adapter name=path` (repeatable)
`--enable-lora` for vLLM or SGLang (`--engine sglang`), `--abliterated
<dir>` for the bake fallback.

## Precision & quantization

The per-tensor diff in convert runs in float32; compare checkpoints at
matching precision:

- **BF16/FP16 pair**: diffed as stored. LoRA on a BF16/FP16 base is fully
  supported by every serving engine here.
- **FP8 pair** (block-scale `float8_e4m3fn` + `<module>.weight_scale_inv`,
  GLM-5.3-Flash-style 128x128 blocks): both sides are dequantized
  automatically before the diff; manifest.json records the dequant method
  and per-tensor block-scale provenance. The FP8 grid re-rounds the edited
  blocks, so the measured delta carries ~2-3% quant noise -- use a looser
  `--tol` (e.g. 0.05) than a BF16 pair's; that residual is the honest
  measurement, not a failure. LoRA on an FP8 base is supported for major
  architectures on recent vLLM; upgrade vLLM or `bake` if yours rejects the
  adapter.
- **NVFP4/MXFP4 (ModelOpt) is out of scope for convert**: the FP4 grid
  re-quantizes every 16 elements and its scales are split across shards
  (per-16 `weight_scale` + per-tensor `weight_scale_2`), so a
  checkpoint-vs-checkpoint diff measures quantization-grid noise, not the
  published edit. ablit2lora rejects NVFP4 checkpoints with that
  explanation; convert the BF16 pair instead
  (e.g. `zai-org/GLM-5.3-Flash-BF16`). LoRA-over-NVFP4 is young on any
  serving engine here -- `bake` is the reliable fallback on any quantized
  base.

The adapter itself is precision-portable: compute it once from a
matching-precision pair, then serve that same adapter over any quantization
of the same base via `--enable-lora` (the LoRA factors stay high-precision
on top of the quantized base), or `bake` it directly into a quantized copy.
Adapter factors default to float32; `--adapter-dtype bfloat16` halves the
size but can miss small tolerances. `lora_alpha == r` always, so the
applied LoRA scale is exactly 1.0 and served weights equal base + the
measured delta, not an approximation of it.

## bake (full-checkpoint fallback)

Fuse W <- W + B @ A into a full checkpoint. Useful for quantized/edge
runtimes and for the two-engine fallback when the serving engine's model
class does not implement `SupportsLoRA`:

    ablit2lora bake --adapter adapter-abliterated --model <base> \
      --output abliterated-fused [--output-dtype bfloat16]

Streams tensors shard by shard (flat RAM, preserved shard layout +
configs). On an **FP8 block-scale base**:

- default (`--output-dtype auto`): edited tensors are dequantized, fused,
  and **requantized** with fresh block scales (absmax/448); untouched
  tensors keep their original bytes. The output stays a valid FP8
  checkpoint for the same serving stack; edited blocks carry FP8
  re-rounding noise.
- `--output-dtype bfloat16`: every tensor is dequantized to BF16, the delta
  is fused exactly, block scales are dropped, and config.json loses its
  `quantization_config`. The accuracy-preserving path, at ~2x disk.

On a BF16 base the fused weights are written back in the base dtype (exact
LoRA merge, like PEFT `merge_and_unload`). Steady-state disk keeps a single
serving copy; peak = base + fused during the pass -- verify, then delete
the source copy and adapter.

## eval

Side-by-side harness: refusal rate over a harmful-prompt set (greedy
generations, conservative keyword classifier -- **harness-grade, not a
safety evaluation**) and perplexity over benign text, for the plain base,
base+adapter, and optionally the abliterated reference checkpoint. vLLM
backend preferred; transformers fallback.

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

## Tested on

Developed and exercised live only on 2 NVIDIA DGX Spark nodes running
GLM-5.3-Flash. Core convert/bake logic is architecture-agnostic by
construction (a pure safetensors-level tensor diff/fuse that never loads
model code) and is covered by the CPU-only synthetic test suite across
dense and MoE checkpoint shapes, differently-named tensors, and FP8
block-scale pairs -- not by a live run on every family in "Supported base
families." The vLLM and SGLang serving commands are unit-tested for correct
flags but have not all been exercised against a live engine here (the
GLM-5.3-Flash LoRA plugin explicitly has not -- see above). Reports from
other base families and other engines welcome.

## Limitations

- **NVFP4 is out of scope for convert** (FP4 grid noise dominates a
  checkpoint-vs-checkpoint diff); convert the BF16 pair and bake instead.
- **FP8-pair diffs carry quant-grid noise** (~2-3% RMS): use a looser
  --tol (0.05) than a BF16 pair; flagged tensors in the manifest tell you
  when the edit does not clear the noise.
- **Embedding edits.** Abliteration recipes that orthogonalize the embedding
  matrix change `embed_tokens`; convert writes it in PEFT's
  `lora_embedding_A/B` layout (PEFT merge reproduces it). Engine support for
  LoRA on embeddings varies: vLLM applies it only for models that declare
  `embedding_modules`, so check your model class. With
  `tie_word_embeddings: true` a runtime LoRA edits the input embedding only,
  not the tied `lm_head` output path (convert records this under `warnings`
  in manifest.json); `bake` writes the shared weight and is exact.
- LoRA cannot edit biases, norms, or other non-2D tensors: changes there
  are reported as unrepresentable in manifest.json, never applied.
- Flagged (retrained) tensors are excluded by default: base+adapter
  reproduces the abliterated model only up to the flagged deltas listed in
  manifest.json. Use eval to judge the gap.
- Refusal classification in eval is a keyword heuristic: harness-grade,
  not a safety evaluation. Abliteration reduces measured refusal, not all
  safety behavior; treat as a research/safety-evaluation tool.

## Dev

    python -m venv .venv && .venv/bin/pip install -e '.[test]' ruff
    .venv/bin/ruff check src tests
    .venv/bin/python -m pytest -q

Tests are CPU-only with synthetic safetensors checkpoint pairs (exact
rank-1 edits, rank-3 edits, fully retrained tensors, changed 1-D tensors,
MoE expert tensors, differently-named tensors) plus a tiny random-weight
LlamaForCausalLM for the PEFT and perplexity end-to-end -- no downloads, no
vLLM/SGLang required. Key tests: an exact rank-1 edit converts to r=1 with
residual ~0; rank-3 edits escalate to the true rank; retrained tensors are
flagged, never approximated; a PEFT merge of the converted adapter
reproduces the abliterated weights; the emitted vLLM/SGLang multi-adapter
commands carry the right flags (names, --max-loras, --max-lora-rank read
from each adapter_config.json); and eval's perplexity matches an
independent cross-entropy computation. CI (`.github/workflows/ci.yml`)
runs ruff and pytest on push/PR against Python 3.10, 3.11, 3.12, and 3.13.

MIT license. Python >= 3.10.
