# Changelog

Versions follow `pyproject.toml`. Dates are commit dates. Git tags exist for
v0.2.0, v0.3.0 and v0.3.1 only.

## 0.4.1 - 2026-10-16

### Fixed

- **convert on BF16 and FP8 pairs.** Storage rounding noise from both
  checkpoints is a full-rank floor on the diff (4-15% of the diff norm for
  BF16 and 50-150% for FP8 in synthetic pairs), so the fixed `--tol` flagged
  nearly every tensor. convert now estimates that floor from the stored
  values (`noise.py`: per-element grid spacing, FP8 pairs dequantized first
  and using the block-scaled grid, a Monte-Carlo ceiling on the noise
  singular values) and accepts a tensor at the number of singular values
  above that ceiling when the remainder is within the noise budget
  (`tol^2 * total + noise_margin * noise energy`). New statuses:
  `within_noise` (accepted) and `noise_only` (the diff is indistinguishable
  from rounding and is not carried). New `--noise-margin` (default 2.0);
  `--noise-margin 0` is the strict v0.4.0 `--tol` rule for 2-D tensors. A
  float32 tensor stored on the BF16 or FP16 grid is treated as that grid,
  and both sides use the coarser one. The console summary lists
  best-effort tensors separately and `manifest.json` records `noise_margin`.
- **1-D tensors** with an exactly zero diff (for example norms stored in
  different dtypes) are skipped as unchanged instead of reported
  `unrepresentable`. A 1-D diff within the noise budget is `noise_only`
  (needs `--noise-margin` above 0).
- **Tied `lm_head`** is recognised when it is stored on either side only
  (it was reported `missing` when only the base stored it), and values are
  compared in float32 rather than by sha256 across dtypes. The zero-diff 1-D
  and tied-`lm_head` fixes apply at `--noise-margin 0` too.
- **Embedding LoRA layout.** Edits to `embed_tokens` (and `wte`,
  `word_embeddings`, `embed_in`, `tok_embeddings`) are written as PEFT's
  `lora_embedding_A/B` (shapes `[r, num_embeddings]` and `[embedding_dim, r]`)
  instead of the Linear layout (`lora_A.weight`/`lora_B.weight`), which PEFT
  ignored for embeddings: it warned about missing adapter keys and the
  merged embedding lost the whole edit while convert reported `exact`.
  `merge_and_unload` and `bake` now reproduce the edited embedding; `read_adapter` reads both
  layouts. With `tie_word_embeddings: true` a runtime LoRA edits the input
  embedding only; convert records a warning in `manifest.json` (`warnings`).
- **vLLM plugin.** `ablit2lora.vllm_plugins.glm5next_lora` now registers both
  `Glm5NextForCausalLM` and `Glm5NextForConditionalGeneration` (the
  architecture the official GLM-5.3-Flash checkpoints declare). Still
  experimental and never run against a live vLLM; on vLLM main the
  ConditionalGeneration class already inherits LoRA support, and the MLA and
  indexer hooks that vllm-project/vllm#56327 adds cannot come from a plugin.
- **`quantize_blockwise`** rounds the block scale to its storage dtype before
  coding, so dequantization no longer carries a per-block gain error (0.37%
  in the regression test).
- **Docs.** The README, `convert --help`, the model card and the `serve`
  notes now say that the adapter carries the edit and not the checkpoints'
  rounding noise, state the FP8 `bake` fidelity truthfully (serve the adapter
  or `--output-dtype bfloat16` when that matters), and soften "cannot
  hot-mount" for GLM-5.3-Flash to "not validated".

### Added

- `--noise-margin` on `convert`.
- README section on llama.cpp (`convert_lora_to_gguf.py`, `llama-server`
  `--lora`) and Hugging Face Hub upload (`hf upload`). Taken from upstream
  docs and source; not run with an ablit2lora adapter.
- `tests/test_noise.py`, `tests/test_embedding.py`, `tests/test_vllm_plugin.py`.

### Changed

- Version 0.4.1. The `test_version` assertion follows it.

## 0.4.0 - 2026-10-16

Breaking for 0.3.x consumers: `serve` flags changed or were removed, `peft`
is a core dependency, and a `vllm.general_plugins` entry point was added.

- `serve` emits native multi-adapter commands: vLLM
  (`--enable-lora --lora-modules name=path ... --max-loras N
  --max-lora-rank R`, plus the dynamic-reload runbook) and SGLang
  (`--engine sglang`). `--adapter name=path` is repeatable; `--name` is gone.
  `--max-loras` and `--max-lora-rank` are validated against the supplied
  adapters.
- Removed the GLM-EXL3 shared-weight serving machinery and the flags that fed
  it (`--directions`, `--directions-manifest`, `--serving-profile`,
  `--serving-proof`, `--exl3-writer-proof`, `--plugin-path`, `--stock-name`),
  and the dead `build_lora_command`.
- `peft` moved from an optional extra to a core dependency. convert writes a
  Hub-ready `README.md` model card next to the adapter.
- Added an experimental, unverified vLLM plugin for GLM-5.3-Flash LoRA
  (`vllm.general_plugins` entry point), registered through vLLM's lazy
  `module:ClassName` string. Tracks vllm-project/vllm#56327.
- Fixed: `eval` read a hardcoded `max_lora_rank=8` instead of the adapter's
  rank; a tied `lm_head` under any prefix (not just two literal names) now
  resolves to `tied_unchanged` using the config's `tie_word_embeddings`;
  convert no longer imports `serve` to build its hint; the GLM-5.3-Flash
  `SupportsLoRA` warning is scoped to that base.
- README rewritten around the multi-adapter pitch; tests grew from 28 to 61
  (serve emitters, eval, MoE and differently named tensors, model card);
  GitHub Actions CI on Python 3.10 to 3.13.

## 0.3.1 - 2026-09-26

- Fixed the shared-weight serving recipes (removed in 0.4.0).
- `fit_rank` reduces residual energy in float64, since squaring large finite
  float32 singular values could overflow and turn rank selection into
  `inf/inf`; it also rejects non-finite deltas, `tol` outside (0, 1) and
  `max_rank < 1`.
- After the tag, `main` gained a source-proven EXL3 shared-weight serving
  profile (#1, 2026-10-09; version string unchanged). It was removed in 0.4.0.

## 0.3.0 - 2026-09-25

- Real-checkpoint readers. FP8 e4m3 block-scale dequant, quant and requant
  (`quant.py`, default 128x128 blocks, `weight_block_size` override from
  `config.json`). FP8-vs-FP8 pairs are dequantized before the float32 diff
  and the manifest records the dequant method and scale provenance; scale
  tensors are excluded. New statuses `precision_mismatch` and
  `quant_unsupported`. NVFP4 (modelopt) is rejected as out of scope.
- `bake` restored as the deployment path: an FP8 base is dequantized, fused
  and requantized with fresh block scales (unedited tensors keep their
  bytes); `--output-dtype bfloat16` writes a plain BF16 checkpoint.
- `serve` defaulted to a two-engine config (base plus baked checkpoint),
  with the hot-mount command behind `--enable-lora`.
- 27 tests.

## 0.2.0 - 2026-09-24

- ablit2lora became a pure converter: `extract`, `emit` and the direction
  finding code were removed. It converts published abliterated checkpoints
  and no longer finds refusal directions.
- New `convert`: lazy per-tensor diff of `--base` against `--abliterated`,
  exact SVD rank diagnostics (rank 1 if the residual is below `--tol`,
  escalating to `--max-rank`), retrained tensors flagged in `manifest.json`
  (`--include-flagged` for best-effort factors), PEFT adapter with
  `lora_alpha == r` and a zero-padded uniform rank, and a ready-to-run
  `vllm serve` command.
- `bake` reworked as an optional streaming fused export (`--fix-bias`
  dropped); `eval` compares base, base plus adapter and the abliterated
  reference.
- README positioning and credits to the abliteration labs.

## 0.1.0 - 2026-09-23

- First release: abliteration as a rank-1 LoRA. `extract` (contrastive
  mean-difference and PCA refusal directions), `emit` (exact PEFT LoRA, MoE
  aware, `lora_alpha == r`), `bake`, `serve` (vLLM base-plus-adapter command)
  and `eval` (refusal rate against perplexity). 19 tests.
