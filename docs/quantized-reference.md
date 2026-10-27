# Independently quantized reference audit (offline)

This is an opt-in measurement path, **not EXL3 conversion or runtime adapter
support**. `convert` still compares source checkpoints with its existing noise
model. It does not implement the reference below. Never discard quantization
residuals as noise when the target is the actual quantized checkpoint.

## Target and provenance

Let `S` be the canonical logical operator decoded from the actual live stock
pack, `Wabl` the publisher's stored checkpoint (including its rounding), and
`T = decode(Q_theta(Wabl))` an independently quantized reference under the
recovered current recipe. Fit **Delta_eff = T - S**. A fitted adapter's real
arithmetic candidate is `S + scale*B*A`, with an additional implementation
error from factor storage, GEMMs, activation/output casts and reductions.

`Q_theta(Wstock)` must first reproduce `S`. Matching a nominal bpw or EXL3
version is insufficient. Freeze/recover the quantizer implementation and
architecture port, calibration token IDs/order and capture precision, allocation
rules, Hessian/LDLQ/trellis parameters, codebook/scales/rotations, RNG states,
tie-break behavior, dtype handling, quantization traversal, and packing rules.
Keep format/recipe controls the same, but recompute checkpoint-dependent
activations/Hessians if the original algorithm does so. Freezing those to stock
changes the target. Downstream tensors can therefore differ even when their
source weights were not edited. Audit **all** logical operators, dense side
weights and buffers, not just the intervention's writers. A writer-only audit
cannot qualify a complete adapter.

An ideal directional operator cannot recreate a publisher's particular
stochastic FP8 rounding realization from its small direction file. Read the
actual pinned changed weights AND scales. A later FP8 or EXL3 requantization of
that ideal edit is a distinct reference, even with apparently similar seeds.

## Bounded public source access

`ablit2lora.range_weights.RangeWeights` is independent of `LazyWeights`. It
requires explicit byte budgets; uses exact HTTP ranges with identity/size
checks; refuses ignored ranges before reading the body; and never calls
`snapshot_download`. `hf_tensor_url` requires a pinned HF commit. Fetch/reuse
the pinned safetensors index first and select the shard by its exact key.

```python
from ablit2lora.range_weights import RangeWeights, hf_tensor_url

reader = RangeWeights(hf_tensor_url(repo, commit, shard), max_bytes=1_048_576)
weight_rows = reader.get_rows("layers.0.attn.wo_b.weight", 0, 32)
scale_rows = reader.get_rows("layers.0.attn.wo_b.scale", 0, 1)
# Decode format-aware using the pinned source config. Do not treat FP4 codes
# as dense weights, or raw E8M0 exponent bytes as numerical scale values.
# reader.receipts records range offsets, byte counts and payload checksums.
```

These row slices are samples, not a whole-model check. A sampled output-row
block has no claim to the full matrix's singular spectrum. Explicitly budget
full-writer reads only when needed, reuse already verified identical sources,
and do not download full duplicate checkpoints. This reader is for public
HTTP(S) files; it does not implement private-Hub credential discovery.

## Dense export / rank sweep

Use the verified quantizer/loader in an isolated offline environment to export
canonical `W[out,in]` matrices as `base_live` and `reference` safetensors entries.
Optional `activations[tokens,in]` must be real, matched held-out module inputs;
random fixtures only test the tool. Record exact tensor key, transpose/rotation,
TP partition/reduction, source and target revisions/pack hashes, recipe identity,
decoder, export dtype and sample coordinates in the provenance JSON. If the
decoder exports rounded BF16 rather than the exact logical matrix, record and
bound that export error too. Do not run an unbounded CPU SVD on a whole model.

```bash
python examples/quantized_reference.py --pair exported-pair.safetensors \
  --provenance exported-pair.json --out rank-audit \
  --ranks 1 2 4 8 16 --factor-dtype float32 --arithmetic-dtype bfloat16
```

The script performs one float64 CPU SVD per bounded pair. It reports the
Eckart-Young weight-space tail at every requested rank, post-factor-cast
residuals versus both Delta_eff and T, adapter bytes, measured activation
residuals, staged two-GEMM CPU error, and a deliberately dropped-adapter
negative control. It writes true-rank unit-scale PEFT factors, reads them
back, verifies bitwise equality, and measures the reloaded residual. These
artifacts use a dummy module/base and are **not deployment adapters**.

Weight-optimal SVD is not necessarily activation-optimal. Calibration-weighted
fitting is a separate optimization and still needs held-out checking. A tiny
relative error versus the large base norm can hide a large error versus the
edit; report both. No threshold is silently selected, and no tail is erased by
the source converter's noise model. Samples establish sample-specific results
only. A high/full-rank quantization delta may require a large correction or
cannot meet a given small-adapter budget.

## Runtime and whole-model acceptance (separate gate)

The CPU forward casts GEMM operands/results and the final sum to the selected
arithmetic dtype. It **does not emulate** EXL3 kernels, TP reductions, CUDA graphs,
expert fusion, mHC, MTP or TensorFold. Test actual serialized factors on the
actual engine with paired activations across prefill/decode, solo/mixed rows,
TP ranks, cache/spill and all supported modality paths. Check stock invariance,
identity isolation and an intentionally dropped/misrouted-adapter control.

For matched teacher-forced logits, `reference.logit_metrics(T_logits, C_logits)`
reports mean/max KL(T||C), top-1 agreement, logit error and max log-probability
error. Separately compute T-versus-S and T-versus-dropped negative controls.
Match token IDs/positions, tokenizer/template, precision, thinking policy and
cache state; do not compare free-running sequences position by position after
they diverge. Near-zero held-out KL is evidence, not proof of all future behavior.

Also compare native published Wabl, independently quantized T, live stock S
and adapter candidate C on capability, refusal/completion and long-context
tests with paired uncertainty. Measure mixed four-lane performance separately.
No reference pipeline is qualified until missing quantizer/calibration inputs
are recovered and stock replay succeeds. An unsupported runtime remains
unsupported regardless of offline audit results.
