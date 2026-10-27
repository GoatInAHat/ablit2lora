import json
import runpy
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from ablit2lora.adapter import read_adapter, write_adapter
from ablit2lora.reference import audit_pair, logit_metrics


def test_effective_quantized_delta_is_not_assumed_low_rank(tmp_path):
    g = torch.Generator().manual_seed(42)
    original = torch.randn(16, 24, generator=g)
    edit = torch.randn(16, 1, generator=g) @ torch.randn(1, 24, generator=g) * 0.1
    # A deterministic coarse quantizer, NOT EXL3; reference independently rounded.
    base = torch.round(original * 8) / 8
    reference = torch.round((original + edit) * 8) / 8
    x = torch.randn(30, 24, generator=g)
    report, factors = audit_pair(base, reference, ranks=[1, 2, 8, 16], activations=x)
    assert torch.linalg.matrix_rank(edit) == 1
    assert torch.linalg.matrix_rank(reference - base) > 1
    assert report["ranks"][0]["best_tail_relative"] > 0.05
    assert report["ranks"][-1]["best_tail_norm"] == 0
    assert report["ranks"][-1]["serialized_residual_vs_delta"]["relative"] < 1e-6
    assert report["ranks"][0]["best_tail_norm"] > report["ranks"][2]["best_tail_norm"]
    A, B = factors[8]
    write_adapter(tmp_path, base_model="synthetic", entries={"layer": (A, B)}, rank=8)
    restored, _ = read_adapter(tmp_path)
    Ar, Br = restored["layer"]
    assert torch.equal(A, Ar) and torch.equal(B, Br)
    measured = float((reference.double() - base - Br.double() @ Ar.double()).norm())
    assert measured == pytest.approx(report["ranks"][2]["serialized_residual_vs_delta"]["norm"])
    assert report["runtime_validated"] is False


def test_factor_cast_and_staged_arithmetic_are_measured_separately():
    g = torch.Generator().manual_seed(71)
    base = torch.randn(12, 10, generator=g)
    ref = base + torch.randn(12, 2, generator=g) @ torch.randn(2, 10, generator=g)
    x = torch.randn(50, 10, generator=g)
    report, _ = audit_pair(base, ref, ranks=[2], activations=x,
                           factor_dtype=torch.bfloat16, arithmetic_dtype=torch.bfloat16)
    rec = report["ranks"][0]
    assert rec["factor_cast_error_vs_ideal"]["norm"] > 0
    assert rec["serialized_residual_vs_delta"]["norm"] > rec["best_tail_norm"]
    assert rec["activation"]["simulated_staged_vs_dense"]["norm"] > 0
    assert rec["activation"]["simulated_dropped_vs_dense"]["norm"] > 10 * rec[
        "activation"]["simulated_staged_vs_dense"]["norm"]


def test_zero_delta_and_zero_activations_are_explicit():
    report, _ = audit_pair(torch.zeros(3, 4), torch.zeros(3, 4), ranks=[1],
                           activations=torch.zeros(2, 4))
    assert report["ranks"][0]["best_tail_relative"] is None
    assert report["ranks"][0]["serialized_residual_vs_delta"]["norm"] == 0


def test_finite_fp16_outputs_do_not_overflow_relative_error_denominators():
    report, _ = audit_pair(torch.zeros(4, 4), torch.eye(4) * 1000, ranks=[1],
                           activations=torch.ones(3000, 4), arithmetic_dtype=torch.float16)
    metrics = report["ranks"][0]["activation"]
    assert metrics["simulated_staged_vs_dense"]["relative"] == pytest.approx(3 ** 0.5 / 2)
    assert metrics["simulated_dropped_vs_dense"]["relative"] == pytest.approx(1)
    assert metrics["simulated_staged_vs_dense"]["relative"] == pytest.approx(
        metrics["real_arithmetic_residual"]["relative"])


def test_small_inputs_cannot_allocate_an_oversized_activation_output():
    with pytest.raises(ValueError, match="activation output audit exceeds"):
        audit_pair(torch.zeros(65, 1), torch.ones(65, 1), ranks=[1],
                   activations=torch.ones(65, 1), max_elements=128)


@pytest.mark.parametrize("kwargs", [{"ranks": []}, {"ranks": [0]}, {"ranks": [True]},
                                    {"ranks": [5]}, {"ranks": [1], "max_elements": 1}])
def test_invalid_budgets_and_ranks_fail(kwargs):
    with pytest.raises(ValueError):
        audit_pair(torch.ones(3, 4), torch.ones(3, 4), **kwargs)


def test_logits_are_paired_with_explicit_negative_control():
    ref = torch.tensor([[0., 8., 1.], [5., 0., 1.]])
    same = logit_metrics(ref, ref)
    dropped = logit_metrics(ref, ref.flip(1))
    assert same["mean_kl_ref_to_candidate"] == 0
    assert same["top1_agreement"] == 1
    assert dropped["mean_kl_ref_to_candidate"] > 1
    with pytest.raises(ValueError):
        logit_metrics(ref, torch.zeros(1, 3))


def test_simulation_overflow_is_not_reported_as_json_metrics():
    with pytest.raises(ValueError, match="overflowed"):
        audit_pair(torch.full((2, 2), 1e10), torch.full((2, 2), 2e10), ranks=[1],
                   activations=torch.ones(1, 2), arithmetic_dtype=torch.float16)
    with pytest.raises(ValueError, match="factor cast overflowed"):
        audit_pair(torch.zeros(2, 2), torch.full((2, 2), 2e10), ranks=[1],
                   factor_dtype=torch.float16)


def test_export_script_produces_reloaded_rank_artifacts(tmp_path, monkeypatch):
    pair = tmp_path / "pair.safetensors"
    save_file({"base_live": torch.zeros(4, 5), "reference": torch.eye(4, 5),
               "activations": torch.ones(2, 5)}, str(pair))
    provenance = tmp_path / "provenance.json"
    provenance.write_text('{"scope": "synthetic CLI test, not EXL3"}')
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["quantized_reference.py", "--pair", str(pair),
                                     "--provenance", str(provenance), "--out", str(out),
                                     "--ranks", "1", "4", "--factor-dtype", "bfloat16"])
    script = Path(__file__).resolve().parents[1] / "examples" / "quantized_reference.py"
    runpy.run_path(str(script), run_name="__main__")
    report = json.loads((out / "audit.json").read_text())
    assert report["runtime_validated"] is False
    assert report["provenance_verified"] is False
    assert report["ranks"][0]["reloaded_residual_norm"] > 1
    assert report["ranks"][1]["reloaded_residual_norm"] == 0
    assert (out / "rank-4" / "adapter_model.safetensors").exists()
