"""Test-only nontruncating protein-prior recurrent target integration.

All records here are synthetic, never campaign training/source evidence.
"""
from dataclasses import replace

import pytest
import torch

from test_native_sequence_priors import make, records
import mora3.model.native_sequence_priors as priors
from mora3.model.source_byte_bounded_adjoint import source_byte_bounded_adjoint


def test_long_protein_uses_full_chunk_adjoint_and_exact_first_order_gradients(monkeypatch):
    model = make()
    protein = replace(records()[1], target_bytes=b"M" * 49 + b"GHQ")
    args = dict(chunk_size=7, recompute=True)
    reference = model._reference_objective((protein,), chunk_size=7, recompute=False)
    params = tuple(model.parameters())
    ref_grad = torch.autograd.grad(reference, params, allow_unused=True)
    entered = []

    def observed(decoder, context, targets, **bounds):
        entered.append((targets, bounds))
        return source_byte_bounded_adjoint(decoder, context, targets, **bounds)

    monkeypatch.setattr(priors, "source_byte_bounded_adjoint", observed, raising=False)
    actual = model._reference_objective((protein,), **args)
    assert len(entered) == 1
    assert entered[0][0] == (protein.target_bytes,)
    assert entered[0][1]["chunk_size"] == 7
    assert entered[0][1]["max_total_target_bytes"] == model.max_total_target_bytes
    torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)
    actual_grad = torch.autograd.grad(actual, params, allow_unused=True)
    for x,y in zip(actual_grad,ref_grad,strict=True):
        assert (x is None) == (y is None)
        if x is not None:
            torch.testing.assert_close(x,y,rtol=1e-10,atol=1e-10)


def test_protein_overbudget_is_refused_before_context_or_native_objective(monkeypatch):
    from mora3.model.native_sequence_priors import NativeSequencePriorModel
    original = make()
    model = NativeSequencePriorModel(original.nucleotide,original.decoder,max_total_target_bytes=4096)
    protein = replace(records()[1], target_bytes=b"M" * 4097)
    def fail(*args, **kwargs):
        raise AssertionError("entered neural computation before source/resource binding")
    monkeypatch.setattr(model, "_condition", fail)
    with pytest.raises(ValueError):
        model._reference_objective((protein,),chunk_size=128,recompute=True)
