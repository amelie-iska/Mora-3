"""Exact native Tox21 call masks and full original molecule decoding."""

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from mora3.data.admet import canonical
from mora3.data.tox21 import INVERSE, Tox21View
from mora3.model.tox21 import SharedTox21Model
from test_pubchem_admet_mup import pubchem


def pattern(known_negative=False):
    from mora3.data.tox21_current import Tox21DesiredCallPattern
    values = (1, 0 if known_negative else None) + (None,) * 10
    return Tox21DesiredCallPattern(values, (True, known_negative) + (False,) * 10)


def make(width=8):
    from mora3.data.tox21_current import Tox21CurrentAxes
    from mora3.model.tox21_current_model import NativeTox21SourceModel
    decoder = pubchem(width)
    tox = SharedTox21Model(decoder.shared, parameterization_policy='mup-tox21-v1')
    return NativeTox21SourceModel(tox, decoder, Tox21CurrentAxes('a' * 64, 'b' * 64, (4, 2, 1), 332)).double()


def source_view(smiles='C[C@H](O)N'):
    query = pattern()
    return Tox21View(INVERSE, query.values, query.observed, canonical(query.inputs()),
                     canonical(dict(canonical_isomeric_smiles=smiles)), b'{}', 'a' * 64)


def test_exact_positive_negative_and_unknown_condition_separation():
    model = make(16)
    seen = []
    hook = model.tox21.call_input.register_forward_pre_hook(lambda _m, args: seen.append(args[0].detach().clone()))
    model.inverse_condition((pattern(), pattern(True)))
    hook.remove()
    assert len(seen) == 2 and seen[0].shape == seen[1].shape == (24,)
    assert torch.equal(seen[0][:12], seen[1][:12])
    assert seen[0][0] == 1 and seen[0][1] == 0
    assert seen[0][13] == 0 and seen[1][13] == 1
    with pytest.raises(ValueError):
        replace(pattern(), values=(1, 0) + (None,) * 10)


def test_complete_original_molecule_eos_and_no_hidden_target_inputs():
    model = make(16)
    original = source_view()
    decoder = model.decoder
    query = pattern()
    context = model.inverse_condition((query,))
    torch.testing.assert_close(context, model.tox21.condition((source_view('O'),)), rtol=0, atol=0)
    ids = torch.tensor(decoder.codec.encode(original.targets()['canonical_isomeric_smiles'].encode()))
    hidden, _ = decoder.rnn(decoder.embedding(ids[:-1][None]), context[None].contiguous())
    expected = F.cross_entropy(decoder._readout(hidden)[0], ids[1:])
    actual = model._reference_inverse_objective((original,), chunk_size=2, recompute=True)
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
    actual.backward()
    assert torch.count_nonzero(model.tox21.call_input.weight.grad)
    assert torch.count_nonzero(decoder.rnn.weight_hh_l0.grad)
    generated = model.generate((query,), max_new_tokens=3)
    assert generated['evidence_class'] == 'model_produced'
    assert generated['target_fields'] == ('canonical_isomeric_smiles',)


def test_both_shared_aliases_and_fixed_call_axes_are_counted_once():
    from mora3.model.tox21_current_model import tox21_current_model_parameter_roles, tox21_current_model_width_family
    members = tuple(make(width) for width in (8, 12, 16))
    axis = ('protein-coordinate',)
    family = tox21_current_model_width_family(*members, identifier_axis=axis)
    assert family.total_numel == tuple(sum(p.numel() for p in m.parameters()) for m in members)
    roles = tox21_current_model_parameter_roles(members[-1], identifier_axis=axis)
    assert roles['tox21.call_input.weight'].role == 'input'
    assert roles['tox21.output.weight'].role == 'output'
    alias = 'shared.backbone.layers.0.attention.qkv.weight'
    assert roles['tox21.' + alias] == roles['decoder.' + alias]
