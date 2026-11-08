"""Source-shaped ExpansionRx official units, nominal coordinates, and full byte targets."""

import math
import json

import pytest
import torch
from torch.nn import functional as F

from mora3.data.admet import AdmetObservation, GATES, NativeCell, canonical
from mora3.model.admet_model import SharedAdmetModel
from test_pubchem_admet_mup import pubchem


def make_axes():
    from mora3.data.expansionrx_current import ExpansionRxSourceAxes, _ENDPOINTS, _NOMINAL, _partition
    parts = []
    for key in _ENDPOINTS:
        endpoint = key.removeprefix('ExpansionRx:')
        nominal = key in _NOMINAL
        parts.append(_partition(dict(
            endpoint=dict(endpoint_key=key, source_endpoint=endpoint, readout=endpoint,
                          unit=None if nominal else 'synthetic-native-unit',
                          unit_basis='unresolved' if nominal else 'official_source_schema',
                          measurement_type='synthetic-observed-coordinate'),
            assay_context={}, molecule_class='small_molecule',
        )))
    template = canonical(dict(molecule_class='small_molecule', representation_type='native_SMILES',
                              native_modifications=None, source_identity_missingness=[]))
    return ExpansionRxSourceAxes('a' * 64, 'b' * 64, tuple(sorted(parts, key=lambda part: part.identity)),
                                 template, 138, 887)


def make(width=8):
    from mora3.model.expansionrx_native_model import ExpansionRxNativeModel, expansionrx_official_contracts
    axes = make_axes()
    decoder = pubchem(width)
    admet = SharedAdmetModel(decoder.shared, expansionrx_official_contracts(axes),
                            parameterization_policy='mup-admet-v1')
    return ExpansionRxNativeModel(admet, decoder, axes).double().eval()


def observation(axes, partition, literal, *, smiles='C(CC)O'):
    native = dict(
        schema='mora3.ADMETNativeObservation/v1',
        entity=dict(json.loads(axes.native_entity_template), representation=smiles),
        known_context=partition.context.as_mapping(), endpoint=partition.definition.native(),
        evidence_role='source_curated_measurement', cell=NativeCell.parse(literal).as_mapping(),
        absence_is_not_negative=True, native_fields_not_projected_remain_in_source_record=True,
        source_native_axis_cell_multiplicity=1, independent_experimental_measurement_count=None,
        inferred_unit_conversions=[], admission={key: False for key in GATES},
        native_source_record={partition.definition.native()['source_endpoint']: literal},
    )
    return AdmetObservation.from_native(canonical(native), applicability='applicable')


def tasks(model, direction):
    from mora3.data.expansionrx_current import ExpansionRxNativeTask
    official = next(part for part in model.axes.partitions if not part.nominal)
    nominal = next(part for part in model.axes.partitions if part.nominal)
    return tuple(ExpansionRxNativeTask(
        partition, direction, 'train', observation(model.axes, partition, literal), ('fixture', i, 'b' * 64))
        for i, (partition, literal) in enumerate(((official, '2.75'), (nominal, '<3.5')))
    )


def test_actual_official_owner_and_nominal_censored_equations_keep_units_separate():
    model = make(16)
    originals = tasks(model, 'entity_to_measurement')
    result = model._prediction_objectives(originals)
    left, right = result
    assert left.partition.definition.unit == 'synthetic-native-unit'
    assert right.partition.definition.unit is None
    assert left.censor_relations == ('=',) and right.censor_relations == ('<',)
    expected_equal = (.5 * ((2.75 - left.mean) / left.scale).square() + left.scale.log()
                      + .5 * math.log(2 * math.pi)).mean()
    expected_left = -torch.special.log_ndtr((3.5 - right.mean) / right.scale).mean()
    torch.testing.assert_close(left.loss, expected_equal, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(right.loss, expected_left, rtol=1e-13, atol=1e-13)
    (left.loss + right.loss).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.admet.heads.parameters())
    assert model.nominal_heads[right.partition.identity].weight.grad.abs().sum() > 0
    assert model.decoder.shared is model.admet.shared
    with pytest.raises(ValueError, match='admission'):
        view, contract = model._official_view_and_contract(originals[0])
        model.admet.observed_loss(dict(mean=left.mean[0], scale=left.scale[0]), view, contract)


def test_inverse_target_full_source_connectivity_and_eos_from_exact_desired_query():
    model = make(16)
    originals = tasks(model, 'measurement_to_entity')
    queries = tuple((row.partition, row.observation.cell) for row in originals)
    context = model.inverse_condition(queries)
    changed = observation(model.axes, originals[0].partition, '2.75', smiles='N(CC)O')
    torch.testing.assert_close(context, model.inverse_condition(
        ((originals[0].partition, changed.cell), queries[1])), rtol=0, atol=0)
    expected = []
    decoder = model.decoder
    for i, row in enumerate(originals):
        ids = torch.tensor(decoder.codec.encode(row.native_connectivity_bytes()))
        hidden, _ = decoder.rnn(decoder.embedding(ids[:-1][None]), context[i:i+1][None].contiguous())
        expected.append(F.cross_entropy(decoder._readout(hidden)[0], ids[1:]))
    actual = model._generation_objective(originals, chunk_size=2, recompute=True)
    torch.testing.assert_close(actual, torch.stack(expected).mean(), rtol=1e-13, atol=1e-13)
    actual.backward()
    assert decoder.rnn.weight_hh_l0.grad.abs().sum() > 0


def test_width_family_exact_all_ten_predictions_and_shared_byte_owner():
    from mora3.model.expansionrx_native_model import expansionrx_parameter_roles, expansionrx_width_family
    members = tuple(make(width) for width in (8, 12, 16))
    axis = ('protein-coordinate',)
    family = expansionrx_width_family(*members, identifier_axis=axis)
    assert family.total_numel == tuple(sum(p.numel() for p in model.parameters()) for model in members)
    roles = expansionrx_parameter_roles(members[-1], identifier_axis=axis)
    outputs = [role.axes[0] for name, role in roles.items() if
               name.endswith('.weight') and name.startswith(('admet.heads.', 'nominal_heads.'))]
    assert len(outputs) == len(set(outputs)) == 10
    alias = 'shared.backbone.layers.0.attention.qkv.weight'
    assert roles['admet.' + alias] == roles['decoder.' + alias]
