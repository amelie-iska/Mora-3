"""NCI source roles, literal assay axes, full likelihood and shared aliases."""

from dataclasses import replace
from decimal import Decimal
import hashlib
import math

import pytest
import torch
from torch.nn import functional as F

from mora3.data.nci_aids import CONTEXT, EVIDENCE, FAMILIES, MISSING, PAIRED_AXIS, PANEL, POLICY, READOUT, REVISION, SOURCE_REF
from mora3.data.uniprot_annotations import _canonical
from test_pubchem_admet_mup import pubchem


def native_record(partition, direction, value, *, smiles='C(C)O'):
    from mora3.data.nci_current_assays import CONCLUSION, ENDPOINT, PAIRED, SMILES
    cond = dict(assay_panel=PANEL, source_stereochemistry_known=False)
    audit = dict(source_ref=dict(native_source=SOURCE_REF),
                 future_generated_outputs_require_model_produced_provenance=True)
    if partition.family != CONCLUSION:
        cond.update(concentration_unit=partition.unit, source_log10_highest_tested_concentration=-3.5)
        if partition.family == ENDPOINT:
            cond['assay_readout'] = partition.readout
            audit.update(source_flag='=', source_endpoint=next(k for k, v in READOUT.items() if v == partition.readout))
        else:
            cond['endpoint_axis'] = list(PAIRED_AXIS)
            audit.update(source_censor_flags=['=', '='], same_cell_or_replicate_pair=False,
                         source_lot_or_full_protocol_identity_verified=False,
                         selectivity_ratio_or_therapeutic_index_derived=False)
    if direction == 'generation':
        cond['desired_' + partition.output_field] = value
        target, dimensions = {SMILES: smiles}, len(smiles)
    else:
        cond[SMILES] = smiles
        target, dimensions = {partition.output_field: value}, 2 if partition.family == PAIRED else 1
    name = FAMILIES[partition.family] + direction + '_prior'
    raw = _canonical(dict(schema='mora3.CorpusTask/v1', task=name, conditions=cond,
                          targets=target, audit=audit, missing=MISSING))
    return dict(
        source_family=partition.family, task_name=name, native_schema='mora3.CorpusTask/v1',
        native_payload=raw, native_payload_sha256=hashlib.sha256(raw).hexdigest(),
        source_revision=REVISION, source_sha256=SOURCE_REF['sha256'], admitted_for_task=True,
        evidence_class=EVIDENCE, split_policy=POLICY, split='train',
        condition_context_json=_canonical(CONTEXT).decode(), full_axis_values=dimensions,
    )


def make(width=8):
    from mora3.data.nci_current_assays import NciAssaySourceAxes, _all_partitions
    from mora3.model.nci_assay_model import NativeNciAssayModel
    return NativeNciAssayModel(
        pubchem(width), NciAssaySourceAxes('a' * 64, 'b' * 64, _all_partitions(), 580, 714),
    ).double().eval()


def tasks(direction):
    from mora3.data.nci_current_assays import CONCLUSION, ENDPOINT, PAIRED, NciAssayPartition, _project
    definitions = (
        (NciAssayPartition(CONCLUSION, None, None), 'CM'),
        (NciAssayPartition(ENDPOINT, READOUT['EC50'], 'M'), -4.125),
        (NciAssayPartition(PAIRED, None, 'u'), [3.125, 1.25]),
    )
    return tuple(_project(native_record(p, direction, value), 'synthetic.parquet', i)
                 for i, (p, value) in enumerate(definitions))


def test_native_projection_preserves_literal_targets_and_refuses_target_or_unit_as_wrong_role():
    from mora3.data.nci_current_assays import SMILES, nci_known_condition
    prediction = tasks('prediction')
    inverse = tasks('generation')
    assert prediction[0].target == 'CM'
    assert prediction[1].target == (Decimal('-4.125'),)
    assert prediction[2].target == (Decimal('3.125'), Decimal('1.25'))
    assert all(record.target == b'C(C)O' and SMILES not in record.inputs() for record in inverse)
    with pytest.raises(ValueError):
        nci_known_condition(inverse[0].partition, 'generation', dict(inverse[0].inputs(), **{SMILES: 'C'}))
    with pytest.raises(ValueError):
        nci_known_condition(prediction[2].partition, 'prediction',
                            dict(prediction[2].inputs(), concentration_unit='M'))


def test_literal_class_ce_and_complete_native_density_match_independent_equations():
    model = make(16)
    originals = tasks('prediction')
    results = {record.partition: record for record in model._prediction_objectives(originals)}
    for source in originals:
        out = results[source.partition]
        if type(source.target) is str:
            expected = F.cross_entropy(out.logits, torch.tensor([source.partition.coordinate_axis.index(source.target)]))
        else:
            value = out.mean.new_tensor([[float(x) for x in source.target]])
            assert value.shape == out.mean.shape == out.scale.shape
            expected = (0.5 * ((value - out.mean) / out.scale).square() + out.scale.log()
                        + math.log(2 * math.pi) / 2).mean()
        torch.testing.assert_close(out.loss, expected, rtol=1e-13, atol=1e-13)
    sum(r.loss for r in results.values()).backward()
    assert torch.count_nonzero(model.decoder.shared.backbone.layers[0].attention.qkv.weight.grad)
    assert all(torch.count_nonzero(model.heads[t.partition.identity].weight.grad) for t in originals)


def test_inverse_full_native_bytes_eos_and_target_exclusion():
    model = make(16)
    sources = tasks('generation')
    queries = tuple((task.partition, task.inputs()) for task in sources)
    context = model.condition(queries, direction='generation')
    altered = replace(sources[0], target=b'NC', source_coordinate=('other', 4, 'e' * 64))
    other_queries = ((altered.partition, altered.inputs()), *queries[1:])
    torch.testing.assert_close(context, model.condition(other_queries, direction='generation'), rtol=0, atol=0)
    actual = model._generation_objective(sources, chunk_size=2, recompute=True)
    independent = []
    for row, source in enumerate(sources):
        decoder = model.decoder
        ids = torch.tensor(decoder.codec.encode(source.target))
        hidden, _ = decoder.rnn(decoder.embedding(ids[:-1][None]), context[row:row+1][None].contiguous())
        independent.append(F.cross_entropy(decoder._readout(hidden)[0], ids[1:]))
    torch.testing.assert_close(actual, torch.stack(independent).mean(), rtol=1e-13, atol=1e-13)
    actual.backward()
    assert torch.count_nonzero(model.decoder.rnn.weight_hh_l0.grad)


def test_partition_readout_roles_and_same_owner_width_family():
    from mora3.model.nci_assay_model import nci_assay_parameter_roles, nci_assay_width_family
    members = tuple(make(width) for width in (8, 12, 16))
    identifier_axis = ('protein-coordinate',)
    family = nci_assay_width_family(*members, identifier_axis=identifier_axis)
    roles = nci_assay_parameter_roles(members[-1], identifier_axis=identifier_axis)
    assert family.total_numel == tuple(sum(p.numel() for p in member.parameters()) for member in members)
    axes = [roles[f'heads.{task.partition.identity}.weight'].axes[0] for task in tasks('prediction')]
    assert len(set(axes)) == 3 and all(axis.kind == 'finite' for axis in axes)
    assert [members[-1].heads[task.partition.identity].out_features for task in tasks('prediction')] == [3, 2, 4]


def test_generated_connectivity_uses_same_decoder_and_retains_termination_and_provenance():
    model = make()
    queries = tuple((task.partition, task.inputs()) for task in tasks('generation'))
    with torch.no_grad():
        model.decoder.output.weight.zero_()
        model.decoder.output.bias.fill_(-100)
        model.decoder.output.bias[model.decoder.codec.eos] = 100
    result = model.generate(queries, max_new_tokens=3)
    assert result['terminated'] == (True, True, True)
    assert result['target_fields'] == ('molecule_source_connectivity_smiles',) * 3
    assert result['representation_bytes'] == (b'', b'', b'')
    assert result['evidence_class'] == 'model_produced' and result['representation_validated'] is False
