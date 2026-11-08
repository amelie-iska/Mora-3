from dataclasses import replace
import copy

import pytest
import torch

from mora3.data.admet import canonical, digest
from mora3.data.string_score_current import (
    FAMILY, REVISION, SCORE_AXIS, SCOPE, TASK, UNIT, StringScoreAxes, StringScoreCondition, project_string_score_row,
)
from mora3.model.string_score_model import NativeStringScoreModel, string_score_parameter_roles
from mora3.model.parameterization import parameter_ledger, width_family
from test_annotation_mup import make as annotation
from test_sciplex_mup import make as response


_IDS = ('9606.ENSP000000001', '9606.ENSP000000002')
_SEQ = dict(zip(_IDS, ('KLMNPQ', 'ACDXA'), strict=True))
_SCORE = (0, 120, 431, 915)


def document():
    return dict(
        schema='mora3.CorpusTask/v1', task=TASK,
        conditions=dict(protein1_source_id=_IDS[0], protein2_source_id=_IDS[1], taxon=9606, association_scope=SCOPE),
        targets={field: dict(unit=UNIT, value=value) for field, value in zip(SCORE_AXIS, _SCORE, strict=True)},
        audit=dict(source_oriented_rows=[2, 100], source_unordered_pair_group='original source pair audit',
                   experimental_channel_supported=False, experimental_truth=False,
                   negative_label=None, direct_binding_demonstrated=None),
        missing=['exact_protein_sequences'],
    )


def packet(doc=None):
    raw = canonical(document() if doc is None else doc)
    return project_string_score_row(dict(
        source_family=FAMILY, source_revision=REVISION, native_schema='mora3.CorpusTask/v1', task_name=TASK,
        split='train', condition_context_json='{}', admitted_for_task=True, model_adapter_supported=False,
        evidence_class='source_integrated_computed_association_confidence_prior_not_experimental_binding',
        native_payload=raw, native_payload_sha256=digest(raw), source_sha256='c'*64,
    ), 'source.parquet', 3, _SEQ, 'c'*64)


def make(width=8, *, physical=False):
    shared = annotation(width, physical=physical, backbone=response(width).backbone)
    axes = StringScoreAxes('a'*64, 'b'*64, 'c'*64, 'd'*64, 8, 20, 'ACDKLMNPQX')
    return NativeStringScoreModel(shared, axes).double().eval()


def test_native_packet_preserves_complete_same_release_sequence_and_literal_zero():
    p = packet()
    assert p.condition.sequences == ('KLMNPQ', 'ACDXA')
    assert p.provider_pair == _IDS and p.oriented_source_lines == (2, 100)
    assert p.scores == _SCORE
    assert not hasattr(p.condition, 'provider_pair')
    changed = document()
    changed['targets']['combined_score']['value'] = 1000
    changed['audit']['source_unordered_pair_group'] = 'independently changed audit'
    other = packet(changed)
    assert p.condition == other.condition and p.scores != other.scores


@pytest.mark.parametrize('change', ('bad_score_unit', 'wrong_range', 'false_source_negative', 'missing_endpoint', 'target_leak'))
def test_malformed_native_profile_refuses(change):
    d = document()
    if change == 'bad_score_unit':
        d['targets']['database_channel']['unit'] = 'probability'
    elif change == 'wrong_range':
        d['targets']['database_channel']['value'] = 1001
    elif change == 'false_source_negative':
        d['audit']['negative_label'] = True
    elif change == 'missing_endpoint':
        d['conditions']['protein2_source_id'] = '9606.ENSP009999999'
    else:
        d['conditions']['experimental_channel'] = 0
    with pytest.raises(ValueError):
        packet(d)


def test_exact_complete_channel_objective_graph_order_and_aliases():
    model = make(16, physical=True)
    p = packet()
    records = []
    hook = model.shared.payload.register_forward_pre_hook(
        lambda _m, args, kwargs: records.append((args[0], kwargs['input_fields'])), with_kwargs=True,
    )
    out = model._reference_objective((p,))
    hook.remove()
    record, fields = records[0]
    assert tuple(node.measurements[0].value for node in record.nodes) == tuple(sorted(_SEQ.values()))
    assert all(ident not in record.canonical_bytes().decode() for ident in _IDS)
    assert len(fields) == 4 and len(record.edges) == 1
    prediction = model((StringScoreCondition(tuple(reversed(p.condition.sequences))),))
    torch.testing.assert_close(prediction, out['predicted_fractions'], rtol=0, atol=0)
    values = torch.tensor(_SCORE, dtype=torch.float64) / 1000.
    expected = (prediction[0] - values).square().mean()
    torch.testing.assert_close(out['loss'], expected, rtol=1e-14, atol=1e-14)
    assert out['observed_source_score_positions'] == 4 and out['literal_zero_positions'] == 1
    out['predicted_fractions'].retain_grad()
    out['loss'].backward()
    assert torch.count_nonzero(out['predicted_fractions'].grad) == 4
    assert torch.count_nonzero(model.shared.backbone.incidence.weight.grad)
    assert torch.count_nonzero(model.readout.weight.grad)


def test_width_family_and_refuses_before_neural_on_invalid_axes_or_condition():
    models = [make(w, physical=True) for w in (8, 12, 16)]
    roles = [string_score_parameter_roles(m, identifier_axis=('context', 'observation')) for m in models]
    family = width_family(*(parameter_ledger(m, r) for m, r in zip(models, roles, strict=True)))
    assert family.total_numel == tuple(sum(p.numel() for p in m.shared.parameters()) + 4*(m.width+1) for m in models)
    model = models[0]
    def no_encoder(*args, **kwargs):
        pytest.fail('invalid condition must refuse before encoder')
    model.shared.payload.forward = no_encoder
    with pytest.raises(ValueError):
        model((StringScoreCondition(('AX', 'K'*21)),))
    with pytest.raises(ValueError):
        model._reference_objective((replace(packet(), scores=(True, 5, 7, 9)),))
    state = copy.deepcopy(model.state_dict())
    state['_extra_state']['objective'] = 'changed'
    state['readout.weight'].add_(1)
    before = model.readout.weight.detach().clone()
    with pytest.raises(ValueError, match='checkpoint'):
        model.load_state_dict(state)
    assert torch.equal(model.readout.weight, before)
