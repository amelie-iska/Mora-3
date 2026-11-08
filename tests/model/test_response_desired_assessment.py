"""Requested intervals are desired conditions, never measured expression labels."""
from dataclasses import FrozenInstanceError, replace, asdict
import copy
import hashlib
import json
import pytest
import torch
from test_sciplex_shared import task
from mora3.model.response_sampling import ResponseContext, sample_response
from mora3.model.response_flow import ResponseFlowHead
from mora3.model.sciplex import SciPlexSharedModel


def api():
    from mora3.model.response_desired import DesiredExpression, assess_desired_expression
    return DesiredExpression, assess_desired_expression


def case(device='cpu'):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.manual_seed(25701)
    c = ResponseContext.from_task(task())
    m = ResponseFlowHead(SciPlexSharedModel(c.axis_ids, width=4)).to(device).double()
    s = sample_response(m, [c], steps=2, seed=25701, sigma_min=.01, negative_policy='preserve')
    return c, s


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_inclusive_intervals_equations_missingness_and_detached_results(device):
    D, assess = api()
    c, s = case(device)
    # Deliberately independent analytic predictions, no neural quality claim.
    s['log_expression'] = torch.tensor([[-.5, 2.5]], device=device, dtype=torch.float64, requires_grad=True)
    d = D(c, (('A', 0., 1.), ('B', 1., 2.)))
    r = assess(s, (d,))
    assert r['per_record'][0]['squared_violation_mean'] == .25
    assert r['per_record'][0]['satisfied_count'] == 0
    assert r['per_record'][0]['assessed_count'] == 2
    assert r['evidence_role'] == 'model_prediction_desired_consistency'
    assert r['observed_label_count'] == 0
    s['log_expression'] = torch.tensor([[0., 2.]], device=device, dtype=torch.float64)
    r = assess(s, (d,))
    assert r['per_record'][0]['satisfied_count'] == 2
    assert r['per_record'][0]['squared_violation_mean'] == 0
    r = assess(s, (D(c, (('A', None, None),)),))
    assert r['per_record'][0]['assessed_count'] == 0
    assert r['per_record'][0]['squared_violation_mean'] is None
    assert r['per_record'][0]['unrequested_gene_count'] == 1
    assert r['per_record'][0]['unavailable_requested_count'] == 1
    assert not any(isinstance(v, torch.Tensor) for v in r.values())
    json.dumps(r, allow_nan=False)


@pytest.mark.parametrize('entries', [(), [('A', 0, 1)], (('Z', 0, 1),),
    (('A', 0, 1), ('A', 1, 2)), (('A', -1, 1),), (('A', 2, 1),),
    (('A', True, 1),), (('A', float('nan'), 1),), (('A', 0, float('inf')),),
    (('A', None, 1),), (('A', 0, None),), (['A', 0, 1],)])
def test_invalid_desired_contract(entries):
    D, _ = api()
    c = ResponseContext.from_task(task())
    with pytest.raises(ValueError): D(c, entries)


def test_immutable_context_axis_and_hidden_target_exclusion():
    D, assess = api()
    c, s = case()
    assert c == ResponseContext.from_task(task(counts=(999, 77), cell='hidden'))
    d = D(c, (('A', 0., 1.),))
    with pytest.raises(FrozenInstanceError): d.intervals = ()
    with pytest.raises(ValueError): D(replace(c, axis_ids=('A','A')), (('A',0,1),))
    with pytest.raises(ValueError): D(None, (('A',0,1),))
    for change in [dict(axis_ids=('B','A')), dict(source_sha256='c'*64),
                   dict(audit_sha256='d'*64), dict(intervention='SAHA'),
                   dict(dose_coordinate=999), dict(exposure_hours=999)]:
        with pytest.raises(ValueError): assess(s, (D(replace(c, **change), d.intervals),))
    before = copy.deepcopy(s['configuration'])
    r = assess(s, (d,))
    assert s['configuration'] == before
    s['log_expression'][0, 1] += 10
    r2 = assess(s, (d,))
    assert r['per_record'] == r2['per_record']
    assert r['prediction_sha256'] != r2['prediction_sha256']
    assert r['request_sha256'] == r2['request_sha256']


@pytest.mark.parametrize('bad', ['binding','space','axis','shape','nan','integer','requests','counts','overflow'])
def test_sample_binding_shape_space_and_finiteness_refusals(bad):
    D, assess = api()
    c, s = case()
    d = D(c, (('A',0.,1.),))
    if bad == 'binding': s['binding'] = 'f'*64
    if bad == 'space': s['configuration']['model']['space'] = 'raw-counts'
    if bad == 'axis': s['configuration']['model']['native']['axis_ids'] = ('B','A')
    if bad == 'shape': s['log_expression'] = torch.zeros(1,3)
    if bad == 'nan': s['log_expression'][0,0] = float('nan')
    if bad == 'integer': s['log_expression'] = torch.zeros(1,2,dtype=torch.int64)
    if bad == 'requests': d = None
    if bad == 'counts': s['configuration']['contexts'].append(asdict(c))
    if bad == 'overflow': s['log_expression'][0,0] = 1e200
    if bad in ('space','axis','counts'):
        s['binding'] = hashlib.sha256(json.dumps(s['configuration'],sort_keys=True).encode()).hexdigest()
    with pytest.raises(ValueError): assess(s, (d,))


@pytest.mark.parametrize('device', ['cpu','cuda'])
def test_batch_keeps_contexts_separate_and_gene_order_explicit(device):
    D, assess = api()
    c, s = case(device)
    c2 = replace(c, dose_coordinate=99)
    s['configuration']['contexts'].append(asdict(c2))
    s['binding'] = hashlib.sha256(json.dumps(s['configuration'],sort_keys=True).encode()).hexdigest()
    s['log_expression'] = torch.tensor([[.5, 3.],[.5, 3.]],device=device,dtype=torch.float64)
    ds = (D(c, (('B',1.,2.),('A',0.,1.))), D(c2, (('A',1.,2.),)))
    r = assess(s, ds)
    assert [x['squared_violation_mean'] for x in r['per_record']] == [.5,.25]
    assert r['per_record'][0]['gene_ids'] == ['B','A']
    assert r['per_record'][1]['unrequested_gene_count'] == 1
    with pytest.raises(ValueError): assess(s, ds[::-1])


def test_duplicate_native_gene_names_preserve_axis_but_refuse_ambiguous_requests():
    D, _ = api()
    c = replace(ResponseContext.from_task(task()), axis_ids=('A','B','B'))
    d = D(c, (('A',0.,1.),))
    assert d.context.axis_ids == ('A','B','B')
    with pytest.raises(ValueError): D(c, (('B',0.,1.),))
