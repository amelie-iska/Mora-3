"""Source-specific connected projection, using synthetic fixtures only."""
import copy
import json
import pytest
import torch
from mora3.data.gperturb_sciplex import array_task
from mora3.data.connected_sciplex import SciPlexTask


def sidecar(direction='response'):
    return array_task(dict(cell_id='cell-a', source_index=0, intervention='Dex',
        dose_coordinate=.001, dose_units='source_scaled_coordinate', counts=[0,2],
        n_genes=1,n_counts=20), ('A','A'), direction=direction)


def make(s=None):
    return SciPlexTask.from_sidecar(sidecar() if s is None else s, source_sha256='a'*64)


def assert_same_inputs(a,b):
    assert a.input_metadata()==b.input_metadata()
    for key,value in a.project()['inputs'].items():
        assert torch.equal(value,b.project()['inputs'][key])


@pytest.mark.parametrize('direction',['response','intervention'])
def test_hidden_targets_and_qc_never_enter_inputs(direction):
    a=sidecar(direction); b=copy.deepcopy(a)
    b['evaluation'].update(cell_id='other',source_index=8,n_counts=900,n_genes=99)
    if direction=='response':
        b['targets']['expression']['values']=[8,9]
        b['targets']['expression']['axis_ids']=['hidden','target']
    else:
        b['targets']['intervention'].update(identity='SAHA',source_coordinate=.5)
    assert_same_inputs(make(a),make(b))
    assert make(a).canonical_bytes()!=make(b).canonical_bytes()


def test_full_projection_roles_units_and_immutable_resume():
    s=sidecar(); t=make(s); raw=t.canonical_bytes()
    s['conditions']['intervention']['source_coordinate']=.9
    assert t.canonical_bytes()==raw
    x=t.project()
    assert x['inputs']['intervention_index'].item()==0
    assert x['inputs']['dose_coordinate'].item()==.001
    assert x['inputs']['exposure_hours'].item()==24
    assert x['targets']['expression'].tolist()==[0,2]
    assert t.input_metadata()['context']['baseline'] is None
    assert t.input_metadata()['dose_units']=='source_scaled_coordinate'
    restored=SciPlexTask.from_bytes(raw)
    assert restored.canonical_bytes()==raw
    restored.assert_resume_matches(t)
    with pytest.raises(ValueError,match='resume'): restored.assert_resume_matches(make(sidecar('intervention')))
    inv=make(sidecar('intervention')).project()
    assert inv['inputs']['expression'].tolist()==[0,2]
    assert 'dose_coordinate' not in inv['inputs']
    assert inv['target_masks']['intervention_index'].item() is True


def test_quarantine_cannot_be_overridden_by_source_or_caller():
    t=make()
    with pytest.raises(ValueError,match='Physical dose'): t.project(for_training=True)
    s=sidecar(); s['training_ready']=True
    with pytest.raises(ValueError): make(s)


@pytest.mark.parametrize('field,value', [('exposure_hours',48),('cell_line','K562'),('baseline',{}),('desired_response',{})])
def test_incompatible_source_context_rejected(field,value):
    s=sidecar();s['conditions']['context'][field]=value
    with pytest.raises(ValueError): make(s)


def test_distinct_doses_and_invalid_resume_metadata():
    s=sidecar();s['conditions']['intervention']['source_coordinate']=.5
    assert make(s).project()['inputs']['dose_coordinate'].item()!=make().project()['inputs']['dose_coordinate'].item()
    for mutation in ({'version':9},{'source_sha256':'unknown'},{'extra':1}):
        d=json.loads(make().canonical_bytes());d.update(mutation)
        with pytest.raises(ValueError): SciPlexTask.from_bytes(json.dumps(d))
    with pytest.raises(ValueError): SciPlexTask.from_bytes('{"version":1,"version":1}')


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_connected_projection():
    for direction in ('response','intervention'):
        t=make(sidecar(direction)); cpu=t.project();gpu=t.project(device='cuda')
        for group in cpu:
            for key in cpu[group]: assert torch.equal(cpu[group][key],gpu[group][key].cpu())
