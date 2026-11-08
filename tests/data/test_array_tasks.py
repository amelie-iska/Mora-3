"""Array projection correctness; synthetic fixtures are not training coverage."""
import json
import pytest
import torch
from mora3.data import array_tasks as api


def field(values=(0, 2), status='observed'):
    return api.OrderedArray('expression', ('A', 'A'), values,
                            'source_column_position', 'counts', status)


def task(role='target', values=(0, 2), status='observed'):
    return api.ArrayTask((field(values, status),), (('expression', role),), 'a'*64)


def test_target_changes_never_change_inputs_or_input_metadata():
    a, b = task(), task(values=(800, 900))
    assert a.project()['inputs'] == b.project()['inputs'] == {}
    assert a.input_metadata() == b.input_metadata()
    assert torch.equal(a.project()['targets']['expression'], torch.tensor([0.,2.]))
    assert a.project()['target_masks']['expression'].tolist() == [True, True]


def test_two_masks_preserve_order_and_do_not_invent_missing_targets():
    x = task('condition').project()
    assert x['inputs']['expression'].tolist() == [0.,2.]
    assert x['targets'] == {}
    y = task(values=None, status='unavailable').project()
    assert y['targets'] == {}  # missing is never a zero vector
    assert y['target_masks']['expression'].tolist() == [False,False]
    for role in ('evaluation','unused'):
        assert task(role).project()['inputs'] == {}


def test_canonical_resume_roundtrip_and_mutation_guard():
    a=task(); raw=a.canonical_bytes()
    assert api.ArrayTask.from_bytes(raw).canonical_bytes()==raw
    assert task('condition').canonical_bytes()!=raw
    bad=json.loads(raw); bad['version']=8
    with pytest.raises(ValueError): api.ArrayTask.from_bytes(json.dumps(bad))
    with pytest.raises(ValueError): api.ArrayTask.from_bytes('{"version":1,"version":1}')


@pytest.mark.parametrize('values,status', [((1,), 'observed'),((0,float('nan')),'observed'),
    ((True,2),'observed'), (None,'observed'),((1,2),'unavailable'),((1,2),'fake')])
def test_invalid_arrays(values,status):
    with pytest.raises(ValueError): field(values,status)


def test_roles_availability_predictions_and_digest_fail_closed():
    for role in ('condition','desired'):
        with pytest.raises(ValueError): task(role, None,'unavailable').project()
    with pytest.raises(ValueError): task(status='source_predicted').project()
    with pytest.raises(ValueError): api.ArrayTask((field(),),(), 'a'*64)
    with pytest.raises(ValueError): api.ArrayTask((field(),), (('expression','target'),)*2,'a'*64)
    with pytest.raises(ValueError): api.ArrayTask((field(),), (('expression','target'),),'unknown')


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_matches_cpu_without_truncation():
    a=api.OrderedArray('x',tuple(map(str,range(5000))),tuple(range(5000)), 'position','counts','observed')
    t=api.ArrayTask((a,), (('x','condition'),), 'a'*64)
    assert torch.equal(t.project(device='cuda')['inputs']['x'].cpu(),t.project()['inputs']['x'])
