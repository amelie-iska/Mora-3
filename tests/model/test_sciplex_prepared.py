"""Prepared handles must preserve admitted semantics and exact learned updates."""
import copy
import pytest
import torch
from test_sciplex_shared import task
from mora3.model.sciplex import SciPlexSharedModel
from mora3.model import sciplex_preparation as preparation


def prepare(t,device='cpu'):
    assert hasattr(preparation,'PreparedSciPlexTask'), 'admission-bound prepared handle missing'
    return preparation.PreparedSciPlexTask.from_admitted(t,axis_ids=('A','B'),device=device,dtype=torch.float32)


@pytest.mark.parametrize('direction',['response','intervention'])
def test_binding_masks_mutation_and_target_exclusion(direction):
    t=task(direction);p=prepare(t)
    assert p.canonical_bytes()==t.canonical_bytes()
    t.assert_resume_matches(p)
    a=p.project();a['inputs'].clear()
    for group,values in t.project(dtype=torch.float32).items():
        for k,v in values.items():assert torch.equal(p.project()[group][k],v)
    other=prepare(task(direction,counts=(99,88)) if direction=='response' else task(direction,identity='SAHA'))
    m=SciPlexSharedModel(('A','B'),width=8)
    for k,v in m.graph(p).items():assert torch.equal(v,m.graph(other)[k])
    with pytest.raises(ValueError): preparation.PreparedSciPlexTask.from_admitted(t.task,axis_ids=('A','B'))
    with pytest.raises(ValueError): preparation.PreparedSciPlexTask.from_admitted(t,axis_ids=('B','A'))
    with pytest.raises(ValueError): SciPlexSharedModel(('B','A'),width=8).graph(p)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('direction',['response','intervention'])
def test_prepared_exact_updates_and_cold_resume(device,direction,tmp_path):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('no CUDA')
    from mora3.model.training import save_checkpoint,load_checkpoint
    torch.manual_seed(3701)
    a=SciPlexSharedModel(('A','B'),width=8).to(device);b=copy.deepcopy(a)
    ts=[task(direction)];ps=[prepare(ts[0],device)]
    oa=torch.optim.Adam(a.parameters());ob=torch.optim.Adam(b.parameters())
    for _ in range(2):
        oa.zero_grad();ob.zero_grad()
        assert torch.equal(a(ts),b(ps))
        x,y=a.loss(ts),b.loss(ps);assert torch.equal(x,y)
        x.backward();y.backward()
        for p,q in zip(a.parameters(),b.parameters()):
            if p.grad is not None:assert torch.equal(p.grad,q.grad)
        oa.step();ob.step()
    path=tmp_path/'state.pt';save_checkpoint(path,b,ob,step=2,binding=b.binding(ps))
    ob.zero_grad();expected=b.loss(ps);expected.backward();ob.step();state=copy.deepcopy(b.state_dict())
    assert load_checkpoint(path,b,ob,binding=b.binding(ts))==2
    cold=[prepare(ts[0],device)];ob.zero_grad();actual=b.loss(cold)
    assert torch.equal(actual,expected);actual.backward();ob.step()
    for k,v in state.items():
        assert torch.equal(v,b.state_dict()[k]) if isinstance(v,torch.Tensor) else v==b.state_dict()[k]


def test_prepared_handle_fields_are_frozen_and_precision_fallback_exact():
    from dataclasses import FrozenInstanceError
    p=prepare(task())
    with pytest.raises(FrozenInstanceError):p.prepared_direction='intervention'
    with pytest.raises(FrozenInstanceError):p.prepared_axis_ids=('B','A')
    with pytest.raises(FrozenInstanceError):del p.prepared_direction
    for g,values in task().project(dtype=torch.float64).items():
        for k,v in values.items():assert torch.equal(p.project(dtype=torch.float64)[g][k],v)
