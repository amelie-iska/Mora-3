import copy
import pytest
import torch
from test_instance_schedule import make, cases
from test_color_optimizer import fingerprint


def boundary(s,lr=.001):
    from mora3.model.instance_optimizer import InstanceOptimizerBoundary
    return InstanceOptimizerBoundary(s,learning_rate=lr)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_all_heads_allocate_once_and_refuse_without_mutation(tmp_path,cases,device,monkeypatch):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    s=make(tmp_path,cases,device);rng=torch.get_rng_state().clone()
    b=boundary(s)
    assert torch.equal(rng,torch.get_rng_state())
    params=tuple(p for g in b.optimizer.param_groups for p in g['params'])
    assert tuple(map(id,params))==tuple(map(id,s.model.parameters()))
    assert len(params)==len(set(map(id,params)))
    assert set(map(id,s.model.instance.parameters())) <= set(map(id,params))
    assert set(map(id,s.model.parent.parameters())) <= set(map(id,params))
    assert b.steps==0 and not b.optimizer.state
    r=b.preflight()
    assert r['parameter_count']==sum(p.numel() for p in params)
    assert r['provider_binding']==s.state_dict()['binding']
    assert r['parameter_names']==[n for n,_ in s.model.named_parameters()]
    assert not any(r[k] for k in ('training_admission','execution_supported','historical_optimizer_adoption'))
    r['optimizer_groups'][0]['lr']=50
    assert b.preflight()['optimizer_groups'][0]['lr']==.001
    s.model.eval();s.model.instance.train()
    next(s.model.parameters()).grad=torch.ones_like(next(s.model.parameters()))
    monkeypatch.setattr(s.model,'instance_objective',lambda *a,**k:pytest.fail('unexpected neural call'))
    before=fingerprint(b)
    with pytest.raises(ValueError,match='admission'):b.step()
    assert fingerprint(b)==before


@pytest.mark.parametrize('change',['schedule','model','weight','source','block','module','alias','config',
    'params','order','lr','state','optimizer','steps'])
def test_live_drift_fails_without_mutation(tmp_path,cases,change):
    s=make(tmp_path,cases);b=boundary(s)
    if change=='schedule':b.schedule=copy.copy(s)
    elif change=='model':b.model=copy.deepcopy(s.model)
    elif change=='weight':s.weight=8.
    elif change=='source':s.tasks=()
    elif change=='block':s.pixel_block_size=2
    elif change=='module':s.model.instance.context.graph.incidence=copy.deepcopy(s.model.instance.context.graph.incidence)
    elif change=='alias':s.model.instance.color.pixels=copy.deepcopy(s.model.instance.color.pixels)
    elif change=='config':s.model.instance.config['id_width']=99
    elif change=='params':b.optimizer.param_groups[0]['params'].pop()
    elif change=='order':b.optimizer.param_groups[0]['params'].reverse()
    elif change=='lr':b.optimizer.param_groups[0]['lr']=1.
    elif change=='state':b.optimizer.state[next(s.model.parameters())]['step']=torch.tensor(1.)
    elif change=='optimizer':b.optimizer=torch.optim.Adam(s.model.parameters(),lr=.001)
    elif change=='steps':b.steps=True
    before=fingerprint(b)
    with pytest.raises(ValueError):b.step()
    assert fingerprint(b)==before


@pytest.mark.parametrize('lr',[True,0,-1,float('nan'),float('inf')])
def test_invalid_rate(tmp_path,cases,lr):
    with pytest.raises(ValueError,match='learning rate'):boundary(make(tmp_path,cases),lr)


def test_exact_provider_required():
    with pytest.raises(ValueError,match='provider'):boundary(object())


def test_future_admission_does_not_enable_untested_execution(tmp_path,cases,monkeypatch):
    from mora3.model.color_instance_task import ColorInstanceTask
    b=boundary(make(tmp_path,cases))
    monkeypatch.setattr(ColorInstanceTask,'require_training',lambda _:None)
    before=fingerprint(b)
    with pytest.raises(ValueError,match='execution/recovery'):b.step()
    assert fingerprint(b)==before
