import copy
import pytest
import torch
from test_color_schedule import make, cases
from mora3.model.profile_reconstruction import _digest


def boundary(schedule, **kwargs):
    from mora3.model.color_optimizer import ColorOptimizerBoundary
    return ColorOptimizerBoundary(schedule, learning_rate=kwargs.get('learning_rate', .001))


def fingerprint(b):
    return _digest(dict(model=b.model.state_dict(), optimizer=b.optimizer.state_dict(),
        gradients=[p.grad for p in b.model.parameters()],
        modes=[m.training for m in b.model.modules()],cpu_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,steps=b.steps))


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_exact_shared_allocation_and_admission_refusal_before_mutation(tmp_path,cases,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    s=make(tmp_path,cases,device); b=boundary(s)
    params=[p for g in b.optimizer.param_groups for p in g['params']]
    assert len(params)==len(set(map(id,params)))==len(tuple(s.model.parameters()))
    assert set(map(id,params))==set(map(id,s.model.parameters()))
    assert not b.optimizer.state and b.steps==0
    receipt=b.preflight()
    assert receipt['parameter_count']==sum(p.numel() for p in s.model.parameters())
    assert receipt['training_admission'] is False and receipt['execution_supported'] is False
    assert receipt['provider_binding']==s.state_dict()['binding']
    receipt['optimizer_groups'][0]['lr']=55
    assert b.preflight()['optimizer_groups'][0]['lr']==.001
    s.model.eval();s.model.color.train()
    next(s.model.parameters()).grad=torch.ones_like(next(s.model.parameters()))
    before=fingerprint(b)
    with pytest.raises(ValueError,match='admission|held-out'):b.step()
    assert fingerprint(b)==before


@pytest.mark.parametrize('change',['provider','weight','params','lr','model','steps','state'])
def test_live_drift_rejected_transactionally(tmp_path,cases,change):
    s=make(tmp_path,cases);b=boundary(s)
    if change=='provider':b.schedule=copy.copy(s)
    if change=='weight':s.weight=7.
    if change=='params':b.optimizer.param_groups[0]['params'].pop()
    if change=='lr':b.optimizer.param_groups[0]['lr']=.5
    if change=='model':b.model=copy.deepcopy(b.model)
    if change=='steps':b.steps=True
    if change=='state':b.optimizer.state[next(b.model.parameters())]['step']=torch.tensor(1.)
    before=fingerprint(b)
    with pytest.raises(ValueError,match='binding|ownership|step|optimizer'):b.step()
    assert fingerprint(b)==before


@pytest.mark.parametrize('lr',[True,0.,-1.,float('nan'),float('inf')])
def test_invalid_learning_rate_rejected(tmp_path,cases,lr):
    s=make(tmp_path,cases)
    with pytest.raises(ValueError,match='learning rate'):boundary(s,learning_rate=lr)
