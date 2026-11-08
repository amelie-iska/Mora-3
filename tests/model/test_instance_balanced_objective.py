"""Independent per-edge oracle catches pooled BCE and wrong boundary grouping."""
import pytest
import torch
from mora3.model import instance_affinity as affinity
from mora3.model.morphology_training import MorphologyTrainer
from test_morphology import task


@pytest.fixture(autouse=True)
def deterministic_runtime():
    previous=torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


def oracle(z, m):
    groups=[[],[],[]]
    for c,(dy,dx) in enumerate(((0,1),(1,0))):
        for y in range(m.shape[1]-dy):
            for x in range(m.shape[2]-dx):
                a=set(torch.where(m[:,y,x])[0].tolist())
                b=set(torch.where(m[:,y+dy,x+dx])[0].tolist())
                if len(a)>1 or len(b)>1: continue
                positive=bool(a & b)
                group=0 if positive else (1 if a or b else 2)
                groups[group].append(torch.nn.functional.softplus(-z[c,y,x] if positive else z[c,y,x]))
    means=[torch.stack(g).mean() for g in groups if g]
    return torch.stack(means).mean() if means else z.sum()*0


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_strata_equation_gradients_and_plane_order(device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    labels=torch.tensor([[1,1,2,0],[0,0,2,0],[0,0,0,0]],device=device)
    m=labels[None]==torch.tensor([1,2],device=device)[:,None,None]
    z=torch.linspace(-2,2,24,dtype=torch.float64,device=device).reshape(2,3,4).requires_grad_()
    expected=oracle(z,m)
    actual=affinity.balanced_affinity_objective(z,m)
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(torch.autograd.grad(actual,z)[0],torch.autograd.grad(expected,z)[0])
    torch.testing.assert_close(actual,affinity.balanced_affinity_objective(z,m.flip(0)))
    assert not torch.isclose(actual,affinity.affinity_objective(z,m))
    assert torch.autograd.gradcheck(lambda v: affinity.balanced_affinity_objective(v,m),(z,))


@pytest.mark.parametrize('shape',[(0,1,1),(0,2,3),(2,1,3)])
def test_absent_groups_and_overlap(shape):
    m=torch.zeros(shape,dtype=torch.bool)
    if shape[0]: m[:]=True
    z=torch.ones((2,*shape[1:]),requires_grad=True)
    actual=affinity.balanced_affinity_objective(z,m)
    torch.testing.assert_close(actual,oracle(z,m))
    actual.backward()
    assert torch.isfinite(z.grad).all()
    with pytest.raises(ValueError): affinity.balanced_affinity_objective(z*float('nan'),m)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_balanced_trainer_resume_and_legacy_config(device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.manual_seed(197)
    t=MorphologyTrainer([task()],width=8,device=device,affinity_mode='balanced-strata')
    t.step(); state=t.snapshot(); expected=t.step()
    r=MorphologyTrainer([task()],width=8,device=device,affinity_mode='balanced-strata')
    r.restore(state); assert r.step()==expected
    assert all(torch.equal(a,b) for a,b in zip(t.model.parameters(),r.model.parameters()))
    legacy=MorphologyTrainer([task()],width=8,device=device)
    assert legacy.config['objective']=='union-bce+affinity-bce'
    with pytest.raises(ValueError,match='config'): legacy.restore(state)
    with pytest.raises(ValueError): MorphologyTrainer([task()],affinity_mode='unknown')
    with pytest.raises(ValueError): MorphologyTrainer([task()],variant='foreground',affinity_mode='balanced-strata')
