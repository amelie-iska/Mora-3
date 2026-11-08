import copy
import pytest
import torch
from test_goa_objective import view
from mora3.model import goa_training as training


def make(tmp_path,device='cpu'):
    v=view(tmp_path)
    model=training.NativeGOAModel([v],width=4,heads=2,layers=1).to(device)
    return training.NativeGOATrainer(model,[v],lr=.001)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_continuation_matches_all_parameters_and_adam(tmp_path,device):
    t=make(tmp_path,device);t.step();state=t.state_dict()
    expected=t.step();parameters=copy.deepcopy(t.model.state_dict())
    t.load_state_dict(state)
    assert t.step()==expected
    for k,v in parameters.items():
        if isinstance(v,torch.Tensor):assert torch.equal(v,t.model.state_dict()[k])
    assert t.steps==2


@pytest.mark.parametrize('bad',['moment','variance','step','weight','binding','rng','missing_slot'])
def test_rejection_is_transactional(tmp_path,bad):
    t=make(tmp_path);t.step();before=t.state_dict();s=copy.deepcopy(before)
    slot=next(iter(s['optimizer']['state'].values()))
    if bad=='moment':slot['exp_avg'].fill_(float('nan'))
    if bad=='variance':slot['exp_avg_sq'].fill_(-1)
    if bad=='step':slot['step'].fill_(100)
    if bad=='weight':next(x for x in s['model'].values() if isinstance(x,torch.Tensor)).fill_(float('inf'))
    if bad=='binding':s['binding']='stale'
    if bad=='missing_slot':s['optimizer']['state'].pop(next(iter(s['optimizer']['state'])))
    if bad=='rng':s['cpu_rng']=torch.zeros(3,dtype=torch.uint8)
    with pytest.raises((ValueError,RuntimeError)):t.load_state_dict(s)
    after=t.state_dict()
    assert t.steps==before['steps'] and torch.equal(before['cpu_rng'],after['cpu_rng'])
    for k,x in before['model'].items():
        if isinstance(x,torch.Tensor):assert torch.equal(x,after['model'][k])
    for pid,slots in before['optimizer']['state'].items():
        for k,x in slots.items():assert torch.equal(x,after['optimizer']['state'][pid][k])


def test_runtime_policy_change_rejects_before_update(tmp_path):
    t=make(tmp_path);old=torch.get_num_threads()
    try:
        torch.set_num_threads(old+1)
        with pytest.raises(ValueError,match='runtime'):t.step()
        assert t.steps==0
    finally:torch.set_num_threads(old)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_every_shared_gradient_matches_signed_softplus(tmp_path,device):
    t=make(tmp_path,device);t.model.double()
    z=t.model(t.packets)
    actual=t.model['head'].objective(z,t.packets)
    y=z.new_tensor([v.labels for v in t.packets])
    mask=torch.tensor([v.observed for v in t.packets],device=device,dtype=torch.bool)
    expected=torch.nn.functional.softplus(-(2*y[mask]-1)*z[mask]).mean()
    parameters=tuple(t.model.parameters())
    ga=torch.autograd.grad(actual,parameters,retain_graph=True)
    gb=torch.autograd.grad(expected,parameters)
    assert len(ga)==len(parameters)
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-14)


@pytest.mark.parametrize('setting',['pooling','max_tokens'])
def test_changed_payload_policy_rejects_before_update(tmp_path,setting):
    t=make(tmp_path)
    setattr(t.model['payload'],setting,'bag' if setting=='pooling' else 4096)
    with pytest.raises(ValueError,match='payload'):t.step()
    assert t.steps==0
