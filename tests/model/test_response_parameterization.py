"""Explicit finite axes for local flow heads; native owner stays separate."""
import pytest
import torch
from torch import nn
from mora3.model.sciplex import SciPlexSharedModel
from mora3.model.response_flow import ResponseFlowHead, ResponseFlowDenoiserV2
from mora3.model.response_desired_flow import DesiredResponseFlow


def roles(model):
    from mora3.model.parameterization import response_head_parameter_roles
    return response_head_parameter_roles(model)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('kind',[ResponseFlowHead,ResponseFlowDenoiserV2,DesiredResponseFlow])
def test_local_map_order_orientation_and_readonly(kind,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(27101)
    m=kind(SciPlexSharedModel(('A','B','C'),width=8)).to(device)
    before={n:p.clone() for n,p in m.named_parameters()}
    rng=torch.random.get_rng_state().clone()
    r=roles(m);prefix='flow.' if kind is DesiredResponseFlow else ''
    excluded=prefix+'native.'
    assert set(r)=={n for n,_ in m.named_parameters(remove_duplicate=False) if not n.startswith(excluded)}
    assert r[prefix+'state.weight'].role=='input'
    assert r[prefix+'output.weight'].role=='output'
    assert r[prefix+'state.weight'].axes==tuple(reversed(r[prefix+'output.weight'].axes))
    assert r[prefix+'time.weight'].axes[1].kind=='finite'
    if kind is DesiredResponseFlow:
        assert r['request.weight'].role=='custom'
        assert r['request.weight'].axes[1].kind=='finite'
    assert torch.equal(rng,torch.random.get_rng_state())
    for n,p in m.named_parameters():assert torch.equal(p,before[n]) and p.grad is None
    other=kind(SciPlexSharedModel(('B','A','C'),width=8))
    assert roles(other)[prefix+'state.weight'].axes!=r[prefix+'state.weight'].axes
    wider=kind(SciPlexSharedModel(('A','B','C'),width=12))
    assert roles(wider)==r


@pytest.mark.parametrize('drift',['extra','shape','type','axis','duplicate','native-extra'])
def test_unknown_ownership_and_axis_refused(drift):
    m=DesiredResponseFlow(SciPlexSharedModel(('A','B','C'),width=8))
    if drift=='extra':m.register_parameter('extra',nn.Parameter(torch.ones(1)))
    if drift=='shape':m.request=nn.Linear(11,8,bias=False)
    if drift=='type':m.request=nn.Sequential(m.request)
    if drift=='axis':m.native.axis_ids=['A','B','C']
    if drift=='duplicate':m.native.axis_ids=('A','A','C')
    if drift=='native-extra':m.flow.extra=nn.Linear(8,8)
    with pytest.raises(ValueError):roles(m)


def test_wrong_root_refused():
    with pytest.raises(ValueError):roles(nn.Linear(3,8))
