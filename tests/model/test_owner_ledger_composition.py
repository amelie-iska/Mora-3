import dataclasses
import pytest
import torch
from torch import nn
from mora3.model import parameterization as p


def fixture():
    model=nn.Module()
    model.left=nn.Linear(3,4)
    model.right=nn.Linear(3,4)
    model.right.weight=model.left.weight
    roles={'weight':p.ParameterRole('hidden',(p.Axis('width','out'),p.Axis('width','in'))),
           'bias':p.ParameterRole('vector',(p.Axis('width','out'),))}
    return model,{n:p.parameter_ledger(c,roles) for n,c in model.named_children()}


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_shared_complete_counts_readonly(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    m,ls=fixture();m.to(device)
    before={n:t.clone() for n,t in m.named_parameters()};rng=torch.get_rng_state().clone()
    result=p.compose_owner_parameter_ledgers(m,ls)
    assert result.total_numel==12+4+4
    assert next(e for e in result.parameters if e.shape==(4,3)).aliases==('left.weight','right.weight')
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(t,before[n]) and t.grad is None for n,t in m.named_parameters())


@pytest.mark.parametrize('drift',['missing','extra','shape','dtype','trainable','alias','count','schema','conflict','root'])
def test_incomplete_stale_or_conflicting_refused(drift):
    m,ls=fixture()
    if drift=='missing':ls.pop('right')
    elif drift=='extra':ls['unknown']=ls['left']
    elif drift=='shape':m.left.bias=nn.Parameter(torch.zeros(5))
    elif drift=='dtype':m.left.bias=nn.Parameter(m.left.bias.double())
    elif drift=='trainable':m.left.bias.requires_grad_(False)
    elif drift=='root':m.register_parameter('new',nn.Parameter(torch.ones(1)))
    else:
        ledger=ls['right'];index=next(i for i,e in enumerate(ledger.parameters) if 'weight' in e.aliases);entry=ledger.parameters[index]
        if drift=='alias':entry=dataclasses.replace(entry,aliases=('unknown',))
        elif drift=='conflict':entry=dataclasses.replace(entry,role='custom')
        elif drift=='count':ls['right']=dataclasses.replace(ledger,total_numel=999)
        elif drift=='schema':ls['right']=dataclasses.replace(ledger,schema='unknown')
        if drift in ('alias','conflict'):ls['right']=dataclasses.replace(ledger,parameters=ledger.parameters[:index]+(entry,)+ledger.parameters[index+1:])
    with pytest.raises(ValueError):p.compose_owner_parameter_ledgers(m,ls)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_native_conditioned_owner_shared_complete_composition(device):
    from test_conditioned_owner_parameterization import owner
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(28401)
    m=nn.Module();m.first=owner().to(device);m.second=m.first
    ledger=p.conditioned_instance_parameter_ledger(m.first,channel_axis=('r','g','b'),identifier_axis=('a','b'))
    result=p.compose_owner_parameter_ledgers(m,{'first':ledger,'second':ledger})
    assert result.total_numel==ledger.total_numel==sum(t.numel() for t in m.parameters())
    assert len(result.parameters)==len(ledger.parameters)
    assert all(any(a.startswith('first.') for a in e.aliases) and
               any(a.startswith('second.') for a in e.aliases) for e in result.parameters)
