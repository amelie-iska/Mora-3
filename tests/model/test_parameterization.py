"""Explicit parameter ownership and finite-axis identity; no inferred muP rules."""
import dataclasses
import pytest
import torch
from torch import nn


def api():
    from mora3.model.parameterization import Axis, ParameterRole, parameter_ledger
    return Axis, ParameterRole, parameter_ledger


def fixture(device='cpu'):
    A,R,L=api()
    m=nn.Module();m.embedding=nn.Embedding(7,3,device=device)
    m.readout=nn.Linear(3,7,bias=False,device=device);m.readout.weight=m.embedding.weight
    m.hidden=nn.Linear(3,5,device=device);m.logZ=nn.Parameter(torch.zeros((),device=device))
    finite=A('finite','a'*64);width=A('width','hidden')
    tied=R('embedding_readout',(finite,width))
    roles={'logZ':R('scalar',()),'embedding.weight':tied,'readout.weight':tied,
           'hidden.weight':R('hidden',(A('width','hidden2'),width)),
           'hidden.bias':R('vector',(A('width','hidden2'),))}
    return m,roles,L


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_unique_counts_aliases_axes_and_read_only(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('no CUDA')
    m,r,L=fixture(device);before={n:p.detach().clone() for n,p in m.named_parameters()}
    rng=torch.random.get_rng_state().clone();m.hidden.eval()
    ledger=L(m,r)
    assert ledger.total_numel==42 and ledger.trainable_numel==42
    assert len(ledger.parameters)==4
    tied=next(p for p in ledger.parameters if 'readout.weight' in p.aliases)
    assert tied.aliases==('embedding.weight','readout.weight') and tied.shape==(7,3)
    assert tied.axes[0].kind=='finite' and tied.axes[1].kind=='width'
    assert torch.equal(rng,torch.random.get_rng_state()) and not m.hidden.training
    for n,p in m.named_parameters():assert torch.equal(before[n],p) and p.grad is None
    with pytest.raises(dataclasses.FrozenInstanceError):ledger.total_numel=0
    r.clear();assert tied.axes[0].identity=='a'*64


@pytest.mark.parametrize('failure',['missing','extra','alias','rank','invalid-role','unknown-axis','bad-hash','list-axes'])
def test_ambiguous_declarations_refuse_without_mutation(failure):
    A,R,L=api();m,r,_=fixture();before=m.embedding.weight.clone()
    with pytest.raises(ValueError):
        if failure=='missing':del r['logZ']
        if failure=='extra':r['absent']=r['logZ']
        if failure=='alias':r['readout.weight']=R('output',r['readout.weight'].axes)
        if failure=='rank':r['logZ']=R('scalar',(A('width','h'),))
        if failure=='invalid-role':r['logZ']=R('guess',())
        if failure=='unknown-axis':A('maybe','h')
        if failure=='bad-hash':A('finite','genes')
        if failure=='list-axes':R('scalar',[])
        L(m,r)
    assert torch.equal(before,m.embedding.weight)


def test_frozen_parameters_and_equal_but_untied_values():
    m,r,L=fixture();m.readout.weight=nn.Parameter(m.embedding.weight.detach().clone(),requires_grad=False)
    result=L(m,r)
    assert result.total_numel==63 and result.trainable_numel==42 and len(result.parameters)==5


def test_custom_rank_three_explicitly_classified_and_finite_identity_retained():
    A,R,L=api();m=nn.Module();m.weight=nn.Parameter(torch.zeros(2,3,4))
    with pytest.raises(ValueError):L(m,{})
    role=R('custom',(A('finite','b'*64),A('width','a'),A('width','b')))
    row=L(m,{'weight':role}).parameters[0]
    assert row.numel==24 and row.axes==role.axes
