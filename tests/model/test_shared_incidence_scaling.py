"""Registered wrapper aliases must preserve exact directed incidence semantics."""
import dataclasses
import pytest
import torch
from torch import nn
from test_incidence_scaling import family, FEATURES, IDS

ORDER=tuple(s+':'+n for s in ('left','right') for n in IDS)

def api():
    from mora3.model import incidence_scaling as m
    assert hasattr(m,'shared_incidence_scaling_contract'), 'missing shared owner binding'
    return m

def wrappers(device='cpu', dtype=torch.float64):
    roots=[]
    for model in family(device,dtype):
        root=nn.Module();root.graph=model
        root.branch=nn.Module();root.branch.endpoint=model.incidence
        root.also=model
        roots.append(root)
    return tuple(roots)

def declarations(m):
    return tuple(m.IncidenceCallsite(p,IDS,ORDER) for p in ('graph.incidence','branch.endpoint','also.incidence'))

def bind(m,roots,**kw):
    options=dict(owner_paths=('graph',)*3,feature_axis=FEATURES,identifier_axis=IDS,
                 callsites=tuple(declarations(m) for _ in roots))
    options.update(kw)
    return m.shared_incidence_scaling_contract(*roots,**options)

@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_shared_bindings_count_once_preserve_outputs_gradients_state_rng(device,dtype):
    m=api()
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(36701);roots=wrappers(device,dtype)
    state=[{n:t.clone() for n,t in r.state_dict().items()} for r in roots]
    modes=[tuple(v.training for v in r.modules()) for r in roots]
    cpu=torch.random.get_rng_state().clone()
    cuda=torch.cuda.get_rng_state().clone() if device=='cuda' else None
    result=bind(m,roots)
    assert result.schema=='mora3.shared-native-incidence-input-scaling.v1'
    assert result.native.widths==(8,12,16)
    assert result.unique_numel==(48,72,96)
    assert result.aliases==(tuple(sorted(p+'.weight' for p in ('graph.incidence','branch.endpoint','also.incidence'))),)*3
    assert result.callsites==tuple(tuple(sorted(declarations(m),key=lambda c:c.path)) for _ in roots)
    with pytest.raises(dataclasses.FrozenInstanceError):result.aliases=()
    assert torch.equal(cpu,torch.random.get_rng_state())
    if cuda is not None:assert torch.equal(cuda,torch.cuda.get_rng_state())
    for root,s,mode in zip(roots,state,modes):
        assert mode==tuple(v.training for v in root.modules())
        for n,v in root.state_dict().items():assert torch.equal(v,s[n])
        assert all(p.grad is None for p in root.parameters())
        x=(torch.arange(12,device=device,dtype=dtype).reshape(2,6)/12).requires_grad_()
        # Both registered callsites use the same physical Parameter, once in inventory.
        actual=root.branch.endpoint(x)+root.also.incidence(2*x)
        w=root.graph.incidence.weight.detach().clone().requires_grad_()
        xr=x.detach().clone().requires_grad_()
        expected=3*xr@w.T
        torch.testing.assert_close(actual,expected,rtol=2e-5,atol=1e-6)
        weights=torch.linspace(.1,1,actual.numel(),device=device,dtype=dtype).reshape_as(actual)
        ag=torch.autograd.grad((actual*weights).sum(),(x,root.graph.incidence.weight))
        rg=torch.autograd.grad((expected*weights).sum(),(xr,w))
        for a,b in zip(ag,rg):torch.testing.assert_close(a,b,rtol=2e-5,atol=1e-6)

@pytest.mark.parametrize('bad',['missing','extra','duplicate','reordered-axis','reversed-endpoints','role','different-module','weight-only-tie','raw-alias','owner','owner-count','list','mutable-axis','family-shared','dtype'])
def test_incomplete_or_conflicting_owner_bindings_refuse_without_mutation(bad):
    m=api();roots=list(wrappers());ds=[declarations(m) for _ in roots];kw={}
    c=ds[1][1]
    if bad=='missing':ds[1]=ds[1][:-1]
    if bad=='extra':ds[1]+= (dataclasses.replace(c,path='absent'),)
    if bad=='duplicate':ds[1]+= (c,)
    if bad=='reordered-axis':ds[1]=(dataclasses.replace(ds[1][0],identifier_axis=IDS[::-1]),)+ds[1][1:]
    if bad=='reversed-endpoints':ds[1]=(dataclasses.replace(ds[1][0],ordered_endpoint_axis=ORDER[::-1]),)+ds[1][1:]
    if bad=='role':ds[1]=(dataclasses.replace(ds[1][0],role='hidden'),)+ds[1][1:]
    if bad=='different-module':roots[1].branch.endpoint=nn.Linear(6,12,bias=False).double()
    if bad=='weight-only-tie':
        roots[1].branch.endpoint=nn.Linear(6,12,bias=False).double()
        roots[1].branch.endpoint.weight=roots[1].graph.incidence.weight
    if bad=='raw-alias':roots[1].raw=roots[1].graph.incidence.weight
    if bad=='owner':kw['owner_paths']=('graph','branch','graph')
    if bad=='owner-count':kw['owner_paths']=('graph',)
    if bad=='list':ds=list(ds);kw['callsites']=ds
    else:kw['callsites']=tuple(ds)
    if bad=='mutable-axis':kw['identifier_axis']=list(IDS)
    if bad=='family-shared':roots[1]=roots[0]
    if bad=='dtype':roots[1].float()
    state=[{n:t.clone() for n,t in r.state_dict().items()} for r in roots]
    rng=torch.random.get_rng_state().clone()
    with pytest.raises(ValueError):bind(m,roots,**kw)
    assert torch.equal(rng,torch.random.get_rng_state())
    for root,s in zip(roots,state):
        for n,v in root.state_dict().items():assert torch.equal(v,s[n])
