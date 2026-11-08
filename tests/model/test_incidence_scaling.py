"""Fixed ordered incidence is a one-scalable-axis input, not a hidden matrix."""
import dataclasses
import math
import pytest
import torch
from mora3.model import parameterization as inventory
from mora3.model.tokengt import TokenGT

FEATURES=('f0','f1','f2')
IDS=('q0','q1','q2')

def api():
    import importlib.util
    assert importlib.util.find_spec('mora3.model.incidence_scaling') is not None, 'missing scaling contract'
    import mora3.model.incidence_scaling as module
    assert hasattr(module,'native_incidence_scaling_contract')
    return module.native_incidence_scaling_contract

def family(device='cpu',dtype=torch.float64,attention='vanilla'):
    return tuple(TokenGT(3,3,width=w,heads=2,layers=1,attention=attention).to(device=device,dtype=dtype) for w in (8,12,16))

@pytest.mark.parametrize('attention',['vanilla','tropical-code','tropical-blockwise'])
def test_fixed_fanin_recipe_and_reference_optimizer_multipliers(attention):
    build=api(); models=family(attention=attention)
    result=build(*models,feature_axis=FEATURES,identifier_axis=IDS)
    assert result.schema=='mora3.native-incidence-input-scaling.v1'
    assert result.widths==(8,12,16) and result.fan_in==6
    assert result.identifier_axis==IDS
    assert result.ordered_endpoint_axis==tuple(s+':'+n for s in ('left','right') for n in IDS)
    assert result.aliases==('incidence.weight',)
    assert result.init_std_multiplier==1.0
    assert result.uniform_bounds==(-1/math.sqrt(6),1/math.sqrt(6))
    assert result.normal_std==pytest.approx(1/math.sqrt(18))
    for width in (8,12,16):
        assert result.lr_multiplier(width,optimizer='adam')==1
        assert result.lr_multiplier(width,optimizer='adamw')==1
        assert result.lr_multiplier(width,optimizer='sgd')==width/8
    with pytest.raises(dataclasses.FrozenInstanceError): result.fan_in=1
    with pytest.raises(ValueError,match='custom'):
        inventory.width_family(*(inventory.tokengt_parameter_ledger(m,feature_axis=FEATURES,identifier_axis=IDS) for m in models))

@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('width',[8,12,16])
def test_native_embed_direction_and_all_input_weight_gradients_unchanged(device,dtype,width):
    build=api()
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(36601)
    models=family(device,dtype); m=models[(8,12,16).index(width)]
    before={n:p.detach().clone() for n,p in m.named_parameters()}
    cpu_rng=torch.random.get_rng_state().clone()
    cuda_rng=torch.cuda.get_rng_state().clone() if device=='cuda' else None
    modes=tuple(z.training for z in m.modules())
    plan=build(*models,feature_axis=FEATURES,identifier_axis=IDS)
    assert torch.equal(cpu_rng,torch.random.get_rng_state())
    if cuda_rng is not None:assert torch.equal(cuda_rng,torch.cuda.get_rng_state())
    assert modes==tuple(z.training for z in m.modules())
    for n,p in m.named_parameters():assert torch.equal(p,before[n]) and p.grad is None
    # Recipe sampled locally only in the fixture; contract never initializes live weights.
    gen=torch.Generator(device=device).manual_seed(36601)
    with torch.no_grad():m.incidence.weight.uniform_(*plan.uniform_bounds,generator=gen)
    x=torch.tensor([[1.,2.,3.],[2.,3.,4.],[4.,2.,1.],[1.,4.,2.]],device=device,dtype=dtype,requires_grad=True)
    p=torch.eye(3,device=device,dtype=dtype,requires_grad=True)
    e=torch.tensor([[0,0],[1,1],[2,2],[0,2]],device=device)
    r=torch.tensor([0,0,0,1],device=device)
    actual=m.embed(dict(features=x,identifiers=p,endpoints=e,roles=r))
    fw=m.feature.weight.detach().clone().requires_grad_(); iw=m.incidence.weight.detach().clone().requires_grad_(); rw=m.role.weight.detach().clone().requires_grad_()
    xr=x.detach().clone().requires_grad_(); pr=p.detach().clone().requires_grad_()
    reference=xr@fw.T + pr[e[:,0]]@iw[:,:3].T + pr[e[:,1]]@iw[:,3:].T + rw[r]
    torch.testing.assert_close(actual,reference,rtol=2e-5,atol=1e-6)
    weights=torch.linspace(.1,1,actual.numel(),device=device,dtype=dtype).reshape_as(actual)
    ag=torch.autograd.grad((actual*weights).sum(),(x,p,m.feature.weight,m.incidence.weight,m.role.weight))
    rg=torch.autograd.grad((reference*weights).sum(),(xr,pr,fw,iw,rw))
    for a,b in zip(ag,rg):torch.testing.assert_close(a,b,rtol=2e-5,atol=1e-6)
    reversed_e=e.clone();reversed_e[-1]=e[-1].flip(0)
    reverse=m.embed(dict(features=x,identifiers=p,endpoints=reversed_e,roles=r))
    assert not torch.allclose(actual[-1],reverse[-1])
    torch.testing.assert_close(actual[-1]-reverse[-1],(p[0]-p[2])@(iw[:,:3]-iw[:,3:]).T,rtol=2e-5,atol=1e-6)

@pytest.mark.parametrize('bad',['identifier-list','identifier-duplicate','identifier-short','feature-drift','delta-shrink','target-shrink','dtype-drift','bias','alias','subclass','optimizer','unknown-width','bool-width'])
def test_refusals_do_not_mutate(bad):
    build=api(); models=list(family());ids=IDS;features=FEATURES
    if bad=='identifier-list':ids=list(IDS)
    if bad=='identifier-duplicate':ids=('q0','q0','q2')
    if bad=='identifier-short':ids=IDS[:2]
    if bad=='feature-drift':features=('f0','f1')
    if bad=='delta-shrink':models[1]=family()[0]
    if bad=='target-shrink':models[2]=family()[0]
    if bad=='dtype-drift':models[1]=models[1].float()
    if bad=='bias':models[1].incidence=torch.nn.Linear(6,12,bias=True).double()
    if bad=='alias':models[1].unexpected=models[1].incidence
    if bad=='subclass':
        class Custom(TokenGT):pass
        models[1]=Custom(3,3,width=12,heads=2,layers=1).double()
    before=[{n:p.detach().clone() for n,p in m.named_parameters()} for m in models]
    rng=torch.random.get_rng_state().clone()
    with pytest.raises(ValueError):
        result=build(*models,feature_axis=features,identifier_axis=ids)
        if bad=='optimizer':result.lr_multiplier(12,optimizer='guess')
        if bad=='unknown-width':result.lr_multiplier(20,optimizer='adam')
        if bad=='bool-width':result.lr_multiplier(True,optimizer='adam')
    assert torch.equal(rng,torch.random.get_rng_state())
    for m,state in zip(models,before):
        for n,p in m.named_parameters():assert torch.equal(p,state[n])
