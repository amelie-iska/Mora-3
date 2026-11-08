"""Synthetic interface references only: no source admission or optimizer updates."""
import pytest
import torch
from test_admet_model import setup


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('kind',['regression','binary'])
def test_native_head_objective_all_gradient_reference(device,kind):
    model,v,c,_,_=setup(device,kind)
    outputs=model((v,),(c,))
    p=outputs[0];one=torch.ones(1,dtype=torch.bool,device=device)
    masks=dict(observed=one,applicable=one,applicability_known=one)
    target=torch.tensor([1.0],device=device,dtype=torch.float64)
    if kind=='binary':
        targets=dict(target=target,**masks)
        ref=torch.nn.functional.binary_cross_entropy_with_logits(p['logit'].reshape(1),target)
    else:
        targets=dict(relation=('equal',),value=target,lower=target,upper=target,**masks)
        ref=(.5*((target-p['mean'])/p['scale']).square()+p['scale'].log()+.5*torch.log(target.new_tensor(2*torch.pi))).mean()
    got=model.reference_objective(outputs,(v,),(c,),native_unit=c.unit,**targets)
    torch.testing.assert_close(got,ref,rtol=1e-12,atol=1e-12)
    params=tuple(model.parameters())
    a=torch.autograd.grad(got,params,retain_graph=True,allow_unused=True)
    b=torch.autograd.grad(ref,params,allow_unused=True)
    assert sum(x is not None for x in a)>10
    for x,y in zip(a,b):
        assert (x is None)==(y is None)
        if x is not None:torch.testing.assert_close(x,y,rtol=1e-12,atol=1e-12)
    with pytest.raises(ValueError,match='admission'):model.observed_loss(outputs[0],v,c,source_admission=True)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('relation',['left','right','interval'])
def test_censored_head_and_desired_are_separate(device,relation):
    model,v,c,_,_=setup(device)
    outputs=model((v,),(c,));p=outputs[0]
    lo=p['mean'].detach().reshape(1)-.7;hi=lo+1.2
    one=torch.ones(1,dtype=torch.bool,device=device)
    targets=dict(relation=(relation,),value=lo*float('nan'),lower=lo,upper=hi,
                 observed=one,applicable=one,applicability_known=one)
    got=model.reference_objective(outputs,(v,),(c,),native_unit=c.unit,**targets)
    normal=torch.distributions.Normal(p['mean'],p['scale'])
    ref={'left':-normal.cdf(hi).log(),'right':-normal.cdf(-lo+2*p['mean']).log(),
         'interval':-(normal.cdf(hi)-normal.cdf(lo)).log()}[relation].mean()
    torch.testing.assert_close(got,ref,rtol=1e-12,atol=1e-12)
    desired=model.desired_objective(outputs,(v,),(c,),native_unit=c.unit,lower=lo,upper=lo,active=one)
    torch.testing.assert_close(desired,(p['mean']-lo).square().mean())
    zero=model.reference_objective(outputs,(v,),(c,),native_unit=c.unit,**dict(targets,observed=~one))
    assert zero==0
    assert all(torch.equal(x,torch.zeros_like(x)) for x in torch.autograd.grad(zero,(p['mean'],p['scale']),retain_graph=True))


@pytest.mark.parametrize('bad',['unit','dtype','shape','output','partition','desired_binary'])
def test_interface_rejects_ambiguous_or_mixed_coordinates(bad):
    model,v,c,_,_=setup('cpu')
    outputs=model((v,),(c,));t=torch.ones(1,dtype=torch.float64);mask=t.bool()
    targets=dict(relation=('equal',),value=t,lower=t,upper=t,observed=mask,applicable=mask,applicability_known=mask)
    unit=c.unit
    if bad=='unit':unit='invented'
    elif bad=='dtype':targets['value']=t.float()
    elif bad=='shape':targets['observed']=mask.reshape(1,1)
    elif bad=='output':outputs=({'mean':outputs[0]['mean'],'scale':outputs[0]['scale'],'hidden_target':t},)
    elif bad=='partition':
        from mora3.model.admet_objectives import AdmetObjectiveContract
        from mora3.data.admet import AssayContext
        other=AdmetObjectiveContract(c.definition,AssayContext.from_mapping({'species':'other'}),'peptide')
        with pytest.raises(ValueError):model.reference_objective(outputs,(v,),(other,),native_unit=unit,**targets)
        return
    else:
        model,v,c,_,_=setup('cpu','binary');outputs=model((v,),(c,))
        with pytest.raises(ValueError):model.desired_objective(outputs,(v,),(c,),native_unit=c.unit,lower=t,upper=t,active=mask)
        return
    with pytest.raises(ValueError):model.reference_objective(outputs,(v,),(c,),native_unit=unit,**targets)
