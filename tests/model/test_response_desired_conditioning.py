"""Catch omitted request channels, context drift, target leakage and wrong gradients."""
import copy
from dataclasses import replace
import pytest
import torch
from mora3.model.sciplex import SciPlexSharedModel
from mora3.model.response_sampling import ResponseContext
from mora3.model.response_desired import DesiredExpression, DesiredExpressionOrdinal


def case(device='cpu'):
    from mora3.model.response_desired_flow import DesiredResponseFlow
    torch.manual_seed(26201)
    c=ResponseContext(('A','B','B'),'a'*64,'b'*64,'Dex',1.,24.)
    h=DesiredResponseFlow(SciPlexSharedModel(c.axis_ids,width=4)).to(device).double()
    d=DesiredExpressionOrdinal(c,((1,'B',0.,2.),(2,'B',None,None)))
    return h,c,d


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_explicit_masks_equation_and_all_gradients(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    h,c,d=case(device);ref=copy.deepcopy(h)
    # channel-major: requested, available, lower, upper. Zero bound != missing.
    z=torch.tensor([[0,1,1,0,1,0,0,0,0,0,2,0]],device=device,dtype=torch.float64)
    assert torch.equal(h.encode_requests([c],[d]),z)
    x=z.new_tensor([[.3,-.2,.8]],requires_grad=True);t=z.new_tensor([.4])
    a=h([c],x,t,requests=[d])
    context=ref.native.backbone([c.graph(ref.native)])[0][:,0]
    hidden=context+torch.nn.functional.linear(x,ref.flow.state.weight,ref.flow.state.bias)
    hidden=hidden+torch.nn.functional.linear(t[:,None],ref.flow.time.weight,ref.flow.time.bias)
    hidden=hidden+torch.nn.functional.linear(z,ref.request.weight,None)
    denoised=torch.nn.functional.linear(torch.nn.functional.silu(hidden),ref.flow.output.weight,ref.flow.output.bias)
    expected=(denoised-.99*x)/(1-.99*t[:,None])
    torch.testing.assert_close(a,expected,rtol=1e-13,atol=1e-13)
    a.square().sum().backward();expected.square().sum().backward()
    for (name,p),(other,q) in zip(h.named_parameters(),ref.named_parameters()):
        assert name==other
        assert (p.grad is None)==(q.grad is None)
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=1e-12,atol=1e-12)
    assert h.request.weight.grad.abs().sum()>0
    assert h.native.expression.weight.grad is None
    assert not torch.equal(a,h([c],x,t,requests=[replace(d,intervals=((1,'B',1.,3.),))]))
    assert not torch.equal(a,h([c],x,t,requests=[replace(d,intervals=((1,'B',0.,2.),))]))
    # Order of explicit requests is irrelevant, axis ordinal is not.
    assert torch.equal(a,h([c],x,t,requests=[replace(d,intervals=d.intervals[::-1])]))


@pytest.mark.parametrize('field,value',[
    ('source_sha256','c'*64),('audit_sha256','d'*64),('intervention','SAHA'),
    ('dose_coordinate',2.),('exposure_hours',48.),('axis_ids',('B','A','B'))])
def test_context_mismatch_refuses(field,value):
    h,c,d=case()
    with pytest.raises(ValueError):h.encode_requests([replace(c,**{field:value})],[d])


def test_name_ordinal_equivalence_and_no_hidden_targets():
    h,c,d=case();name=DesiredExpression(c,(('A',0.,1.),))
    ordinal=DesiredExpressionOrdinal(c,((0,'A',0.,1.),))
    assert torch.equal(h.encode_requests([c],[name]),h.encode_requests([c],[ordinal]))
    from test_sciplex_shared import task
    with pytest.raises(ValueError):h.encode_requests([task(counts=(999,888),cell='hidden')],[name])
    x=torch.ones(1,3,dtype=torch.float64);t=x.new_tensor([.5])
    with pytest.raises(ValueError):h([c],x,t,requests=None)
    with pytest.raises(TypeError):h([c],x,t,requests=[name],measured_response=x)


@pytest.mark.parametrize('kind',['empty','wrong-count','bad-request','bad-context','nan','time','shape','dtype','overflow'])
def test_invalid_inputs_refuse(kind):
    h,c,d=case();cs=[c];ds=[d];x=torch.ones(1,3,dtype=torch.float64);t=x.new_tensor([.5])
    if kind=='empty':cs=[];ds=[]
    if kind=='wrong-count':ds=[d,d]
    if kind=='bad-request':ds=[{'target':1}]
    if kind=='bad-context':cs=[object()]
    if kind=='nan':x[0,0]=float('nan')
    if kind=='time':t+=1
    if kind=='shape':x=x[:,:1]
    if kind=='dtype':x=x.float()
    if kind=='overflow':
        h=h.float();x=x.float();t=t.float()
        ds=[DesiredExpression(c,(('A',0.,1e100),))]
    with pytest.raises(ValueError):h(cs,x,t,requests=ds)


def test_strict_versioned_state_and_existing_sampler_refusal():
    from mora3.model.response_flow import ResponseFlowDenoiserV2
    from mora3.model.response_sampling import sample_response
    h,c,d=case();twin=copy.deepcopy(h);twin.load_state_dict(h.state_dict())
    bad=copy.deepcopy(h.state_dict());bad['_extra_state']['encoding']='unknown'
    with pytest.raises(ValueError):twin.load_state_dict(bad)
    with pytest.raises(ValueError):sample_response(h,[c],steps=1,seed=1,sigma_min=.01,negative_policy='preserve')
    with pytest.raises(ValueError):h.load_state_dict(ResponseFlowDenoiserV2(SciPlexSharedModel(c.axis_ids,width=4)).double().state_dict())
    assert ResponseFlowDenoiserV2(SciPlexSharedModel(c.axis_ids,width=4)).config['version']==2
