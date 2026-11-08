"""Explicit request/supervision separation and independent CFM arithmetic."""
import copy
from dataclasses import replace
import pytest
import torch
from test_sciplex_shared import task
from mora3.model.response_flow_trainer import ResponseFlowBatch
from mora3.model.response_sampling import ResponseContext
from mora3.model.response_desired import DesiredExpressionOrdinal
from mora3.model.response_desired_flow import DesiredResponseFlow
from mora3.model.sciplex import SciPlexSharedModel


def case(device='cpu'):
    torch.manual_seed(26501)
    h=DesiredResponseFlow(SciPlexSharedModel(('A','B'),width=4)).to(device).double()
    b=ResponseFlowBatch((task(),task(counts=(3,7),cell='b')))
    cs=tuple(ResponseContext.from_task(t) for t in b.tasks)
    ds=tuple(DesiredExpressionOrdinal(c,((0,'A',0.,1.),(1,'B',None,None))) for c in cs)
    return h,b,cs,ds


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_path_loss_and_all_gradients(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from mora3.model.response_desired_objective import desired_response_cfm_loss
    h,b,cs,ds=case(device);ref=copy.deepcopy(h)
    noise=h.native.edge.new_tensor([[.1,-.2],[.3,.4]])
    time=noise.new_tensor([0.,1.])
    rng=torch.get_rng_state().clone()
    result=desired_response_cfm_loss(h,b,requests=ds,noise=noise,time=time,sigma_min=.01)
    y=torch.log1p(noise.new_tensor([[1,2],[3,7]]))
    x=(1-.99*time[:,None])*noise+time[:,None]*y
    u=y-.99*noise
    z=noise.new_tensor([[1,1,1,0,0,0,1,0]]).expand(2,-1)
    c=ref.native.backbone([v.graph(ref.native) for v in cs])[0][:,0]
    f=torch.nn.functional
    hidden=c+f.linear(z,ref.request.weight)+f.linear(x,ref.flow.state.weight,ref.flow.state.bias)+f.linear(time[:,None],ref.flow.time.weight,ref.flow.time.bias)
    den=f.linear(f.silu(hidden),ref.flow.output.weight,ref.flow.output.bias)
    expected=(((den-.99*x)/(1-.99*time[:,None])-u)**2).mean()
    torch.testing.assert_close(result['loss'],expected,rtol=1e-12,atol=1e-12)
    result['loss'].backward();expected.backward()
    for (n,p),(m,q) in zip(h.named_parameters(),ref.named_parameters()):
        assert n==m and (p.grad is None)==(q.grad is None)
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=1e-11,atol=1e-11)
    assert h.request.weight.grad.abs().sum()>0
    assert h.native.expression.weight.grad is None
    assert torch.equal(rng,torch.get_rng_state())
    assert result['configuration']['request_role']=='externally-supplied-desired-not-observed'
    assert result['configuration']['supervision_role']=='source-observed-response-only'


def test_hidden_targets_do_not_enter_requests_or_graph_but_bind_loss():
    from mora3.model.response_desired_objective import desired_response_cfm_loss
    h,b,cs,ds=case();changed=ResponseFlowBatch((task(counts=(90,80),cell='hidden'),b.tasks[1]))
    assert cs==tuple(ResponseContext.from_task(t) for t in changed.tasks)
    noise=torch.ones(2,2,dtype=torch.float64);time=noise.new_full((2,),.3)
    a=desired_response_cfm_loss(h,b,requests=ds,noise=noise,time=time,sigma_min=.01)
    z=desired_response_cfm_loss(h,changed,requests=ds,noise=noise,time=time,sigma_min=.01)
    assert a['binding']!=z['binding'] and not torch.equal(a['loss'],z['loss'])
    # The API cannot establish independence of caller-supplied requests; it never derives them.
    assert a['configuration']['requests']==z['configuration']['requests']
    a['configuration']['model']['version']=99
    assert h.config['version']==1
    with pytest.raises(TypeError):desired_response_cfm_loss(h,b,requests=ds,noise=noise,time=time,sigma_min=.01,measured_context=noise)


@pytest.mark.parametrize('kind',['sigma','bool-sigma','request-source','request-axis','request-count','request-none','axis','model','batch','nan','noise-shape','time','dtype'])
def test_invalid_binding_or_path_refuses_without_state_change(kind):
    from mora3.model.response_desired_objective import desired_response_cfm_loss
    h,b,cs,ds=case();noise=torch.ones(2,2,dtype=torch.float64);time=noise.new_full((2,),.3);sigma=.01
    if kind=='sigma':sigma=.1
    if kind=='bool-sigma':sigma=True
    if kind=='request-source':ds=(replace(ds[0],context=replace(cs[0],source_sha256='c'*64)),ds[1])
    if kind=='request-axis':ds=(replace(ds[0],context=replace(cs[0],axis_ids=('B','A')),intervals=((0,'B',0.,1.),)),ds[1])
    if kind=='request-count':ds=ds[:1]
    if kind=='request-none':ds=None
    if kind=='axis':h=DesiredResponseFlow(SciPlexSharedModel(('B','A'),width=4)).double()
    if kind=='model':h=h.flow
    if kind=='batch':b=b.tasks
    if kind=='nan':noise[0,0]=float('nan')
    if kind=='noise-shape':noise=noise[:,:1]
    if kind=='time':time+=1
    if kind=='dtype':noise=noise.float();time=time.float()
    before=copy.deepcopy(h.state_dict());rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):desired_response_cfm_loss(h,b,requests=ds,noise=noise,time=time,sigma_min=sigma)
    assert torch.equal(rng,torch.get_rng_state())
    for k,v in before.items():
        if isinstance(v,torch.Tensor):assert torch.equal(v,h.state_dict()[k])
