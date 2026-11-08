"""Opt-in shared ownership, independent CFM arithmetic, and no trainer promotion."""
import copy
import pytest
import torch
from test_flow_unified import build
from test_response_desired_objective import case
from mora3.model.response_sampling import ResponseContext
from mora3.model.response_desired import DesiredExpressionOrdinal
from mora3.model.response_flow_trainer import ResponseFlowBatch


def cooperative(device='cpu'):
    from mora3.model.response_desired_cooperative import DesiredResponseCooperative
    torch.manual_seed(26601)
    parent=build(device).model.double()
    model=DesiredResponseCooperative(parent,sigma_min=.01)
    batch=ResponseFlowBatch(tuple(build().pools['response']))
    contexts=tuple(ResponseContext.from_task(t) for t in batch.tasks)
    requests=tuple(DesiredExpressionOrdinal(c,((0,'A',0.,1.),)) for c in contexts)
    return model,batch,contexts,requests


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_shared_loss_and_every_gradient(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    m,b,cs,ds=cooperative(device);r=copy.deepcopy(m)
    assert m.desired.native is m.parent.native.response
    assert m.desired.native.backbone is m.parent.annotation.backbone is m.parent.native.image.backbone
    assert m.desired.request.weight.dtype==torch.float64
    assert len(list(m.parameters()))==len({id(p) for p in m.parameters()})
    noise=m.desired.native.edge.new_tensor([[.1,.2],[-.3,.4]])
    t=noise.new_tensor([0.,1.]);before=copy.deepcopy(m.state_dict())
    result=m.desired_objective(b,requests=ds,noise=noise,time=t)
    y=torch.log1p(noise.new_tensor([[1,2],[3,7]]));x=(1-.99*t[:,None])*noise+t[:,None]*y
    z=noise.new_tensor([[1,0,1,0,0,0,1,0]]).expand(2,-1)
    h=r.desired;f=torch.nn.functional
    context=h.native.backbone([c.graph(h.native) for c in cs])[0][:,0]
    hidden=context+f.linear(z,h.request.weight)+f.linear(x,h.flow.state.weight,h.flow.state.bias)+f.linear(t[:,None],h.flow.time.weight,h.flow.time.bias)
    den=f.linear(f.silu(hidden),h.flow.output.weight,h.flow.output.bias)
    expected=(((den-.99*x)/(1-.99*t[:,None])-(y-.99*noise))**2).mean()
    torch.testing.assert_close(result['loss'],expected,rtol=1e-12,atol=1e-12)
    result['loss'].backward();expected.backward()
    for (n,p),(k,q) in zip(m.named_parameters(),r.named_parameters()):
        assert n==k and (p.grad is None)==(q.grad is None)
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=1e-11,atol=1e-11)
    assert m.parent.native.response.backbone.feature.weight.grad.abs().sum()>0
    assert m.desired.request.weight.grad.abs().sum()>0
    assert m.parent.flow.output.weight.grad is None
    assert m.parent.native.image.head.weight.grad is None
    assert m.parent.annotation.physical.unit.weight.grad is None
    for k,v in before.items():
        if isinstance(v,torch.Tensor):assert torch.equal(v,m.state_dict()[k])
    assert result['configuration']['training_admission'] is False
    assert not hasattr(m,'optimizer') and not hasattr(m,'step')


@pytest.mark.parametrize('drift',['response','image','annotation','parent-config','desired-config','sigma','flow-native'])
def test_live_drift_refuses_before_neural_call(drift):
    m,b,cs,ds=cooperative()
    if drift=='response':m.parent.native.response=copy.deepcopy(m.parent.native.response)
    if drift=='image':m.parent.native.image.backbone=copy.deepcopy(m.parent.native.image.backbone)
    if drift=='annotation':m.parent.annotation.backbone=copy.deepcopy(m.parent.annotation.backbone)
    if drift=='parent-config':m.parent.configuration['version']=9
    if drift=='desired-config':m.desired.config['version']=9
    if drift=='sigma':m.desired.flow.config['sigma_min']=.2
    if drift=='flow-native':m.parent.flow.native=copy.deepcopy(m.parent.flow.native)
    calls=[]
    hook=m.desired.register_forward_pre_hook(lambda *a:calls.append(1))
    with pytest.raises(ValueError):m.desired_objective(b,requests=ds,noise=torch.ones(2,2,dtype=torch.float64),time=torch.ones(2,dtype=torch.float64))
    hook.remove();assert calls==[]


def test_strict_checkpoint_and_legacy_trainer_refusal():
    from mora3.model.flow_unified import FlowUnifiedTrainer
    m,b,cs,ds=cooperative();state=copy.deepcopy(m.state_dict())
    fresh=cooperative()[0];fresh.load_state_dict(state)
    state['_extra_state']['version']=2
    with pytest.raises(ValueError):fresh.load_state_dict(state)
    with pytest.raises((RuntimeError,ValueError)):fresh.load_state_dict(m.parent.state_dict())
    tr=build()
    with pytest.raises(ValueError):FlowUnifiedTrainer(m,tr.batch,tr.pools,batch_size=1,flow_batch_size=1)
    result=m.desired_objective(b,requests=ds,noise=torch.ones(2,2,dtype=torch.float64),time=torch.zeros(2,dtype=torch.float64))
    result['configuration']['model']['version']=99
    assert m.get_extra_state()['version']==1


def test_wrong_parent_and_mismatched_request_refuse():
    from mora3.model.response_desired_cooperative import DesiredResponseCooperative
    from dataclasses import replace
    with pytest.raises(ValueError):DesiredResponseCooperative(case()[0])
    m,b,cs,ds=cooperative()
    ds=(replace(ds[0],context=replace(cs[0],source_sha256='f'*64)),ds[1])
    with pytest.raises(ValueError):m.desired_objective(b,requests=ds,noise=torch.ones(2,2,dtype=torch.float64),time=torch.zeros(2,dtype=torch.float64))
