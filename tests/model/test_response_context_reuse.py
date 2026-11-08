import copy
import pytest
import torch
from test_sciplex_shared import task
from mora3.model.sciplex import SciPlexSharedModel
from mora3.model.response_flow import ResponseFlowHead, ResponseFlowDenoiserV2
from mora3.model.response_sampling import ResponseContext, sample_response

@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('head',[ResponseFlowHead,ResponseFlowDenoiserV2])
def test_call_local_context_exact_euler_and_state(device,head):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(25601)
    h=head(SciPlexSharedModel(('A','B'),width=8)).to(device).double()
    h.train();h.native.backbone.eval()
    modes=[m.training for m in h.modules()]
    state=copy.deepcopy(h.state_dict())
    contexts=[ResponseContext.from_task(task()),ResponseContext.from_task(task(identity='SAHA'))]
    args=dict(steps=4,seed=25601,sigma_min=.01,negative_policy='preserve')
    rng=torch.random.get_rng_state().clone()
    cuda_rng=torch.cuda.get_rng_state().clone() if device=='cuda' else None
    calls=[]
    hook=h.native.backbone.register_forward_hook(lambda *a:calls.append(1))
    ref=sample_response(h,contexts,**args)
    assert len(calls)==4
    calls.clear()
    out=sample_response(h,contexts,reuse_context=True,**args)
    assert len(calls)==1
    assert torch.equal(ref['log_expression'],out['log_expression'])
    assert ref['configuration']==out['configuration'] and ref['binding']==out['binding']
    assert not out['log_expression'].requires_grad
    assert modes==[m.training for m in h.modules()]
    assert torch.equal(rng,torch.random.get_rng_state())
    if cuda_rng is not None:assert torch.equal(cuda_rng,torch.cuda.get_rng_state())
    assert all(torch.equal(v,h.state_dict()[k]) if isinstance(v,torch.Tensor) else v==h.state_dict()[k] for k,v in state.items())
    # A new call must recompute features after parameter/context changes.
    with torch.no_grad():h.native.context.bias.add_(.2)
    other=[ResponseContext.from_task(task(identity='SAHA'))]
    assert torch.equal(sample_response(h,other,**args)['log_expression'],sample_response(h,other,reuse_context=True,**args)['log_expression'])
    hook.remove()

@pytest.mark.parametrize('bad',[None,1,'yes',[],{}])
def test_reuse_policy_strict_before_forward(bad):
    h=ResponseFlowHead(SciPlexSharedModel(('A','B'),width=8))
    with pytest.raises(ValueError,match='reuse'):
        sample_response(h,[ResponseContext.from_task(task())],steps=4,seed=1,sigma_min=.01,negative_policy='preserve',reuse_context=bad)

def test_reuse_refuses_axis_and_restores_modes_on_failure():
    h=ResponseFlowHead(SciPlexSharedModel(('B','A'),width=8))
    args=dict(steps=4,seed=1,sigma_min=.01,negative_policy='preserve',reuse_context=True)
    with pytest.raises(ValueError,match='gene order'):sample_response(h,[ResponseContext.from_task(task())],**args)
    h=ResponseFlowHead(SciPlexSharedModel(('A','B'),width=8))
    modes=[m.training for m in h.modules()]
    def fail(*a):raise RuntimeError('injected')
    hook=h.native.backbone.register_forward_pre_hook(fail)
    with pytest.raises(RuntimeError,match='injected'):sample_response(h,[ResponseContext.from_task(task())],**args)
    assert modes==[m.training for m in h.modules()]
    hook.remove()
