import copy
import pytest
import torch
from test_sciplex_shared import task
from mora3.model.sciplex import SciPlexSharedModel
from mora3.model.response_flow import ResponseFlowHead, ResponseFlowDenoiserV2
from mora3.model.flow_matching import gaussian_conditional_velocity, velocity_loss

@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_equation_full_rank_contraction_and_target_exclusion(device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.manual_seed(6101)
    h = ResponseFlowDenoiserV2(SciPlexSharedModel(('A','B'), width=8), sigma_min=.01).to(device).double()
    x = torch.tensor([[.2,.3]], device=device, dtype=torch.float64, requires_grad=True)
    t = x.new_tensor([.7])
    y = x.new_tensor([[.4,.5]])
    with torch.no_grad():
        h.output.weight.zero_(); h.output.bias.copy_(y[0])
    v = h([task()], x, t)
    torch.testing.assert_close(v, gaussian_conditional_velocity(x,y,t), atol=1e-14, rtol=1e-14)
    j = torch.autograd.functional.jacobian(lambda z:h([task()],z,t), x).reshape(2,2)
    torch.testing.assert_close(j, -torch.eye(2,device=device,dtype=x.dtype)*.99/(1-.99*.7))
    assert torch.linalg.matrix_rank(j) == 2
    assert torch.equal(v, h([task(counts=(999,888),cell='hidden')], x,t))
    assert torch.isfinite(h([task()],x,t.new_ones(1))).all()

@pytest.mark.parametrize('device', ['cpu','cuda'])
def test_shared_context_state_and_time_gradients(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(6102)
    h=ResponseFlowDenoiserV2(SciPlexSharedModel(('A','B'),width=8)).to(device).double()
    x=torch.randn(1,2,device=device,dtype=torch.float64,requires_grad=True)
    t=x.new_tensor([.4],requires_grad=True)
    v=h([task()],x,t)
    assert not torch.equal(v,h([task(identity='SAHA')],x,t))
    loss=velocity_loss(v,torch.ones_like(v),torch.ones_like(v,dtype=torch.bool));loss.backward()
    assert x.grad.abs().sum()>0 and t.grad.abs().sum()>0
    assert h.native.backbone.feature.weight.grad.abs().sum()>0
    assert h.output.weight.grad.abs().sum()>0
    assert h.native.expression.weight.grad is None
    assert h.native.response_head.weight.grad is None
    assert h.native.intervention_head.weight.grad is None


def test_version_sigma_axis_and_input_rejection():
    for sigma in [0,1,-.1,float('nan'),True,'0.01']:
        with pytest.raises(ValueError):ResponseFlowDenoiserV2(SciPlexSharedModel(('A','B'),width=8),sigma_min=sigma)
    h=ResponseFlowDenoiserV2(SciPlexSharedModel(('A','B'),width=8)).double()
    for other in [ResponseFlowHead(SciPlexSharedModel(('A','B'),width=8)),
                  ResponseFlowDenoiserV2(SciPlexSharedModel(('A','B'),width=8),sigma_min=.02),
                  ResponseFlowDenoiserV2(SciPlexSharedModel(('B','A'),width=8))]:
        with pytest.raises(ValueError):other.double().load_state_dict(h.state_dict())
    twin=copy.deepcopy(h);twin.load_state_dict(h.state_dict())
    x=torch.ones(1,2,dtype=torch.float64);t=torch.ones(1,dtype=torch.float64)*.5
    assert torch.equal(h([task()],x,t),twin([task()],x,t))
    for z,s in [(x.float(),t),(x*float('nan'),t),(x,t+1),(x[:,:1],t)]:
        with pytest.raises(ValueError):h([task()],z,s)


def test_trainer_rejects_inconsistent_path_sigma():
    from mora3.model.response_flow_trainer import ResponseFlowTrainer, ResponseFlowBatch
    batch=ResponseFlowBatch((task(),))
    h=ResponseFlowDenoiserV2(SciPlexSharedModel(('A','B'),width=8),sigma_min=.02)
    with pytest.raises(ValueError,match='sigma'):
        ResponseFlowTrainer(h,batch,batch_size=1,sigma_min=.01)
