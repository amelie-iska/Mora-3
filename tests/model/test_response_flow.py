import copy
import pytest
import torch
from test_sciplex_shared import task
from mora3.model.sciplex import SciPlexSharedModel
from mora3.model.response_flow import ResponseFlowHead

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_context_excludes_clean_target_and_state_time_are_explicit(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(5401)
    native=SciPlexSharedModel(('A','B'),width=8).to(device).double()
    head=ResponseFlowHead(native).to(device).double()
    x=torch.tensor([[.1,.2]],device=device,dtype=torch.float64,requires_grad=True)
    t=torch.tensor([.4],device=device,dtype=torch.float64,requires_grad=True)
    a=head([task()],x,t)
    assert torch.equal(a,head([task(counts=(99,77),cell='hidden')],x,t))
    assert not torch.equal(a,head([task(identity='SAHA')],x,t))
    a.sum().backward()
    assert (x.grad.abs()>0).all() and t.grad.abs().sum()>0
    assert native.backbone.feature.weight.grad.abs().sum()>0
    assert native.response_head.weight.grad is None
    assert native.intervention_head.weight.grad is None
    assert native.expression.weight.grad is None
    assert head.native is native


def test_native_equation_parity_and_validation():
    torch.manual_seed(5401)
    head=ResponseFlowHead(SciPlexSharedModel(('A','B'),width=8)).double()
    x=torch.tensor([[.2,.3]],dtype=torch.float64);t=torch.tensor([.5],dtype=torch.float64)
    context=head.native.backbone([head.native.graph(task())])[0][:,0]
    expected=head.output(torch.nn.functional.silu(head.state(x)+head.time(t[:,None])+context))
    torch.testing.assert_close(head([task()],x,t),expected,atol=1e-15,rtol=1e-15)
    for ts,z,s in [([task('intervention')],x,t),([],x,t),([task()],x[:,:1],t),([task()],x,t+1),([task()],x.float(),t),([task()],x*float('nan'),t)]:
        with pytest.raises(ValueError):head(ts,z,s)
    other=ResponseFlowHead(SciPlexSharedModel(('B','A'),width=8)).double()
    with pytest.raises(ValueError):other.load_state_dict(head.state_dict())
    if torch.cuda.is_available():
        gpu=copy.deepcopy(head).cuda()
        a=head([task()],x,t);b=gpu([task()],x.cuda(),t.cuda())
        a.sum().backward();b.sum().backward()
        torch.testing.assert_close(a,b.cpu(),atol=1e-10,rtol=1e-10)
        for p,q in zip(head.parameters(),gpu.parameters()):
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad.cpu(),atol=1e-9,rtol=1e-8)
