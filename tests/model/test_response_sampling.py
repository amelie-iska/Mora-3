import pytest
import torch
from test_sciplex_shared import task
from mora3.model.sciplex import SciPlexSharedModel
from mora3.model.response_flow import ResponseFlowHead
from mora3.model.response_sampling import ResponseContext, sample_response

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_target_free_context_parity_seed_and_explicit_euler(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(5601)
    h=ResponseFlowHead(SciPlexSharedModel(('A','B'),width=8)).to(device).double()
    c=ResponseContext.from_task(task())
    assert c==ResponseContext.from_task(task(counts=(99,88),cell='hidden'))
    assert c!=ResponseContext.from_task(task(identity='SAHA'))
    assert not hasattr(c,'task')
    x=torch.tensor([[.1,.2]],device=device,dtype=torch.float64)
    t=x.new_tensor([.4])
    torch.testing.assert_close(h([task()],x,t),h([c],x,t),rtol=0,atol=0)
    h.train();before=torch.random.get_rng_state().clone()
    r=sample_response(h,[c],steps=4,seed=56,sigma_min=.01,negative_policy='preserve')
    assert h.training and torch.equal(before,torch.random.get_rng_state())
    assert torch.equal(r['log_expression'],sample_response(h,[c],steps=4,seed=56,sigma_min=.01,negative_policy='preserve')['log_expression'])
    g=torch.Generator(device=device).manual_seed(56)
    z=torch.randn((1,2),generator=g,device=device,dtype=torch.float64)
    with torch.no_grad():
        for i in range(4):z=z+h([c],z,z.new_full((1,),i/4))/4
    torch.testing.assert_close(r['log_expression'],z,rtol=0,atol=0)
    assert not r['log_expression'].requires_grad
    assert r['counts'] is None and r['binding']
    clipped=sample_response(h,[c],steps=4,seed=56,sigma_min=.01,negative_policy='clamp-zero')
    torch.testing.assert_close(clipped['counts'],torch.expm1(z.clamp_min(0)))
    assert not torch.equal(z,sample_response(h,[c],steps=4,seed=57,sigma_min=.01,negative_policy='preserve')['log_expression'])


def test_invalid_axis_solver_policy_and_context_rejected():
    h=ResponseFlowHead(SciPlexSharedModel(('B','A'),width=8)).double()
    c=ResponseContext.from_task(task())
    with pytest.raises(ValueError):sample_response(h,[c],steps=2,seed=1,sigma_min=.01,negative_policy='preserve')
    with pytest.raises(ValueError):ResponseContext.from_task(task('intervention'))
    h=ResponseFlowHead(SciPlexSharedModel(('A','B'),width=8)).double()
    for kw in [dict(steps=0),dict(seed=True),dict(sigma_min=1),dict(negative_policy='implicit')]:
        args=dict(steps=2,seed=1,sigma_min=.01,negative_policy='preserve');args.update(kw)
        with pytest.raises(ValueError):sample_response(h,[c],**args)
