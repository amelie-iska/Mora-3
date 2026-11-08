"""Analytic capacity audit: target-aware oracles are never learned samplers."""
from dataclasses import replace
from mora3.model.response_sampling import ResponseContext
import pytest
import torch
from mora3.model import flow_matching as fm
from mora3.model.response_flow import ResponseFlowHead
from mora3.model.sciplex import SciPlexSharedModel
from test_sciplex_shared import task


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_conditional_state_velocity_matches_path_and_full_rank_jacobian(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('no CUDA')
    oracle = getattr(fm, 'gaussian_conditional_velocity', None)
    assert callable(oracle), 'missing target-aware analytic state-velocity reference'
    torch.manual_seed(6001)
    noise = torch.randn(3, 24, device=device, dtype=torch.float64)
    target = torch.randn_like(noise).requires_grad_()
    time = noise.new_tensor([0., .4, 1.])
    state, expected = fm.gaussian_path(noise, target, time, sigma_min=.01)
    actual = oracle(state, target, time, sigma_min=.01)
    torch.testing.assert_close(actual, expected, atol=3e-14, rtol=3e-14)
    assert not actual.requires_grad
    x = state[:1].detach().requires_grad_()
    y = target[:1].detach().requires_grad_()
    t = time[1:2]
    jac = torch.autograd.functional.jacobian(lambda z: oracle(z, y, t), x).reshape(24, 24)
    expected_jac = torch.eye(24, device=device, dtype=x.dtype) * (-.99 / (1-.99*t.item()))
    torch.testing.assert_close(jac, expected_jac)
    assert torch.linalg.matrix_rank(jac).item() == 24
    grads = torch.autograd.grad(oracle(x,y,t).sum(), (x,y), allow_unused=True)
    assert grads[0] is not None and grads[1] is None
    # State-dependent oracle Euler integration recovers the affine exact path.
    result = fm.euler_flow(lambda z,s: oracle(z,target.detach(),s), noise, steps=16)
    torch.testing.assert_close(result, target.detach()+.01*noise, atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_v1_state_nullspace_and_integrated_output_subspace(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('no CUDA')
    torch.manual_seed(6001)
    head = ResponseFlowHead(SciPlexSharedModel(tuple(f'G{i}' for i in range(24)), width=8)).to(device).double()
    context = replace(ResponseContext.from_task(task()), axis_ids=head.native.axis_ids)
    hidden = replace(ResponseContext.from_task(task(counts=(99,77),cell='hidden')), axis_ids=head.native.axis_ids)
    x = torch.randn(1,24, device=device, dtype=torch.float64)
    t = x.new_tensor([.4])
    # Full SVD of small test matrix only; real audit never makes 5000x5000 matrices.
    _,_,vh = torch.linalg.svd(head.state.weight.detach(), full_matrices=True)
    invisible = vh[-1:]
    torch.testing.assert_close(head([context], x, t), head([context], x+invisible, t), atol=1e-13, rtol=1e-13)
    jac = torch.autograd.functional.jacobian(lambda z:head([context],z,t), x).reshape(24,24)
    assert torch.linalg.matrix_rank(jac).item() <= 8
    basis = torch.linalg.qr(torch.cat((head.output.weight.detach(),head.output.bias.detach()[:,None]),1),mode='reduced')[0]
    residual = lambda z: z - (z@basis)@basis.T
    out = fm.euler_flow(lambda z,s:head([context],z,s),x,steps=16)
    torch.testing.assert_close(residual(out), residual(x),atol=1e-13,rtol=1e-13)
    assert torch.equal(head([context],x,t),head([hidden],x,t))


@pytest.mark.parametrize('fault', ['singular','time','shape','nonfinite','dtype'])
def test_conditional_reference_fails_closed(fault):
    oracle = getattr(fm, 'gaussian_conditional_velocity', None)
    assert callable(oracle), 'missing target-aware analytic state-velocity reference'
    x=torch.ones(2,3,dtype=torch.float64);y=x.clone();t=x.new_zeros(2);sigma=.01
    if fault=='singular':sigma=0.;t.fill_(1.)
    if fault=='time':t.fill_(1.1)
    if fault=='shape':y=y[:1]
    if fault=='nonfinite':x[0,0]=float('nan')
    if fault=='dtype':y=y.float()
    with pytest.raises(ValueError):oracle(x,y,t,sigma_min=sigma)
