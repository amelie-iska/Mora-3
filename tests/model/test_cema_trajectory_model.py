"""Independent dense Gaussian check for exact irregular missing-time marginalization."""
import torch

from mora3.model.cema_trajectory_model import irregular_trace_likelihood


def test_markov_score_and_gradients_equal_exact_dense_marginal_after_gaps():
    dtype=torch.float64
    full=torch.tensor([0.,.02,.37,.45,1.17,3.5],dtype=dtype)
    mean=torch.tensor([.2,-.3,.01,.18,.1,.42],dtype=dtype,requires_grad=True)
    scale=torch.tensor(.74,dtype=dtype,requires_grad=True)
    rate=torch.tensor(.18,dtype=dtype,requires_grad=True)
    keep=torch.tensor([0,1,4,5])
    y=torch.tensor([.23,-.49,-.1,.77],dtype=dtype)
    scored=irregular_trace_likelihood(mean,scale,rate,full,keep,y).sum()
    covariance=scale.square()*torch.exp(-rate*(full[:,None]-full[None,:]).abs())
    oracle=-torch.distributions.MultivariateNormal(mean[keep],covariance[keep][:,keep]).log_prob(y)
    torch.testing.assert_close(scored,oracle,rtol=5e-13,atol=5e-13)
    observed_grads=torch.autograd.grad(scored,(mean,scale,rate))
    expected_grads=torch.autograd.grad(oracle,(mean,scale,rate))
    for a,b in zip(observed_grads,expected_grads):torch.testing.assert_close(a,b,rtol=5e-12,atol=5e-12)
    assert torch.equal(observed_grads[0][[2,3]],torch.zeros(2,dtype=dtype))
