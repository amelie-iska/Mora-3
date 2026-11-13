"""A tied square must not change the auxiliary objective under node relabeling."""
from itertools import permutations
import torch
from mora3.reasoning.persistent_topology import topological_metric_loss


def test_tied_filtration_loss_and_gradient_are_permutation_equivariant():
    source = torch.tensor([[0.,0.],[1.,0.],[1.,1.],[0.,1.]], dtype=torch.float64)
    latent = torch.tensor([[0.,0.],[.2,0.],[1.,0.],[2.,0.]], dtype=torch.float64)
    reference = torch.cdist(source, source)
    z = latent.clone().requires_grad_()
    base = topological_metric_loss(reference, torch.cdist(z,z))['loss']
    gradient, = torch.autograd.grad(base, z)
    for order in permutations(range(4)):
        indices = list(order)
        z = latent[indices].clone().requires_grad_()
        actual = topological_metric_loss(reference[indices][:,indices], torch.cdist(z,z))
        torch.testing.assert_close(actual['loss'], base, rtol=1e-12, atol=1e-12)
        changed, = torch.autograd.grad(actual['loss'], z)
        torch.testing.assert_close(changed, gradient[indices], rtol=1e-12, atol=1e-12)
    assert actual['tie_policy'] == 'all-edges-at-persistence-critical-levels'
