"""Exact objective reference; synthetic masks are not source coverage."""
import itertools
import pytest
import torch
from torch.nn import functional as F
from mora3.model.instance_affinity import overlapping_set_objective

DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])

@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('n', [0, 1, 3])
@pytest.mark.parametrize('block', [1, 4, 100])
def test_blocked_matches_exhaustive_and_dense(device, n, block):
    torch.manual_seed(22701)
    x = torch.randn(3, 2, 5, device=device, dtype=torch.float64, requires_grad=True)
    y = torch.rand(n, 2, 5, device=device) > .4
    padded = torch.cat([y, torch.zeros(3-n, 2, 5, device=device, dtype=torch.bool)])
    choices = [(F.binary_cross_entropy_with_logits(x, padded[list(p)].double()), p)
               for p in itertools.permutations(range(3))]
    oracle, _ = min(choices, key=lambda v: v[0].item())
    dense = overlapping_set_objective(x, y)
    blocked = overlapping_set_objective(x, y, pixel_block_size=block)
    torch.testing.assert_close(blocked, oracle, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(blocked, dense, rtol=1e-12, atol=1e-12)
    g, = torch.autograd.grad(blocked, x, retain_graph=True)
    reference, = torch.autograd.grad(oracle, x)
    torch.testing.assert_close(g, reference, rtol=1e-12, atol=1e-12)

@pytest.mark.parametrize('device', DEVICES)
def test_exact_ties_preserve_valid_minimizing_gradient(device):
    x = torch.zeros(3, 2, 3, device=device, dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[[1,1,0],[0,0,0]], [[0,1,1],[0,0,0]]], device=device, dtype=torch.bool)
    a = overlapping_set_objective(x, y)
    b = overlapping_set_objective(x, y, pixel_block_size=2)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(a,x)[0], torch.autograd.grad(b,x)[0], rtol=0, atol=0)

@pytest.mark.parametrize('block', [True, False, 0, -1, 1.5, '2'])
def test_invalid_block_refused(block):
    with pytest.raises(ValueError, match='block'):
        overlapping_set_objective(torch.zeros(2,2,2), torch.ones(1,2,2,dtype=torch.bool), pixel_block_size=block)

@pytest.mark.parametrize('device', DEVICES)
def test_policy_and_target_validation_unchanged(device):
    for x,y in [(torch.zeros(65,1,1),torch.ones(1,1,1,dtype=torch.bool)),
                (torch.zeros(1,1,1),torch.ones(2,1,1,dtype=torch.bool)),
                (torch.zeros(2,1,1),torch.zeros(1,1,1,dtype=torch.bool)),
                (torch.full((2,1,1),float('nan')),torch.ones(1,1,1,dtype=torch.bool))]:
        with pytest.raises(ValueError):
            overlapping_set_objective(x.to(device), y.to(device), pixel_block_size=1)

@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_strided_inputs_preserve_all_pixels_and_gradients(device, dtype):
    torch.manual_seed(22702)
    x = torch.randn(4, 5, 7, device=device, dtype=dtype).transpose(1,2).requires_grad_()
    y = (torch.rand(2,5,7,device=device) > .5).transpose(1,2)
    a = overlapping_set_objective(x,y)
    b = overlapping_set_objective(x,y,pixel_block_size=6)
    torch.testing.assert_close(a,b)
    torch.testing.assert_close(torch.autograd.grad(a,x)[0],torch.autograd.grad(b,x)[0])


def test_fp64_conversion_workspace_is_pixel_bounded():
    # Inspect actual conversion operand shapes, not an estimate of peak memory.
    from torch.utils._python_dispatch import TorchDispatchMode
    conversions = []
    class Observe(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if func == torch.ops.aten._to_copy.default and kwargs.get('dtype') == torch.float64:
                conversions.append(tuple(args[0].shape))
            return func(*args, **kwargs)
    x = torch.randn(3,5,7)
    y = torch.ones(2,5,7,dtype=torch.bool)
    with Observe():
        overlapping_set_objective(x,y,pixel_block_size=6)
    assert conversions and all(len(s)==2 and s[1]<=6 for s in conversions)
    assert (3,35) not in conversions and (2,35) not in conversions
