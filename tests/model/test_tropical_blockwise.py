"""Catch lost cross-tile winners, tie-gradient changes and padding leakage."""
import copy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import pytest
import torch
from mora3.model import tropical


def compare(q, k, v, mask, block):
    a = tropical.tropical_context(q, k, v, mask)
    b = tropical.tropical_context_blockwise(q, k, v, mask, block_size=block)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    weight = torch.randn_like(a)
    ga = torch.autograd.grad((a * weight).sum(), (q, k, v), retain_graph=True)
    gb = torch.autograd.grad((b * weight).sum(), (q, k, v))
    for x, y in zip(ga, gb):
        torch.testing.assert_close(x, y, rtol=2e-5, atol=2e-5)
    return gb


@pytest.mark.parametrize('block', [1, 2, 4, 32])
def test_irregular_tiles_and_masked_blocks(block):
    torch.manual_seed(19)
    xs = [torch.randn(2, 2, n, d, dtype=torch.double, requires_grad=True)
          for n, d in [(5, 4), (7, 4), (7, 3)]]
    mask = torch.tensor([[False]*4 + [True, False, True], [True]*7])
    grads = compare(*xs, mask, block)
    assert grads[1][0, :, :4].abs().sum() == 0
    assert grads[2][0, :, :4].abs().sum() == 0


def test_ties_across_tiles_choose_first_valid_global_key_and_coordinate():
    q = torch.tensor([[[[2., 2., 0.], [2., 2., 0.]]]], requires_grad=True)
    k = torch.zeros(1, 1, 5, 3, requires_grad=True)
    v = torch.ones(1, 1, 5, 2, requires_grad=True)
    mask = torch.tensor([[False, True, True, True, True]])
    out = tropical.tropical_context_blockwise(q, k, v, mask, block_size=2)
    torch.testing.assert_close(out, torch.full_like(out, -1))
    out.sum().backward()
    assert q.grad.tolist() == [[[[-2., 0., 2.], [-2., 0., 2.]]]]
    assert k.grad[0, 0, 1].tolist() == [4., 0., -4.]
    assert v.grad[0, 0, :, 0].tolist() == [0., 2., 0., 0., 0.]
    compare(q, k, v, mask, 2)


def test_gradcheck_noncontiguous_and_contracts():
    torch.manual_seed(19)
    xs = [torch.randn(1, 1, 3, 4, dtype=torch.double).transpose(2, 3).requires_grad_()
          for _ in range(3)]
    mask = torch.ones(1, 4, dtype=torch.bool)
    f = lambda *args: tropical.tropical_context_blockwise(*args, mask, block_size=3)
    assert torch.autograd.gradcheck(f, tuple(xs))
    for size in [0, -1, True, 1.5]:
        with pytest.raises(ValueError):
            tropical.tropical_context_blockwise(*xs, mask, block_size=size)
    with pytest.raises(ValueError):
        tropical.tropical_context_blockwise(*xs, ~mask, block_size=2)
    with pytest.raises(ValueError):
        tropical.tropical_context_blockwise(*xs, mask.float(), block_size=2)


def test_wrapper_and_classifier_parameter_gradient_parity():
    from mora3.model.training import RNAClassifier
    from mora3.tokenization.shortest_path import ByteTokenizer
    from test_training import record
    torch.manual_seed(19)
    a = RNAClassifier(ByteTokenizer(controls=('pad','node','edge')), ('a','b'), attention='tropical-code').double()
    b = RNAClassifier(ByteTokenizer(controls=('pad','node','edge')), ('a','b'),
                      attention='tropical-blockwise', tropical_block_size=2).double()
    b.load_state_dict(a.state_dict())
    rows = [record('a'), record('b')]
    x, y = a(rows), b(rows)
    torch.testing.assert_close(x, y, rtol=0, atol=0)
    x.square().sum().backward(); y.square().sum().backward()
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_gpu_parity_and_deterministic_backward():
    torch.manual_seed(19)
    xs = [torch.randn(2, 2, n, d, requires_grad=True) for n, d in [(5,4),(7,4),(7,3)]]
    mask = torch.tensor([[False]*4+[True]*3, [True]*7])
    cpu = tropical.tropical_context_blockwise(*xs, mask, block_size=2)
    ys = [x.detach().cuda().requires_grad_() for x in xs]
    old = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        gpu = tropical.tropical_context_blockwise(*ys, mask.cuda(), block_size=2)
        torch.testing.assert_close(cpu, gpu.cpu())
        cpu.square().sum().backward(); gpu.square().sum().backward()
        for x, y in zip(xs, ys):
            torch.testing.assert_close(x.grad, y.grad.cpu(), rtol=2e-5, atol=2e-5)
        compare(*ys, mask.cuda(), 2)
    finally:
        torch.use_deterministic_algorithms(old)
