"""Independent tensor/equation oracles; synthetic fixtures are tests only."""
import importlib
import copy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import pytest
import torch


def api():
    return importlib.import_module('mora3.model.tokengt')


def graph(dtype=torch.float64):
    return dict(features=torch.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]], dtype=dtype),
                endpoints=torch.tensor([[0, 0], [1, 1], [0, 1], [1, 1]]),
                roles=torch.tensor([0, 0, 1, 1]), identifiers=torch.eye(2, dtype=dtype))


def model(dtype=torch.float64):
    torch.manual_seed(19)
    return api().TokenGT(2, 2, width=8, heads=2, layers=2, ffn_width=8).to(dtype=dtype)


def test_exact_ids_and_rank_guard():
    f = api().orthogonal_identifiers
    p = f(4, 7, seed=19, dtype=torch.float64)
    torch.testing.assert_close(p @ p.T, torch.eye(4, dtype=p.dtype), atol=1e-12, rtol=0)
    assert torch.equal(p, f(4, 7, seed=19, dtype=p.dtype))
    with pytest.raises(ValueError, match='width'):
        f(5, 4)


def test_endpoint_equation_and_explicit_self_loop_role():
    m, g = model(), graph()
    x = m.embed(g)
    p = g['identifiers']
    joined = torch.cat((p[g['endpoints'][:, 0]], p[g['endpoints'][:, 1]]), -1)
    expected = m.feature(g['features']) + joined @ m.incidence.weight.T + m.role(g['roles'])
    torch.testing.assert_close(x, expected)
    assert (joined[2] @ joined[:2].T).tolist() == [1., 1.]
    assert (joined[3] @ joined[:2].T).tolist() == [0., 2.]
    same = copy.deepcopy(g)
    same['features'][3] = same['features'][1]
    torch.testing.assert_close(m.embed(same)[3] - m.embed(same)[1], m.role.weight[1] - m.role.weight[0])
    reversed_g = copy.deepcopy(g)
    reversed_g['endpoints'][2] = torch.tensor([1, 0])
    assert not torch.allclose(m.embed(reversed_g)[2], x[2])


@pytest.mark.parametrize('change', ['endpoint', 'role', 'node', 'nonorthogonal', 'nan'])
def test_bad_contract_rejected(change):
    m, g = model(), graph()
    if change == 'endpoint': g['endpoints'][2, 0] = -1
    if change == 'role': g['roles'][0] = 2
    if change == 'node': g['endpoints'][0, 1] = 1
    if change == 'nonorthogonal': g['identifiers'][0] = 0
    if change == 'nan': g['features'][0, 0] = float('nan')
    with pytest.raises(ValueError): m([g])


def test_attention_equation_and_gradients():
    a = api().DenseAttention(8, 2).double()
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
    valid = torch.tensor([[True, True, False, False], [True, True, True, True]])
    actual = a(x, valid)
    # Independent per-batch, per-head equation, without implementation reshape logic.
    q, k, v = a.qkv(x).chunk(3, dim=-1)
    oracle = []
    for b in range(2):
        hs = []
        for h in range(2):
            sl = slice(h * 4, (h + 1) * 4)
            logits = q[b, :, sl] @ k[b, :, sl].T / 2
            logits[:, ~valid[b]] = -torch.inf
            hs.append(logits.softmax(-1) @ v[b, :, sl])
        oracle.append(a.output(torch.cat(hs, -1)))
    expected = torch.stack(oracle)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    variables = (x, *a.parameters())
    ga = torch.autograd.grad(actual.square().sum(), variables, retain_graph=True)
    ge = torch.autograd.grad(expected.square().sum(), variables)
    for left, right in zip(ga, ge): torch.testing.assert_close(left, right)
    with pytest.raises(ValueError): a(x, torch.zeros_like(valid))


def test_permutation_transport_padding_and_gradients():
    m, g = model(), graph()
    base, valid = m([g])
    # New node order [1,0], entities [node1,node0,self-loop,edge].
    order = torch.tensor([1, 0, 3, 2])
    perm = dict(features=g['features'][order], roles=g['roles'][order],
                endpoints=1 - g['endpoints'][order], identifiers=g['identifiers'].flip(0))
    other, _ = m([perm])
    torch.testing.assert_close(other[:, 0], base[:, 0])
    torch.testing.assert_close(other[:, 1:], base[:, 1:][:, order])
    short = dict(features=g['features'][:1], roles=torch.tensor([0]),
                 endpoints=torch.tensor([[0, 0]]), identifiers=g['identifiers'][:1])
    alone, _ = m([short])
    batched, mask = m([short, g])
    torch.testing.assert_close(alone[0], batched[0, :2])
    assert mask.tolist() == [[True, True, False, False, False], [True] * 5]
    assert torch.count_nonzero(batched[0, 2:]) == 0
    batched[mask].square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    assert api().parameter_count(m) == sum(p.numel() for p in m.parameters())


def test_conditional_mask_rejected_and_empty_batch_rejected():
    m = model()
    with pytest.raises(ValueError, match='conditional'): m([graph()], attention_mask=torch.ones(5, 5))
    with pytest.raises(ValueError): m([])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_gpu_forward_backward_parity():
    cpu, g = model(torch.float32), graph(torch.float32)
    gpu = copy.deepcopy(cpu).cuda()
    a, _ = cpu([g])
    b, _ = gpu([{k: v.cuda() for k, v in g.items()}])
    torch.testing.assert_close(a, b.cpu(), atol=2e-5, rtol=2e-5)
    a.square().sum().backward()
    b.square().sum().backward()
    for p, q in zip(cpu.parameters(), gpu.parameters()):
        torch.testing.assert_close(p.grad, q.grad.cpu(), atol=2e-4, rtol=2e-4)
