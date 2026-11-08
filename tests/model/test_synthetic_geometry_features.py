"""Synthetic numerical references; no actual-source geometry admission."""
import importlib
import math
import pytest
import torch


def api():
    try:
        return importlib.import_module('mora3.model.synthetic_geometry_features')
    except ModuleNotFoundError:
        pytest.fail('synthetic geometry feature constructor is missing')


def call(x, keys=('a', 'b', 'c'), **kw):
    return api().synthetic_internal_features(x, keys, origin='synthetic_literal', **kw)


DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


@pytest.mark.parametrize('device', DEVICES)
def test_independent_scalar_equations_and_gradients(device):
    x = torch.tensor([[0., 0., 0.], [3., 0., 0.], [0., 4., 0.]],
                     dtype=torch.float64, device=device, requires_grad=True)
    out = call(x)
    # Independent Python scalar equations, deliberately without tensor pair ops.
    rows = x.detach().cpu().tolist()
    center = [sum(r[k] for r in rows)/3 for k in range(3)]
    distances = [[math.sqrt(sum((a[k]-b[k])**2 for k in range(3)))
                  for b in rows] for a in rows]
    dots = [[sum((a[k]-center[k])*(b[k]-center[k]) for k in range(3))
             for b in rows] for a in rows]
    expected = torch.tensor([list(zip(a,b)) for a,b in zip(distances,dots)], device=device)
    torch.testing.assert_close(out.values, expected.to(torch.float64))
    assert out.feature_order == ('distance', 'centered_dot')
    assert out.atom_keys == ('a','b','c') and out.reference_keys == out.atom_keys
    assert out.available.dtype == torch.bool and out.available.all()
    assert out.available.shape == out.values.shape == (3,3,2)
    weights = torch.arange(18, dtype=x.dtype, device=device).reshape(3,3,2)/7
    actual = torch.autograd.grad((out.values*weights).sum(), x)[0]
    # Scalar-loop differentiable reference, including exact diagonal zero.
    y = x.detach().clone().requires_grad_()
    c = y.mean(0)
    loss = y.sum()*0
    for i in range(3):
        for j in range(3):
            d = y.sum()*0 if i == j else sum((y[i,k]-y[j,k])**2 for k in range(3)).sqrt()
            dot = sum((y[i,k]-c[k])*(y[j,k]-c[k]) for k in range(3))
            loss = loss + weights[i,j,0]*d + weights[i,j,1]*dot
    torch.testing.assert_close(actual, torch.autograd.grad(loss,y)[0])
    assert torch.isfinite(actual).all()
    for method in ('require_source_use','require_model_use','require_training_use'):
        with pytest.raises(ValueError, match='synthetic'):
            getattr(out,method)()


@pytest.mark.parametrize('device', DEVICES)
def test_rigid_motion_permutation_and_duplicate_points(device):
    torch.manual_seed(32101)
    x = torch.randn(4,3, dtype=torch.float64, device=device)
    keys = ('a','b','c','d')
    base = call(x, keys)
    rotation = torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]], dtype=x.dtype, device=device)
    moved = call(x@rotation + torch.tensor([5.,-7.,2.], device=device), keys)
    torch.testing.assert_close(base.values, moved.values)
    p = torch.tensor([2,0,3,1], device=device)
    shuffled = call(x[p], tuple(keys[i] for i in p.cpu().tolist()))
    torch.testing.assert_close(shuffled.values, base.values[p][:,p])
    duplicate = torch.zeros(3,3, dtype=x.dtype, device=device, requires_grad=True)
    values = call(duplicate).values
    values.sum().backward()
    assert torch.isfinite(duplicate.grad).all()
    assert torch.equal(values, torch.zeros_like(values))


@pytest.mark.parametrize('bad', [torch.zeros(3,2), torch.zeros(3,3,dtype=torch.int64),
    torch.full((3,3),float('nan')), torch.full((3,3),float('inf')),
    torch.empty(0,3), torch.zeros(3,3,dtype=torch.float16)])
def test_invalid_arrays_refuse(bad):
    with pytest.raises(ValueError):
        call(bad)


@pytest.mark.parametrize('keys', [('a','a','c'), ('a','b'), ['a','b','c'], ('a','','c')])
def test_axis_contract_refuses(keys):
    with pytest.raises(ValueError):
        call(torch.zeros(3,3),keys)


@pytest.mark.parametrize('kwargs', [dict(origin='source_observed'), dict(origin='predicted'),
    dict(origin='synthetic_literal', max_atoms=True), dict(origin='synthetic_literal', max_atoms=2),
    dict(origin='synthetic_literal', max_atoms=257)])
def test_roles_and_budget_refuse(kwargs):
    with pytest.raises(ValueError):
        api().synthetic_internal_features(torch.zeros(3,3),('a','b','c'),**kwargs)


def test_overflow_refuses_and_singleton_has_zero_features():
    with pytest.raises(ValueError):
        call(torch.tensor([[1e30,0.,0.],[-1e30,0.,0.],[0.,0.,0.]]))
    x = torch.tensor([[1.,2.,3.]], requires_grad=True)
    out = call(x, ('only',))
    assert out.values.shape == (1,1,2)
    assert torch.equal(out.values, torch.zeros_like(out.values))
    out.values.sum().backward()
    assert torch.equal(x.grad,torch.zeros_like(x))
