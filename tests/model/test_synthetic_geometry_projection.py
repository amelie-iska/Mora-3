"""Independent synthetic composition checks; no native source admission."""
import importlib
import pytest
import torch
from mora3.model.invariant_atom_projection import InvariantAtomProjection

DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
AXIS = ('mean_distance', 'mean_squared_distance', 'mean_squared_centered_dot')


def compose(x, keys, s, model):
    module = importlib.import_module('mora3.model.synthetic_geometry_features')
    assert hasattr(module, 'synthetic_projected_atoms'), 'missing synthetic composition'
    return module.synthetic_projected_atoms(x, keys, s, model, origin='synthetic_literal')


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_scalar_affine_and_all_gradients(device, dtype):
    torch.manual_seed(32201)
    x = torch.tensor([[0.,0.,0.],[3.,0.,0.],[0.,4.,0.]],device=device,dtype=dtype,requires_grad=True)
    s = torch.randn(3,4,device=device,dtype=dtype,requires_grad=True)
    model = InvariantAtomProjection(AXIS,4,width_identity='synthetic').to(device=device,dtype=dtype)
    out = compose(x,('a','b','c'),s,model)
    y = x.detach().clone().requires_grad_()
    center = [sum(y[i,k] for i in range(3))/3 for k in range(3)]
    rows = []
    for i in range(3):
        distances = [sum((y[i,k]-y[j,k])**2 for k in range(3)) for j in range(3)]
        dots = [sum((y[i,k]-center[k])*(y[j,k]-center[k]) for k in range(3)) for j in range(3)]
        rows.append(torch.stack((sum(v.sqrt() for j,v in enumerate(distances) if j!=i)/3,
                                 sum(distances)/3, sum(v*v for v in dots)/3)))
    features = torch.stack(rows)
    expected = torch.stack([torch.stack([sum(features[i,f]*model.finite.weight[w,f] for f in range(3))
        + model.finite.bias[w] + sum(s[i,k]*model.sequence.weight[w,k] for k in range(4))
        for w in range(4)]) for i in range(3)])
    torch.testing.assert_close(out.finite_features,features)
    torch.testing.assert_close(out.projected,expected)
    assert out.feature_order == AXIS and out.available.all()
    params = tuple(model.parameters())
    a = torch.autograd.grad(out.projected.square().sum(),(x,s)+params,retain_graph=True)
    b = torch.autograd.grad(expected.square().sum(),(y,s)+params)
    for actual, reference in zip(a,b):
        torch.testing.assert_close(actual,reference,rtol=3e-5,atol=3e-5)
    for name in ('require_source_use','require_model_use','require_training_use'):
        with pytest.raises(ValueError,match='synthetic'): getattr(out,name)()


@pytest.mark.parametrize('device', DEVICES)
def test_permutation_rigid_motion_and_singleton(device):
    torch.manual_seed(32201)
    model = InvariantAtomProjection(AXIS,4,width_identity='synthetic').double().to(device)
    x = torch.randn(3,3,device=device,dtype=torch.float64)
    s = torch.randn(3,4,device=device,dtype=torch.float64)
    base = compose(x,('a','b','c'),s,model)
    p = [2,0,1]
    moved = compose(x[p] @ x.new_tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])+7,
                    ('c','a','b'),s[p],model)
    torch.testing.assert_close(moved.projected,base.projected[p])
    singleton = compose(x[:1],('a',),s[:1],model)
    assert torch.equal(singleton.finite_features,torch.zeros_like(singleton.finite_features))


@pytest.mark.parametrize('axis', [('distance','centered_dot'), tuple(reversed(AXIS))])
def test_wrong_order_refuses_before_projection(axis):
    model = InvariantAtomProjection(axis,4,width_identity='synthetic')
    with pytest.raises(ValueError,match='channel'):
        compose(torch.zeros(3,3),('a','b','c'),torch.zeros(3,4),model)
