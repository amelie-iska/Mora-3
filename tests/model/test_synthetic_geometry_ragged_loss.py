"""Synthetic projected-velocity arithmetic is not a native Cartesian decoder."""
import importlib
import pytest
import torch
from mora3.model.invariant_atom_projection import InvariantAtomProjection
from mora3.model.synthetic_geometry_features import SYNTHETIC_SUMMARY_AXIS

DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])

def compose(*args, **kwargs):
    module = importlib.import_module('mora3.model.synthetic_geometry_features')
    assert hasattr(module, 'synthetic_projected_velocity_loss'), 'missing synthetic ragged composition'
    return module.synthetic_projected_velocity_loss(*args, **kwargs)

@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_independent_ragged_scalar_and_end_to_end_gradients(device,dtype):
    torch.manual_seed(32301)
    model=InvariantAtomProjection(SYNTHETIC_SUMMARY_AXIS,3,width_identity='synthetic').to(device=device,dtype=dtype)
    xs=[torch.randn(n,3,device=device,dtype=dtype,requires_grad=True) for n in (2,4)]
    ss=[torch.randn_like(x,requires_grad=True) for x in xs]
    us=[torch.randn_like(x,requires_grad=True) for x in xs]
    masks=[torch.tensor(m,device=device) for m in ([True,False],[True,True,True,False])]
    out=compose(xs,[('a','b'),('c','d','e','f')],ss,model,us,masks,(True,True),origin='synthetic_literal')
    ys=[x.detach().clone().requires_grad_() for x in xs]
    cases=[]
    for y,s,u,mask in zip(ys,ss,us,masks):
        n=len(y); center=[sum(y[i,k] for i in range(n))/n for k in range(3)]
        residuals=[]
        for i in range(n):
            if not mask[i]: continue
            ds=[sum((y[i,k]-y[j,k])**2 for k in range(3)) for j in range(n)]
            gs=[sum((y[i,k]-center[k])*(y[j,k]-center[k]) for k in range(3)) for j in range(n)]
            f=[sum(ds[j].sqrt() for j in range(n) if j!=i)/n,sum(ds)/n,sum(g*g for g in gs)/n]
            for w in range(3):
                p=sum(f[k]*model.finite.weight[w,k] for k in range(3))+model.finite.bias[w]+sum(s[i,k]*model.sequence.weight[w,k] for k in range(3))
                residuals.append((p-u[i,w].detach())**2)
        cases.append(sum(residuals)/len(residuals))
    expected=sum(cases)/2
    torch.testing.assert_close(out.loss,expected)
    params=tuple(model.parameters())
    a=torch.autograd.grad(out.loss,tuple(xs+ss)+params,retain_graph=True)
    b=torch.autograd.grad(expected,tuple(ys+ss)+params)
    for actual,reference in zip(a,b): torch.testing.assert_close(actual,reference,rtol=8e-5,atol=8e-5)
    assert all(u.grad is None for u in us)
    assert out.active_cases==(0,1)
    for method in ('require_source_use','require_model_use','require_training_use'):
        with pytest.raises(ValueError,match='synthetic'):getattr(out,method)()


def test_inactive_payloads_opaque_and_empty_preflight_no_projection():
    model=InvariantAtomProjection(SYNTHETIC_SUMMARY_AXIS,3,width_identity='synthetic')
    calls=[]
    hook=model.register_forward_pre_hook(lambda *a:calls.append(True))
    args=([torch.zeros(1,3),object()],[('a',),object()],[torch.ones(1,3),object()],model,[torch.zeros(1,3),object()],[torch.tensor([True]),torch.tensor([True])])
    before=torch.random.get_rng_state().clone()
    with pytest.raises(ValueError,match='active'):compose(*args,(False,False),origin='synthetic_literal')
    assert not calls and torch.equal(before,torch.random.get_rng_state())
    out=compose(*args,(True,False),origin='synthetic_literal')
    assert out.active_cases==(0,) and len(calls)==1
    hook.remove()

@pytest.mark.parametrize('failure',['width','length','origin'])
def test_refuses_invalid_composition(failure):
    model=InvariantAtomProjection(SYNTHETIC_SUMMARY_AXIS,4 if failure=='width' else 3,width_identity='synthetic')
    with pytest.raises(ValueError):
        compose([torch.zeros(1,3)],[] if failure=='length' else [('a',)],[torch.ones(1,3)],model,[torch.zeros(1,3)],[torch.tensor([True])],(True,),origin='source' if failure=='origin' else 'synthetic_literal')
