"""Removing either endpoint context or equal-case reduction breaks references."""
import importlib
from dataclasses import replace
import pytest
import torch
from test_ragged_current_frame_velocity import inputs
from mora3.model.current_frame_velocity import SplitContextualCoefficients


def api():
    module = importlib.import_module('mora3.model.current_frame_velocity')
    assert hasattr(module, 'split_context_current_frame_velocity_loss'), 'missing split ragged composition'
    return module.split_context_current_frame_velocity_loss


def setup(device='cpu', dtype=torch.float64, width=8):
    a = inputs(device, dtype)
    torch.manual_seed(33501)
    axis = tuple(f'h{i}' for i in range(width))
    model = SplitContextualCoefficients(axis, max_atoms=8).to(device=device, dtype=dtype)
    context = [torch.randn(len(x), width, device=device, dtype=dtype, requires_grad=True) for x in a[0]]
    return a, model, context, axis


def call(a, model, context, axis):
    return api()(a[0], context, model, a[3], a[4], axis, a[6], a[7], a[8], origin='synthetic_literal', width_identity='diagnostic')


def reference(x, h, model, mask):
    n = len(x)
    center = sum(x)/n
    rows = []
    for i in range(n):
        terms = []
        for j in range(n):
            distance = torch.linalg.vector_norm(x[j]-x[i])
            dot = sum((x[i,k]-center[k])*(x[j,k]-center[k]) for k in range(3))
            scalar = model.geometry.weight[0,0]*distance + model.geometry.weight[0,1]*dot
            scalar = scalar + sum(model.receiver.weight[0,k]*h[i,k] + model.neighbor.weight[0,k]*h[j,k] for k in range(h.shape[1]))
            terms.append(scalar*(x[j]-x[i]))
        rows.append(sum(terms)/n)
    velocity = torch.stack(rows)
    return torch.where(mask[:,None], velocity, torch.zeros_like(velocity))


@pytest.mark.parametrize('device', ['cpu','cuda'])
@pytest.mark.parametrize('dtype', [torch.float32,torch.float64])
@pytest.mark.parametrize('width', [8,12,16])
def test_independent_ragged_loss_gradients_and_covariance(device,dtype,width):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    a,m,h,axis = setup(device,dtype,width)
    out = call(a,m,h,axis)
    vs = [reference(x,c,m,mask) for x,c,mask in zip(a[0],h,a[7])]
    expected = sum((v[mask]-u.detach()[mask]).square().mean() for v,u,mask in zip(vs,a[6],a[7]))/2
    tol = 3e-4 if dtype == torch.float32 else 1e-9
    torch.testing.assert_close(out.loss,expected,rtol=tol,atol=tol)
    operands = tuple(a[0]+h)+tuple(m.parameters())
    actual = torch.autograd.grad(out.loss,operands,retain_graph=True)
    target = torch.autograd.grad(expected,operands,retain_graph=True)
    for g,e in zip(actual,target): torch.testing.assert_close(g,e,rtol=tol,atol=tol)
    assert torch.autograd.grad(out.loss,tuple(a[6]),allow_unused=True) == (None,None)
    q = torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,-1.]],device=device,dtype=dtype)
    b=list(a); b[0]=[x@q.T+2 for x in a[0]]; b[6]=[u@q.T for u in a[6]]
    torch.testing.assert_close(call(b,m,h,axis).loss,out.loss,rtol=tol,atol=tol)
    from mora3.data.geometry_frame_policy import axis_digest
    b=list(a); b[0]=[x.flip(0) for x in a[0]]; b[6]=[u.flip(0) for u in a[6]]; b[7]=[mask.flip(0) for mask in a[7]]
    b[4]=[ax[::-1] for ax in a[4]]
    b[3]=[replace(binding,native_axis=ax,native_axis_sha256=axis_digest(ax),free_atoms=binding.free_atoms[::-1]) for binding,ax in zip(a[3],b[4])]
    torch.testing.assert_close(call(b,m,[c.flip(0) for c in h],axis).loss,out.loss,rtol=tol,atol=tol)
    assert out.active_cases == (0,1)
    for name in ('require_source_use','require_model_use','require_training_use'):
        with pytest.raises(ValueError): getattr(out,name)()


@pytest.mark.parametrize('bad',['context_nan','context_shape','context_dtype','xyz','target','axis','mask','binding','length','context_axis','parameters','owner','empty'])
def test_all_active_preflight_before_any_child(bad):
    a,m,h,axis=setup(); calls=[]
    for mod in (m.geometry,m.receiver,m.neighbor): mod.register_forward_pre_hook(lambda *args:calls.append(1))
    if bad=='context_nan': h[1]=torch.full_like(h[1],float('nan'))
    if bad=='context_shape': h[1]=h[1][:,:1]
    if bad=='context_dtype': h[1]=h[1].float()
    if bad=='xyz': a[0][1]=torch.full_like(a[0][1],float('nan'))
    if bad=='target': a[6][1]=torch.full_like(a[6][1],float('nan'))
    if bad=='axis': a[4][1]=a[4][1][::-1]
    if bad=='mask': a[7][1]=torch.tensor([False,True])
    if bad=='binding': a[3][1]=replace(a[3][1],xyz_available=(True,False))
    if bad=='length': h=[]
    if bad=='context_axis': axis=axis[::-1]
    if bad=='parameters':
        with torch.no_grad(): m.neighbor.weight[0,0]=float('nan')
    if bad=='owner': m.extra=torch.nn.Linear(1,1)
    if bad=='empty': a[-1]=(False,False)
    rng=torch.random.get_rng_state().clone()
    with pytest.raises(ValueError): call(a,m,h,axis)
    assert calls==[] and torch.equal(rng,torch.random.get_rng_state())


def test_inactive_opaque_and_fixed_target_nan():
    a,m,h,axis=setup(); a[-1]=(True,False)
    for index in (0,3,4,6): a[index][1]=object()
    h[1]=object(); a[6][0]=a[6][0].detach().clone(); a[6][0][1]=float('nan')
    assert call(a,m,h,axis).active_cases==(0,)
    with pytest.raises(TypeError): api()(target_xyz=a[0])
