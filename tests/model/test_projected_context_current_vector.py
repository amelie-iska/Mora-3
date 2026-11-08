"""Composition witnesses: omitting sequence projection breaks gradient parity."""
import importlib
from dataclasses import replace
import pytest
import torch
from test_ragged_current_frame_velocity import inputs
from mora3.model.current_frame_velocity import ContextualCurrentFrameVelocityReadout
from mora3.model.invariant_atom_projection import InvariantAtomProjection
from mora3.model.synthetic_geometry_features import SYNTHETIC_SUMMARY_AXIS


def api():
    m=importlib.import_module('mora3.model.current_frame_velocity')
    assert hasattr(m,'projected_context_current_frame_velocity_loss'), 'missing projected context composition'
    return m.projected_context_current_frame_velocity_loss


def setup(device='cpu',dtype=torch.float64,width=2):
    a=inputs(device,dtype);torch.manual_seed(33101)
    axis=tuple('h'+str(i) for i in range(width))
    p=InvariantAtomProjection(SYNTHETIC_SUMMARY_AXIS,width,width_identity='diagnostic').to(device=device,dtype=dtype)
    r=ContextualCurrentFrameVelocityReadout(axis,max_atoms=8).to(device=device,dtype=dtype)
    s=[torch.randn(len(x),width,device=device,dtype=dtype,requires_grad=True) for x in a[0]]
    return a,p,r,s,axis


def call(a,p,r,s,axis):
    return api()(a[0],s,p,r,a[3],a[4],axis,a[6],a[7],a[8],origin='synthetic_literal',width_identity='diagnostic')


def reference(x,s,p,r,mask):
    n=len(x);center=sum(x)/n
    ds=[[torch.linalg.vector_norm(x[j]-x[i]) for j in range(n)] for i in range(n)]
    gs=[[sum((x[i,k]-center[k])*(x[j,k]-center[k]) for k in range(3)) for j in range(n)] for i in range(n)]
    f=torch.stack([torch.stack((sum(ds[i])/n,sum(d*d for d in ds[i])/n,sum(g*g for g in gs[i])/n)) for i in range(n)])
    h=f@p.finite.weight.T+p.finite.bias+s@p.sequence.weight.T
    w=r.coefficients.weight[0];width=s.shape[1]
    v=torch.stack([sum((w[0]*ds[i][j]+w[1]*gs[i][j]+sum(w[2+k]*h[i,k]+w[2+width+k]*h[j,k] for k in range(width)))*(x[j]-x[i]) for j in range(n))/n for i in range(n)])
    return torch.where(mask[:,None],v,torch.zeros_like(v))


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('width',[2,8])
def test_independent_ragged_equations_and_all_gradients(device,dtype,width):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    a,p,r,s,axis=setup(device,dtype,width)
    out=call(a,p,r,s,axis)
    refs=[reference(x,q,p,r,mask) for x,q,mask in zip(a[0],s,a[7])]
    expected=sum(((v[mask]-u.detach()[mask])**2).mean() for v,u,mask in zip(refs,a[6],a[7]))/2
    tol=3e-4 if dtype==torch.float32 else 1e-9
    torch.testing.assert_close(out.loss,expected,rtol=tol,atol=tol)
    operands=tuple(a[0]+s)+tuple(p.parameters())+tuple(r.parameters())
    ag=torch.autograd.grad(out.loss,operands,retain_graph=True)
    eg=torch.autograd.grad(expected,operands,retain_graph=True)
    for g,e in zip(ag,eg):torch.testing.assert_close(g,e,rtol=tol,atol=tol)
    assert torch.autograd.grad(out.loss,tuple(a[6]),allow_unused=True)==(None,None)
    assert sum(t.numel() for t in p.parameters())+sum(t.numel() for t in r.parameters())==width*width+6*width+2
    for name in ('require_source_use','require_model_use','require_training_use'):
        with pytest.raises(ValueError):getattr(out,name)()
    # Rotate/reflection and translation: transform only current XYZ and velocity targets.
    q=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,-1.]],device=device,dtype=dtype)
    aa=list(a);aa[0]=[x@q.T+2 for x in a[0]];aa[6]=[u@q.T for u in a[6]]
    torch.testing.assert_close(call(aa,p,r,s,axis).loss,out.loss,rtol=tol,atol=tol)
    from mora3.data.geometry_frame_policy import axis_digest
    aa=list(a);aa[0]=[x.flip(0) for x in a[0]];aa[6]=[u.flip(0) for u in a[6]];aa[7]=[m.flip(0) for m in a[7]]
    aa[4]=[ax[::-1] for ax in a[4]]
    aa[3]=[replace(b,native_axis=ax,native_axis_sha256=axis_digest(ax),free_atoms=b.free_atoms[::-1]) for b,ax in zip(a[3],aa[4])]
    torch.testing.assert_close(call(aa,p,r,[t.flip(0) for t in s],axis).loss,out.loss,rtol=tol,atol=tol)


@pytest.mark.parametrize('bad',['sequence_nan','sequence_shape','sequence_dtype','xyz','target','axis','mask','binding','length','width_axis','width_identity','finite_axis','device_dtype','all_inactive'])
def test_late_invalid_case_refuses_before_either_learned_operator(bad):
    a,p,r,s,axis=setup();calls=[]
    for mod in (p.finite,p.sequence,r.coefficients):mod.register_forward_pre_hook(lambda *args:calls.append(1))
    if bad=='sequence_nan':s[1]=torch.full_like(s[1],float('nan'))
    if bad=='sequence_shape':s[1]=s[1][:,:1]
    if bad=='sequence_dtype':s[1]=s[1].float()
    if bad=='xyz':a[0][1]=torch.full_like(a[0][1],float('nan'))
    if bad=='target':a[6][1]=torch.full_like(a[6][1],float('nan'))
    if bad=='axis':a[4][1]=a[4][1][::-1]
    if bad=='mask':a[7][1]=torch.tensor([False,True])
    if bad=='binding':a[3][1]=replace(a[3][1],xyz_available=(True,False))
    if bad=='length':s=[]
    if bad=='width_axis':axis=axis[::-1]
    if bad=='width_identity':p._width_identity='wrong'
    if bad=='finite_axis':p._finite_axis=SYNTHETIC_SUMMARY_AXIS[::-1]
    if bad=='device_dtype':r.float()
    if bad=='all_inactive':a[-1]=(False,False)
    rng=torch.random.get_rng_state().clone()
    with pytest.raises(ValueError):call(a,p,r,s,axis)
    assert calls==[] and torch.equal(rng,torch.random.get_rng_state())


def test_inactive_opaque_fixed_target_nan_and_no_target_argument():
    a,p,r,s,axis=setup();a[-1]=(True,False)
    for i in (0,3,4,6):a[i][1]=object()
    s[1]=object();a[6][0]=a[6][0].detach().clone();a[6][0][1]=float('nan')
    out=call(a,p,r,s,axis);assert out.active_cases==(0,)
    with pytest.raises(TypeError):api()(target_xyz=a[0])
