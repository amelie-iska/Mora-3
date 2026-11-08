"""Synthetic context coefficient witnesses: dropping h_j must break these tests."""
import importlib
from dataclasses import replace
import pytest
import torch
from test_ragged_current_frame_velocity import inputs


def module():
    m = importlib.import_module('mora3.model.current_frame_velocity')
    assert hasattr(m, 'ContextualCurrentFrameVelocityReadout'), 'missing atom-context vector readout'
    assert hasattr(m, 'contextual_current_frame_velocity_loss'), 'missing atom-context loss composition'
    return m


def setup(device='cpu', dtype=torch.float64):
    a = inputs(device, dtype)
    torch.manual_seed(33001)
    m = module().ContextualCurrentFrameVelocityReadout(('h0', 'h1'), max_atoms=8).to(device=device, dtype=dtype)
    hs = [torch.randn(len(x), 2, device=device, dtype=dtype, requires_grad=True) for x in a[0]]
    return a, m, hs


def call(a, m, hs, axis=('h0', 'h1')):
    return module().contextual_current_frame_velocity_loss(
        a[0], hs, m, a[3], a[4], axis, a[6], a[7], a[8], origin='synthetic_literal')


def bound(m, x, h, b, axis):
    return m.forward_bound_diagnostic(x, h, binding=b, atom_axis=axis, context_axis=('h0', 'h1'))


def reference(x, h, w, mask):
    center = sum(x)/len(x)
    rows = []
    for i in range(len(x)):
        terms = []
        for j in range(len(x)):
            d = x[j]-x[i]
            c = (w[0, 0]*torch.linalg.vector_norm(d)
                 + w[0, 1]*sum((x[i,k]-center[k])*(x[j,k]-center[k]) for k in range(3))
                 + sum(w[0,2+k]*h[i,k]+w[0,4+k]*h[j,k] for k in range(2)))
            terms.append(c*d)
        rows.append(sum(terms)/len(x) if mask[i] else x[i]*0)
    return torch.stack(rows)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_values_all_input_parameter_gradients_symmetry_and_target_exclusion(device, dtype):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    a, m, hs = setup(device, dtype)
    out = call(a, m, hs)
    refs = [reference(x,h,m.coefficients.weight,mask) for x,h,mask in zip(a[0],hs,a[7])]
    expected = sum(((v[mask]-u.detach()[mask])**2).mean() for v,u,mask in zip(refs,a[6],a[7]))/2
    tol = 3e-5 if dtype==torch.float32 else 1e-10
    torch.testing.assert_close(out.loss,expected,rtol=tol,atol=tol)
    operands=tuple(a[0]+hs)+tuple(m.parameters())
    ag=torch.autograd.grad(out.loss,operands,retain_graph=True)
    rg=torch.autograd.grad(expected,operands,retain_graph=True)
    for g,r in zip(ag,rg): torch.testing.assert_close(g,r,rtol=tol,atol=tol)
    assert ag[2][1].abs().sum()>0  # fixed atom context still contributes to free output
    assert torch.autograd.grad(out.loss,tuple(a[6]),allow_unused=True)==(None,None)
    q=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,-1.]],device=device,dtype=dtype)
    for x,h,b,axis,v in zip(a[0],hs,a[3],a[4],refs):
        actual=bound(m,x,h,b,axis)
        torch.testing.assert_close(actual,v,rtol=tol,atol=tol)
        torch.testing.assert_close(bound(m,x@q.T+3,h,b,axis),v@q.T,rtol=tol,atol=tol)
        p=torch.arange(len(x)-1,-1,-1,device=device)
        order=tuple(reversed(range(len(x))))
        new_axis=tuple(axis[i] for i in order)
        from mora3.data.geometry_frame_policy import axis_digest
        pb=replace(b,native_axis=new_axis,native_axis_sha256=axis_digest(new_axis),free_atoms=tuple(b.free_atoms[i] for i in order))
        torch.testing.assert_close(bound(m,x[p],h[p],pb,new_axis),v[p],rtol=tol,atol=tol)
    assert sum(p.numel() for p in m.parameters())==6
    for method in ('require_source_use','require_model_use','require_training_use'):
        with pytest.raises(ValueError): getattr(m,method)()
        with pytest.raises(ValueError): getattr(out,method)()


@pytest.mark.parametrize('bad',['context_nan','context_shape','context_dtype','xyz_nan','target_nan','atom_order','context_order','mask','binding','length','inactive_all'])
def test_all_active_preflight_before_any_learned_call(bad):
    a,m,hs=setup();calls=[]
    m.coefficients.register_forward_pre_hook(lambda *args:calls.append(1))
    axis=('h0','h1')
    if bad=='context_nan': hs[1]=torch.full((2,2),float('nan'),dtype=torch.float64)
    if bad=='context_shape': hs[1]=hs[1][:,:1]
    if bad=='context_dtype': hs[1]=hs[1].float()
    if bad=='xyz_nan': a[0][1]=torch.full((2,3),float('nan'),dtype=torch.float64)
    if bad=='target_nan': a[6][1]=torch.full((2,3),float('nan'),dtype=torch.float64)
    if bad=='atom_order': a[4][1]=a[4][1][::-1]
    if bad=='context_order': axis=axis[::-1]
    if bad=='mask': a[7][1]=torch.tensor([False,True])
    if bad=='binding': a[3][1]=replace(a[3][1],xyz_available=(False,True))
    if bad=='length': hs=[]
    if bad=='inactive_all': a[-1]=(False,False)
    rng=torch.random.get_rng_state().clone()
    with pytest.raises(ValueError): call(a,m,hs,axis)
    assert calls==[] and torch.equal(rng,torch.random.get_rng_state())


def test_context_sensitivity_fixed_inputs_opaque_cases_and_checkpoint_version():
    a,m,hs=setup();v=bound(m,a[0][0],hs[0],a[3][0],a[4][0])
    changed=hs[0].detach().clone();changed[1]+=3
    assert not torch.allclose(v,bound(m,a[0][0],changed,a[3][0],a[4][0]))
    assert torch.equal(v[~a[7][0]],torch.zeros_like(v[~a[7][0]]))
    a[-1]=(True,False)
    for i in (0,3,4,6): a[i][1]=object()
    hs[1]=object()
    a[6][0]=a[6][0].detach().clone();a[6][0][1]=float('nan')
    assert call(a,m,hs).active_cases==(0,)
    same=module().ContextualCurrentFrameVelocityReadout(('h0','h1'),max_atoms=8).double()
    same.load_state_dict(m.state_dict())
    torch.testing.assert_close(bound(same,a[0][0],hs[0],a[3][0],a[4][0]),v)
    other=module().ContextualCurrentFrameVelocityReadout(('h1','h0'),max_atoms=8).double()
    with pytest.raises(ValueError,match='configuration'): other.load_state_dict(m.state_dict())
    from mora3.model.current_frame_velocity import CurrentFrameVelocityReadout
    legacy=CurrentFrameVelocityReadout(('distance','centered_dot'),max_atoms=8).double()
    with pytest.raises((ValueError,RuntimeError)): same.load_state_dict(legacy.state_dict())
    with pytest.raises(TypeError): m.forward_bound_diagnostic(a[0][0],hs[0],binding=a[3][0],atom_axis=a[4][0],context_axis=('h0','h1'),target_xyz=a[0][0])


@pytest.mark.parametrize('axis,bound_value',[((),8),(('h','h'),8),(['h'],8),((' ',),8),(('h',),True),(('h',),257),(('h',),0)])
def test_constructor_refuses_incomplete_or_unbounded_context(axis,bound_value):
    with pytest.raises(ValueError): module().ContextualCurrentFrameVelocityReadout(axis,max_atoms=bound_value)
