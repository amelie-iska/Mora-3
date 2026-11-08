"""Independent vector/loss references; synthetic diagnostics only."""
import importlib
from dataclasses import replace
import pytest
import torch
from mora3.model.current_frame_velocity import CurrentFrameVelocityReadout
from test_current_frame_request_binding import fixture, bind


def compose(*args):
    m=importlib.import_module('mora3.model.current_frame_velocity')
    assert hasattr(m,'bound_current_frame_velocity_loss'), 'missing bound ragged vector composition'
    return m.bound_current_frame_velocity_loss(*args)


def inputs(device='cpu',dtype=torch.float64):
    torch.manual_seed(32701)
    m=CurrentFrameVelocityReadout(('scalar',),max_atoms=8).to(device=device,dtype=dtype)
    configs=[fixture(),fixture()]
    c=configs[1];c['native_axis']=c['native_axis'][:2];c['native_axis_sha256']=__import__('mora3.data.geometry_frame_policy',fromlist=['axis_digest']).axis_digest(c['native_axis'])
    c['xyz_available']=(True,True);c['free_atoms']=(True,True)
    bs=[bind(c) for c in configs]
    xs=[torch.randn(n,3,device=device,dtype=dtype,requires_grad=True) for n in (3,2)]
    fs=[torch.randn(n,n,1,device=device,dtype=dtype,requires_grad=True) for n in (3,2)]
    us=[torch.randn_like(x,requires_grad=True) for x in xs]
    masks=[torch.tensor(b.free_atoms,device=device) for b in bs]
    return [xs,fs,m,bs,[b.native_axis for b in bs],('scalar',),us,masks,(True,True)]


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_scalar_vector_loss_and_all_predictor_gradients(device,dtype):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    a=inputs(device,dtype);out=compose(*a)
    xs,fs,m,bs,_,_,us,masks,_=a
    refs=[]
    for x,f,u,mask in zip(xs,fs,us,masks):
        rows=[]
        for i in range(len(x)):
            rows.append(torch.stack([sum(m.coefficients.weight[0,0]*f[i,j,0]*(x[j,k]-x[i,k]) for j in range(len(x)))/len(x) for k in range(3)]) if mask[i] else x[i]*0)
        v=torch.stack(rows);refs.append(((v[mask]-u.detach()[mask])**2).mean())
    ref=sum(refs)/2
    torch.testing.assert_close(out.loss,ref)
    operands=tuple(xs+fs)+tuple(m.parameters())
    actual=torch.autograd.grad(out.loss,operands,retain_graph=True)
    expected=torch.autograd.grad(ref,operands)
    for x,y in zip(actual,expected):torch.testing.assert_close(x,y,rtol=1e-4,atol=1e-5)
    assert all(u.grad is None for u in us)
    assert out.active_cases==(0,1)
    for method in ('require_source_use','require_model_use','require_training_use'):
        with pytest.raises(ValueError):getattr(out,method)()


@pytest.mark.parametrize('bad',['binding','axis','feature','mask','target','xyz','features','length','empty'])
def test_all_active_preflight_before_learned_work(bad):
    a=inputs();calls=[];h=a[2].coefficients.register_forward_pre_hook(lambda *x:calls.append(1))
    if bad=='binding':a[3][1]=replace(a[3][1],xyz_available=(True,False))
    if bad=='axis':a[4][1]=a[4][1][::-1]
    if bad=='feature':a[5]=('other',)
    if bad=='mask':a[7][1]=torch.tensor([True,False])
    if bad=='target':a[6][1]=torch.full((2,3),float('nan'),dtype=torch.float64)
    if bad=='xyz':a[0][1]=torch.zeros(1,3,dtype=torch.float64)
    if bad=='features':a[1][1]=torch.full((2,2,1),float('nan'),dtype=torch.float64)
    if bad=='length':a[3]=[]
    if bad=='empty':a[-1]=(False,False)
    before=torch.random.get_rng_state().clone()
    with pytest.raises(ValueError):compose(*a)
    h.remove();assert calls==[] and torch.equal(before,torch.random.get_rng_state())


def test_inactive_opaque_fixed_target_nan_and_target_exclusion():
    a=inputs();a[-1]=(True,False)
    for index in (0,1,3,4,6):a[index][1]=object()
    a[6][0]=a[6][0].detach().clone();a[6][0][1]=float('nan')
    out=compose(*a);assert out.active_cases==(0,) and torch.isfinite(out.loss)
    with pytest.raises(TypeError):
        importlib.import_module('mora3.model.current_frame_velocity').bound_current_frame_velocity_loss(*a,target_xyz=torch.zeros(3,3))
