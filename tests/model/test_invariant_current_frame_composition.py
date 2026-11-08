"""End-to-end synthetic internal-feature vector equation and covariance."""
import importlib
import pytest
import torch
from test_ragged_current_frame_velocity import inputs
from mora3.model.current_frame_velocity import CurrentFrameVelocityReadout


def call(a):
    module=importlib.import_module('mora3.model.current_frame_velocity')
    assert hasattr(module,'invariant_current_frame_velocity_loss'), 'missing invariant current-frame composition'
    return module.invariant_current_frame_velocity_loss(a[0],a[2],a[3],a[4],a[6],a[7],a[8],origin='synthetic_literal')


def setup(device='cpu',dtype=torch.float64):
    a=inputs(device,dtype)
    a[2]=CurrentFrameVelocityReadout(('distance','centered_dot'),max_atoms=8).to(device=device,dtype=dtype)
    return a


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_independent_equation_gradients_and_rigid_motion(device,dtype):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    a=setup(device,dtype);out=call(a);refs=[]
    for x,u,mask in zip(a[0],a[6],a[7]):
        center=sum(x)/len(x);rows=[]
        for i in range(len(x)):
            terms=[]
            for j in range(len(x)):
                delta=x[j]-x[i]
                dist=torch.linalg.vector_norm(delta)
                dot=sum((x[i,k]-center[k])*(x[j,k]-center[k]) for k in range(3))
                coef=a[2].coefficients.weight[0,0]*dist+a[2].coefficients.weight[0,1]*dot
                terms.append(coef*delta)
            rows.append(sum(terms)/len(x))
        v=torch.stack(rows);refs.append(((v[mask]-u.detach()[mask])**2).mean())
    ref=sum(refs)/len(refs);torch.testing.assert_close(out.loss,ref)
    operands=tuple(a[0])+tuple(a[2].parameters())
    g=torch.autograd.grad(out.loss,operands,retain_graph=True);r=torch.autograd.grad(ref,operands)
    for x,y in zip(g,r):torch.testing.assert_close(x,y,rtol=2e-4,atol=2e-5)
    q=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,-1.]],device=device,dtype=dtype)
    b=list(a);b[0]=[x@q+3 for x in a[0]];b[6]=[u@q for u in a[6]]
    torch.testing.assert_close(call(b).loss,out.loss,rtol=2e-4,atol=2e-5)
    assert all(u.grad is None for u in a[6])
    for name in ('require_source_use','require_model_use','require_training_use'):
        with pytest.raises(ValueError):getattr(out,name)()


@pytest.mark.parametrize('bad',['axis','xyz','target','feature'])
def test_invalid_second_case_precedes_any_learned_work(bad):
    a=setup();calls=[];a[2].coefficients.register_forward_pre_hook(lambda *args:calls.append(1))
    if bad=='axis':a[4][1]=a[4][1][::-1]
    if bad=='xyz':a[0][1]=torch.full((2,3),float('nan'),dtype=torch.float64)
    if bad=='target':a[6][1]=torch.full((2,3),float('nan'),dtype=torch.float64)
    if bad=='feature':a[2]=CurrentFrameVelocityReadout(('other',),max_atoms=8).double();a[2].coefficients.register_forward_pre_hook(lambda *args:calls.append(1))
    with pytest.raises(ValueError):call(a)
    assert not calls


def test_inactive_payload_opaque_and_no_target_coordinate_api():
    a=setup();a[-1]=(True,False)
    for i in (0,3,4,6):a[i][1]=object()
    assert call(a).active_cases==(0,)
