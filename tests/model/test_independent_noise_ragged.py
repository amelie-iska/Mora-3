"""Equal case weighting and complete preflight for generated-noise diagnostics."""
from dataclasses import replace
import pytest
import torch
from test_independent_noise_vector import setup
from test_independent_current_noise import args
from mora3.data.independent_current_noise import construct_independent_current,replay_independent_current
from mora3.data.geometry_frame_policy import axis_digest
import mora3.model.current_frame_velocity as mod


def inputs(device='cpu',dtype=torch.float64):
    a=setup(device,dtype);q=args();b=q['binding'];axis=b.native_axis[:2]
    q['binding']=replace(b,native_axis=axis,native_axis_sha256=axis_digest(axis),
        xyz_available=(True,True),free_atoms=(True,True))
    q['seed']=34101
    r=construct_independent_current(**q)
    c=dict(a,artifact=r,current_xyz=replay_independent_current(r,device=device,dtype=dtype).requires_grad_(),
        binding=q['binding'],atom_axis=axis,free_atoms=(True,True))
    return [a,c], [torch.arange(n*3,device=device,dtype=dtype).reshape(n,3).requires_grad_() for n in (3,2)]


def run(cases,targets,ready=(True,True)):
    assert hasattr(mod,'independent_noise_current_vector_loss'),'missing replay-verified ragged noise loss'
    return mod.independent_noise_current_vector_loss(cases,targets,ready)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_independent_equations_gradients_and_equal_case_weights(device,dtype):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    cases,targets=inputs(device,dtype);out=run(cases,targets);terms=[]
    for c,u in zip(cases,targets):
        x=c['current_xyz'];w=c['readout'].coefficients.weight;z=x-x.mean(0);n=len(x)
        v=torch.stack([sum((w[0,0]*(x[j]-x[i]).norm()+w[0,1]*(z[i]*z[j]).sum())*(x[j]-x[i]) for j in range(n))/n for i in range(n)])
        terms.append((v-u.detach()).square().mean())
    ref=sum(terms)/2
    torch.testing.assert_close(out.loss,ref)
    operands=tuple(c['current_xyz'] for c in cases)+(cases[0]['readout'].coefficients.weight,)
    for a,b in zip(torch.autograd.grad(out.loss,operands,retain_graph=True),torch.autograd.grad(ref,operands)):
        torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-5)
    assert torch.autograd.grad(out.loss,targets,allow_unused=True)==(None,None)
    assert out.active_cases==(0,1)
    assert not torch.isclose(ref,(terms[0]*3+terms[1]*2)/5)
    for name in ('require_source_use','require_model_use','require_training_use'):
        with pytest.raises(ValueError):getattr(out,name)()


@pytest.mark.parametrize('bad',['target','receipt','xyz','axis','context','extra','length','ready','owner'])
def test_late_invalid_case_refuses_before_any_learned_call(bad):
    cases,targets=inputs();ready=(True,True);calls=[]
    hook=cases[0]['readout'].coefficients.register_forward_pre_hook(lambda *a:calls.append(1))
    if bad=='target':targets[1]=torch.full((2,3),float('nan'),dtype=torch.float64)
    if bad=='receipt':cases[1]['artifact']=replace(cases[1]['artifact'],xyz_bytes=b'bad')
    if bad=='xyz':cases[1]['current_xyz']=cases[1]['current_xyz']+1
    if bad=='axis':cases[1]['atom_axis']=cases[1]['atom_axis'][::-1]
    if bad=='context':cases[1]['context_sha256']='a'*64
    if bad=='extra':cases[1]['target_xyz']=targets[1]
    if bad=='length':targets.pop()
    if bad=='ready':ready=(True,1)
    if bad=='owner':cases[1]['readout']=object()
    with pytest.raises(ValueError):run(cases,targets,ready)
    hook.remove();assert calls==[]


def test_inactive_payload_opaque_and_targets_do_not_condition_predictions():
    cases,targets=inputs();a=run(cases,targets)
    cases[1]=object();targets[1]=object()
    b=run(cases,targets,(True,False));assert b.active_cases==(0,)
    assert torch.isfinite(a.loss) and torch.isfinite(b.loss)
