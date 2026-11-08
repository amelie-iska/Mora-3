"""Noise-to-vector composition cannot accept gold or stale current coordinates."""
from dataclasses import replace
import importlib
import pytest
import torch
from test_independent_current_noise import args
from mora3.data.independent_current_noise import construct_independent_current,replay_independent_current
from mora3.model.current_frame_velocity import CurrentFrameVelocityReadout


def setup(device='cpu',dtype=torch.float64):
    a=args();a['seed']=34001
    artifact=construct_independent_current(**a)
    x=replay_independent_current(artifact,device=device,dtype=dtype).requires_grad_()
    m=CurrentFrameVelocityReadout(('distance','centered_dot'),max_atoms=8).to(device=device,dtype=dtype)
    with torch.no_grad():m.coefficients.weight.copy_(torch.tensor([[.2,-.3]],device=device,dtype=dtype))
    return dict(artifact=artifact,current_xyz=x,readout=m,binding=a['binding'],
        atom_axis=a['binding'].native_axis,context_sha256=a['context_sha256'],free_atoms=(True,)*3)


def run(a):
    mod=importlib.import_module('mora3.model.current_frame_velocity')
    assert hasattr(mod,'independent_noise_current_vector_diagnostic')
    return mod.independent_noise_current_vector_diagnostic(**a)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_exact_noise_vector_equation_and_input_parameter_gradients(device,dtype):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('no CUDA')
    a=setup(device,dtype);x=a['current_xyz'];m=a['readout']
    cpu=torch.random.get_rng_state().clone();gpu=torch.cuda.get_rng_state_all()
    v=run(a);center=x-x.mean(0)
    ref=torch.stack([sum((m.coefficients.weight[0,0]*torch.linalg.vector_norm(x[j]-x[i])+
        m.coefficients.weight[0,1]*(center[i]*center[j]).sum())*(x[j]-x[i]) for j in range(3))/3 for i in range(3)])
    torch.testing.assert_close(v,ref)
    probe=torch.arange(9,device=device,dtype=dtype).reshape(3,3)
    for actual,expected in zip(torch.autograd.grad((v*probe).sum(),(x,m.coefficients.weight),retain_graph=True),
            torch.autograd.grad((ref*probe).sum(),(x,m.coefficients.weight))):
        torch.testing.assert_close(actual,expected,atol=2e-5 if dtype==torch.float32 else 1e-10,rtol=2e-5 if dtype==torch.float32 else 1e-10)
    assert torch.equal(cpu,torch.random.get_rng_state())
    assert all(torch.equal(l,r) for l,r in zip(gpu,torch.cuda.get_rng_state_all()))
    for name in ('require_source_use','require_model_use','require_training_use'):
        with pytest.raises(ValueError):getattr(m,name)()


@pytest.mark.parametrize('bad',['xyz','axis','context','mask','receipt','binding','feature','bound','weight','owner'])
def test_refusal_precedes_learned_calls(bad):
    a=setup();m=a['readout'];calls=[]
    hook=m.coefficients.register_forward_pre_hook(lambda *unused:calls.append(1))
    if bad=='xyz':a['current_xyz']=a['current_xyz']+1
    if bad=='axis':a['atom_axis']=a['atom_axis'][::-1]
    if bad=='context':a['context_sha256']='a'*64
    if bad=='mask':a['free_atoms']=(True,False,True)
    if bad=='receipt':a['artifact']=replace(a['artifact'],xyz_bytes=b'wrong')
    if bad=='binding':a['binding']=replace(a['binding'],free_atoms=(False,)*3)
    if bad=='feature':m._feature_axis=('wrong','axis')
    if bad=='bound':m._max_atoms=2
    if bad=='weight':
        with torch.no_grad():m.coefficients.weight.fill_(float('nan'))
    if bad=='owner':m.coefficients=torch.nn.Linear(2,1)
    with pytest.raises(ValueError):run(a)
    hook.remove();assert calls==[]


def test_targets_are_not_an_input_and_original_unqualified_binding_is_preserved():
    a=setup();before=a['binding'].current.source
    run(a)
    assert a['binding'].current.source==before and not a['binding'].synthetic_diagnostic
    with pytest.raises(TypeError):run(dict(a,target_xyz=torch.zeros(3,3)))


def test_malformed_artifact_metadata_refuses_before_learned_work():
    a=setup();a['artifact']=replace(a['artifact'],metadata=object())
    calls=[];h=a['readout'].coefficients.register_forward_pre_hook(lambda *unused:calls.append(1))
    with pytest.raises(ValueError):run(a)
    h.remove();assert calls==[]
