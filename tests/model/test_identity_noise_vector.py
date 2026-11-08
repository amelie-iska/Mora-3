"""Native identity declarations never become source/model authority."""
from dataclasses import replace, asdict
import hashlib
import importlib
import json
import pytest
import torch
from test_independent_current_noise import args
from mora3.data.native_atom_context import native_atom_context
from mora3.data.geometry_frame_policy import axis_digest
from mora3.data.independent_current_noise import construct_independent_current, replay_independent_current
from mora3.model.current_frame_velocity import SplitContextualCoefficients

AXIS=tuple(f'element:{i}' for i in range(1,119))+tuple('missing:'+s for s in ('electronic_charge','spin_multiplicity','enzyme','cellular_assay','solvent'))

def setup(device='cpu',dtype=torch.float64):
    a=args();b=a['binding']
    axis=tuple(replace(k,element=e) for k,e in zip(b.native_axis,('H','C','Og')))
    b=replace(b,native_axis=axis,native_axis_sha256=axis_digest(axis));a['binding']=b
    c=native_atom_context(dict(source_record_id='diagnostic:identity',source_reaction_group='diagnostic:group',source_native_split='val',atomic_numbers=[1,6,118],electronic_charge=None,spin_multiplicity=None,electronic_charge_known=False,spin_multiplicity_known=False,task_admitted=False),ordered_atom_ids=tuple(k.name for k in b.native_axis),source_payload_sha256='a'*64)
    digest=hashlib.sha256(json.dumps(asdict(c),sort_keys=True,separators=(',',':')).encode()).hexdigest()
    a.update(context_sha256=digest,expected_context_sha256=digest)
    artifact=construct_independent_current(**a)
    x=replay_independent_current(artifact,device=device,dtype=dtype).requires_grad_()
    coeff=SplitContextualCoefficients(AXIS,max_atoms=8).to(device=device,dtype=dtype)
    with torch.no_grad():
        for p in coeff.parameters():p.copy_(torch.arange(p.numel(),device=device,dtype=dtype).reshape(p.shape)/311+.01)
    return dict(atom_context=c,artifact=artifact,current_xyz=x,coefficients=coeff,binding=b,atom_axis=b.native_axis,free_atoms=b.free_atoms,width_identity='diagnostic-identity-123')

def run(a):
    m=importlib.import_module('mora3.model.identity_noise_vector')
    return m.identity_noise_vector_diagnostic(**a)

@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_independent_equation_and_xyz_all_parameter_gradients(device,dtype):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    a=setup(device,dtype);x=a['current_xyz'];c=a['coefficients'];before=a['binding']
    cpu=torch.random.get_rng_state().clone();gpu=torch.cuda.get_rng_state_all()
    v=run(a);z=x-x.mean(0)
    # Independent full element axis plus separate missingness channels.
    h=torch.tensor([[int(j==n) for j in range(1,119)]+[1]*5 for n in [1,6,118]],device=device,dtype=dtype)
    ref=torch.stack([sum((c.geometry.weight[0,0]*(x[j]-x[i]).norm()+c.geometry.weight[0,1]*(z[i]*z[j]).sum()+(h[i]*c.receiver.weight[0]).sum()+(h[j]*c.neighbor.weight[0]).sum())*(x[j]-x[i]) for j in range(3))/3 for i in range(3)])
    torch.testing.assert_close(v,ref)
    operands=(x,*c.parameters());probe=torch.arange(9,device=device,dtype=dtype).reshape(3,3)
    for l,r in zip(torch.autograd.grad((v*probe).sum(),operands,retain_graph=True),torch.autograd.grad((ref*probe).sum(),operands)):
        torch.testing.assert_close(l,r,rtol=2e-4,atol=2e-4 if dtype==torch.float32 else 1e-9)
    assert a['binding'] is before and not before.synthetic_diagnostic
    assert torch.equal(cpu,torch.random.get_rng_state())
    assert all(torch.equal(l,r) for l,r in zip(gpu,torch.cuda.get_rng_state_all()))
    for obj in (a['atom_context'],c, a['artifact'].metadata):
        with pytest.raises(ValueError):obj.require_model_use()

@pytest.mark.parametrize('bad',['axis','split','identity','source','mask','version','missing','xyz','free','receipt','owner','channels','target'])
def test_refusal_before_learned_calls(bad):
    a=setup();c=a['coefficients'];ctx=a['atom_context'];calls=[]
    hooks=[m.register_forward_pre_hook(lambda *unused:calls.append(1)) for m in (c.geometry,c.receiver,c.neighbor)]
    if bad=='axis':a['atom_context']=replace(ctx,ordered_atom_ids=ctx.ordered_atom_ids[::-1])
    if bad=='split':a['atom_context']=replace(ctx,source_native_split='test')
    if bad=='identity':a['atom_context']=replace(ctx,atomic_numbers=(2,6,118))
    if bad=='source':a['atom_context']=replace(ctx,source_payload_sha256='b'*64)
    if bad=='mask':a['atom_context']=replace(ctx,missing_context=('enzyme',))
    if bad=='version':a['atom_context']=replace(ctx,version='future')
    if bad=='missing':a['atom_context']=replace(ctx,electronic_charge=0)
    if bad=='xyz':a['current_xyz']=a['current_xyz']+1
    if bad=='free':a['free_atoms']=(False,True,True)
    if bad=='receipt':a['artifact']=replace(a['artifact'],xyz_bytes=b'bad')
    if bad=='owner':c.receiver=torch.nn.Linear(123,1)
    if bad=='channels':c._context_axis=c._context_axis[::-1]
    if bad=='target':a['target_xyz']=torch.zeros(3,3)
    with pytest.raises((ValueError,TypeError)):run(a)
    for hook in hooks:hook.remove()
    assert calls==[]
