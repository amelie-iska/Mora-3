"""Equal-case identity/noise loss and all-active refusal before learned work."""
from dataclasses import replace, asdict
import hashlib
import json
import pytest
import torch
from test_identity_noise_vector import setup
from test_independent_current_noise import args
from mora3.data.geometry_frame_policy import axis_digest
from mora3.data.independent_current_noise import construct_independent_current, replay_independent_current
import mora3.model.identity_noise_vector as mod


def cases(device='cpu', dtype=torch.float64):
    a=setup(device,dtype); b=setup(device,dtype)
    axis=b['atom_axis'][:2]
    binding=replace(b['binding'],native_axis=axis,native_axis_sha256=axis_digest(axis),xyz_available=(True,True),free_atoms=(True,True))
    context=replace(b['atom_context'],ordered_atom_ids=b['atom_context'].ordered_atom_ids[:2],atomic_numbers=(1,6))
    digest=hashlib.sha256(json.dumps(asdict(context),sort_keys=True,separators=(',',':')).encode()).hexdigest()
    q=args();q.update(binding=binding,context_sha256=digest,expected_context_sha256=digest,seed=34801)
    artifact=construct_independent_current(**q)
    b.update(atom_context=context,binding=binding,atom_axis=axis,free_atoms=binding.free_atoms,artifact=artifact,current_xyz=replay_independent_current(artifact,device=device,dtype=dtype).requires_grad_())
    targets=[torch.arange(n*3,device=device,dtype=dtype).reshape(n,3).requires_grad_() for n in (3,2)]
    return [a,b],targets


def run(c,t,r=(True,True)):
    assert hasattr(mod,'identity_noise_vector_loss'),'missing identity/noise ragged loss'
    return mod.identity_noise_vector_loss(c,t,r)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_independent_equal_case_equations_and_all_gradients(device,dtype):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    c,t=cases(device,dtype);rng=torch.random.get_rng_state().clone();gpu=torch.cuda.get_rng_state_all()
    result=run(c,t);terms=[];operands=[]
    for a,u in zip(c,t):
        x=a['current_xyz'];p=a['coefficients'];z=x-x.mean(0);n=len(x)
        h=torch.tensor([[int(j==v) for j in range(1,119)]+[1]*5 for v in a['atom_context'].atomic_numbers],device=device,dtype=dtype)
        v=torch.stack([sum((p.geometry.weight[0,0]*(x[j]-x[i]).norm()+p.geometry.weight[0,1]*(z[i]*z[j]).sum()+(h[i]*p.receiver.weight[0]).sum()+(h[j]*p.neighbor.weight[0]).sum())*(x[j]-x[i]) for j in range(n))/n for i in range(n)])
        terms.append((v-u.detach()).square().mean());operands.extend((x,*p.parameters()))
    ref=sum(terms)/2
    torch.testing.assert_close(result.loss,ref)
    for l,r in zip(torch.autograd.grad(result.loss,operands,retain_graph=True),torch.autograd.grad(ref,operands)):
        torch.testing.assert_close(l,r,rtol=3e-4,atol=3e-4 if dtype==torch.float32 else 1e-8)
    assert torch.autograd.grad(result.loss,t,allow_unused=True)==(None,None)
    assert result.active_cases==(0,1)
    assert not torch.isclose(ref,(3*terms[0]+2*terms[1])/5)
    assert torch.equal(rng,torch.random.get_rng_state())
    assert all(torch.equal(l,r) for l,r in zip(gpu,torch.cuda.get_rng_state_all()))
    for name in ('require_source_use','require_model_use','require_training_use'):
        with pytest.raises(ValueError):getattr(result,name)()


@pytest.mark.parametrize('bad',['context','axis','xyz','receipt','target','extra','owner','ready','length','dtype','container'])
def test_late_refusal_before_any_learned_work(bad):
    c,t=cases();r=(True,True);calls=[]
    hooks=[m.register_forward_pre_hook(lambda *a:calls.append(1)) for a in c for m in a['coefficients'].modules()]
    if bad=='context':c[1]['atom_context']=replace(c[1]['atom_context'],atomic_numbers=(2,6))
    if bad=='axis':c[1]['atom_axis']=c[1]['atom_axis'][::-1]
    if bad=='xyz':c[1]['current_xyz']=c[1]['current_xyz']+1
    if bad=='receipt':c[1]['artifact']=replace(c[1]['artifact'],xyz_bytes=b'bad')
    if bad=='target':t[1]=t[1]*float('nan')
    if bad=='extra':c[1]['target_xyz']=t[1]
    if bad=='owner':c[1]['coefficients']=object()
    if bad=='ready':r=(True,1)
    if bad=='length':t.pop()
    if bad=='dtype':t[1]=t[1].float()
    if bad=='container':c[1]=object()
    with pytest.raises(ValueError):run(c,t,r)
    for h in hooks:h.remove()
    assert calls==[]


def test_inactive_payload_stays_opaque_and_no_active_refuses():
    c,t=cases();c[1]=object();t[1]=object()
    assert run(c,t,(True,False)).active_cases==(0,)
    with pytest.raises(ValueError):run(c,t,(False,False))
