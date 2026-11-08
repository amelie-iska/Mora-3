"""Fresh retained-source verification for every active diagnostic case."""
from dataclasses import replace
from mora3.data.geometry_frame_policy import axis_digest
import shutil
import pytest
import torch
from test_retained_identity_noise import args
from test_transition1x_atom_identity import PAYLOAD, AXIS
from mora3.data.independent_current_noise import replay_independent_current
from mora3.model.current_frame_velocity import SplitContextualCoefficients
import mora3.model.identity_noise_vector as mod


def inputs(device='cpu', dtype=torch.float64):
    selections=[args(),args()]
    b=selections[1]['binding']
    record='Transition1x_original:v4:C2H2N2O2:rxn3534'
    numbers=(8,6,7,8,6,7,1,1)
    axis=tuple(replace(b.native_axis[0],ordinal=i,name=f'{record}:atom:{i}',element={1:'H',6:'C',7:'N',8:'O'}[z]) for i,z in enumerate(numbers))
    source=replace(b.current.source,source_id=record)
    binding=replace(b,current=replace(b.current,source=source),native_axis=axis,native_axis_sha256=axis_digest(axis),free_atoms=(True,)*8,xyz_available=(True,)*8)
    selections[1].update(seed=35101,binding=binding,ordinal=14,expected_record_id=record,expected_reaction_group='Transition1x:v4:C2H2N2O2:rxn3534',expected_native_split='train',expected_atomic_numbers=numbers)
    cases=[];targets=[]
    for i,s in enumerate(selections):
        prepared=mod.prepare_retained_identity_noise(**s)
        x=replay_independent_current(prepared['artifact'],device=device,dtype=dtype).requires_grad_()
        c=SplitContextualCoefficients(mod.IDENTITY_CONTEXT_AXIS,max_atoms=8).to(device=device,dtype=dtype)
        with torch.no_grad():
            for w in c.parameters():w.copy_(torch.arange(w.numel(),device=device,dtype=dtype).reshape(w.shape)/311+.01)
        cases.append(dict(current_xyz=x,coefficients=c,width_identity='diagnostic-identity-123'))
        targets.append((torch.arange(len(x)*3,device=device,dtype=dtype).reshape(len(x),3)/17+i).requires_grad_())
    return selections,cases,targets


def run(s,c,t,ready=(True,True)):
    assert hasattr(mod,'retained_identity_noise_vector_loss'),'missing retained ragged preparation'
    return mod.retained_identity_noise_vector_loss(s,c,t,ready)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_equal_case_independent_equations_and_gradients(device,dtype):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    s,c,t=inputs(device,dtype);rng=torch.random.get_rng_state().clone();gpu=torch.cuda.get_rng_state_all()
    result=run(s,c,t);terms=[];operands=[]
    for selection,case,target in zip(s,c,t):
        x=case['current_xyz'];p=case['coefficients'];z=x-x.mean(0);n=len(x)
        h=torch.tensor([[int(j==v) for j in range(1,119)]+[1]*5 for v in selection['expected_atomic_numbers']],device=device,dtype=dtype)
        v=torch.stack([sum((p.geometry.weight[0,0]*(x[j]-x[i]).norm()+p.geometry.weight[0,1]*(z[i]*z[j]).sum()+(h[i]*p.receiver.weight[0]).sum()+(h[j]*p.neighbor.weight[0]).sum())*(x[j]-x[i]) for j in range(n))/n for i in range(n)])
        mask=torch.tensor(selection['binding'].free_atoms,device=device)
        terms.append((v[mask]-target.detach()[mask]).square().mean());operands.extend((x,*p.parameters()))
    ref=sum(terms)/2
    torch.testing.assert_close(result.loss,ref)
    for a,b in zip(torch.autograd.grad(result.loss,operands,retain_graph=True),torch.autograd.grad(ref,operands)):
        torch.testing.assert_close(a,b,rtol=3e-4,atol=3e-4 if dtype==torch.float32 else 1e-8)
    assert torch.autograd.grad(result.loss,t,allow_unused=True)==(None,None)
    assert not torch.isclose(ref,(7*terms[0]+8*terms[1])/15)
    assert result.active_cases==(0,1)
    assert torch.equal(rng,torch.random.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(gpu,torch.cuda.get_rng_state_all()))
    assert all(p.grad is None for case in c for p in case['coefficients'].parameters())
    for name in ('require_source_use','require_model_use','require_training_use'):
        with pytest.raises(ValueError):getattr(result,name)()


@pytest.mark.parametrize('bad',['record','split','axis','context','payload','target','xyz','extra','selection_extra','owner','ready','length','container'])
def test_late_refusal_precedes_all_learned_calls(bad,tmp_path):
    s,c,t=inputs();ready=(True,True);calls=[]
    hooks=[m.register_forward_pre_hook(lambda *a:calls.append(1)) for case in c for m in case['coefficients'].modules()]
    if bad=='record':s[1]['expected_record_id']='other'
    if bad=='split':s[1]['expected_native_split']='val'
    if bad=='axis':s[1]['expected_atomic_numbers']=AXIS[::-1]
    if bad=='context':s[1]['expected_reaction_group']='other'
    if bad=='payload':
        path=tmp_path/'payload.parquet';shutil.copyfile(PAYLOAD,path)
        raw=bytearray(path.read_bytes());raw[len(raw)//2]^=1;path.write_bytes(raw);s[1]['payload_path']=path
    if bad=='target':t[1]=t[1]*float('nan')
    if bad=='xyz':c[1]['current_xyz']=c[1]['current_xyz']+1
    if bad=='extra':c[1]['atom_context']=object()
    if bad=='selection_extra':s[1]['target_xyz']=object()
    if bad=='owner':c[1]['coefficients']=object()
    if bad=='ready':ready=(True,1)
    if bad=='length':s.pop()
    if bad=='container':s[1]=object()
    with pytest.raises(ValueError):run(s,c,t,ready)
    for h in hooks:h.remove()
    assert calls==[]


def test_each_call_reauthenticates_and_inactive_values_opaque(tmp_path):
    s,c,t=inputs();path=tmp_path/'payload.parquet';shutil.copyfile(PAYLOAD,path)
    s[0]['payload_path']=path;s[1]=c[1]=t[1]=object()
    assert run(s,c,t,(True,False)).active_cases==(0,)
    raw=bytearray(path.read_bytes());raw[len(raw)//2]^=1;path.write_bytes(raw)
    with pytest.raises(ValueError,match='payload'):run(s,c,t,(True,False))
    with pytest.raises(ValueError):run(s,c,t,(False,False))
