"""Fresh retained source binding precedes any generated-noise diagnostic."""
from dataclasses import replace
import importlib
import shutil
import pytest
import torch
from test_transition1x_atom_identity import MANIFEST, PAYLOAD, RECORD, GROUP, AXIS
from mora3.data import spatial_design_constraints as q, geometry_frame_policy as p
from mora3.data.current_frame_request_binding import bind_current_frame_request
from mora3.model.identity_noise_vector import IDENTITY_CONTEXT_AXIS, identity_noise_vector_diagnostic
from mora3.model.current_frame_velocity import SplitContextualCoefficients


def args():
    ids=tuple(f'{RECORD}:atom:{i}' for i in range(7))
    axis=tuple(q.AtomKey('current',i,{1:'H',6:'C',7:'N',8:'O'}[z],name) for i,(z,name) in enumerate(zip(AXIS,ids)))
    req=q.DesignRequest(q.Direction.DE_NOVO,(q.InputSpec('current',q.InputKind.TS_GEOMETRY,q.Role.USER_SUPPLIED),),q.Budget(1,8,8,32,1,1048576,1))
    sha='2da99f8b29473c1e3676e8fd902e26eab606cb0f548fd2fbad199f0d0f53b3cf'
    src=p.SourceDeclaration(RECORD,'retained-native-v1','diagnostic-identity-only',sha,sha,'c'*64,'d'*64,p.Origin.UNQUALIFIED_SOURCE,p.Purpose.CONDITION_INPUT)
    b=bind_current_frame_request(request=req,compiled=q.compile_design_request(req),current=p.CurrentFrame('current','current','angstrom',src),role=q.Role.USER_SUPPLIED,native_axis=axis,native_axis_sha256=p.axis_digest(axis),xyz_available=(True,)*7,free_atoms=(True,)*7)
    return dict(manifest_path=MANIFEST,payload_index=0,ordinal=13,expected_record_id=RECORD,expected_reaction_group=GROUP,expected_native_split='val',expected_atomic_numbers=AXIS,binding=b,seed=35001)


def prepare(**a):
    m=importlib.import_module('mora3.model.identity_noise_vector')
    assert hasattr(m,'prepare_retained_identity_noise'), 'retained identity noise adapter missing'
    return m.prepare_retained_identity_noise(**a)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_real_identity_generated_noise_equation_and_gradients(device,dtype):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    a=args();prepared=prepare(**a)
    from mora3.data.independent_current_noise import replay_independent_current
    x=replay_independent_current(prepared['artifact'],device=device,dtype=dtype).requires_grad_()
    c=SplitContextualCoefficients(IDENTITY_CONTEXT_AXIS,max_atoms=8).to(device=device,dtype=dtype)
    with torch.no_grad():
        for weight in c.parameters():weight.copy_(torch.arange(weight.numel(),device=device,dtype=dtype).reshape(weight.shape)/311+.01)
    rng=torch.random.get_rng_state().clone()
    v=identity_noise_vector_diagnostic(**prepared,current_xyz=x,coefficients=c,width_identity='diagnostic-identity-123')
    h=torch.tensor([[int(j==z) for j in range(1,119)]+[1]*5 for z in AXIS],device=device,dtype=dtype)
    z=x-x.mean(0)
    ref=torch.stack([sum((c.geometry.weight[0,0]*(x[j]-x[i]).norm()+c.geometry.weight[0,1]*(z[i]*z[j]).sum()+(h[i]*c.receiver.weight[0]).sum()+(h[j]*c.neighbor.weight[0]).sum())*(x[j]-x[i]) for j in range(7))/7 for i in range(7)])
    torch.testing.assert_close(v,ref)
    for l,r in zip(torch.autograd.grad(v.square().sum(),(x,*c.parameters()),retain_graph=True),torch.autograd.grad(ref.square().sum(),(x,*c.parameters()))):
        torch.testing.assert_close(l,r,rtol=3e-4,atol=3e-4 if dtype==torch.float32 else 1e-8)
    assert torch.equal(rng,torch.random.get_rng_state())
    assert prepared['binding'] is a['binding']
    for obj in (prepared['atom_context'],prepared['artifact'].metadata):
        with pytest.raises(ValueError):obj.require_model_use()


@pytest.mark.parametrize('bad',['record','source','axis','element','ordinal','split','group','seed'])
def test_rebinding_refused(bad):
    a=args();b=a['binding'];s=b.current.source
    if bad in ('record','source'):
        s=replace(s,**({'source_id':'other'} if bad=='record' else {'original_identity_sha256':'a'*64}))
        a['binding']=replace(b,current=replace(b.current,source=s))
    if bad in ('axis','element'):
        axis=tuple(replace(k,**({'name':'other'} if bad=='axis' else {'element':'He'})) for k in b.native_axis)
        a['binding']=replace(b,native_axis=axis,native_axis_sha256=p.axis_digest(axis))
    if bad=='ordinal':a['ordinal']=12
    if bad=='split':a['expected_native_split']='train'
    if bad=='group':a['expected_reaction_group']='other'
    if bad=='seed':a['seed']=True
    with pytest.raises(ValueError):prepare(**a)


def test_every_preparation_rehashes_payload(tmp_path):
    path=tmp_path/'payload.parquet';shutil.copyfile(PAYLOAD,path)
    a=dict(args(),payload_path=path);prepare(**a)
    raw=bytearray(path.read_bytes());raw[len(raw)//2]^=1;path.write_bytes(raw)
    with pytest.raises(ValueError,match='payload'):prepare(**a)
