"""Independent current generation must never consume endpoint/TS coordinates."""
from dataclasses import replace
import hashlib
import importlib
import pytest
import torch
from test_native_current_preflight import candidate


def module():
    return importlib.import_module('mora3.data.independent_current_noise')


def args():
    c=candidate()
    # No gold coordinates are needed even for atoms formerly marked fixed.
    b=replace(c['binding'],free_atoms=(True,)*len(c['binding'].native_axis))
    return dict(binding=b,context_sha256=c['context_sha256'],
        expected_context_sha256=c['expected_context_sha256'],seed=33901,
        native_split='val',expected_native_split='val')


def test_recipe_bytes_match_independent_normal_reference_and_preserve_rng():
    m=module();a=args();before=torch.random.get_rng_state().clone()
    r=m.construct_independent_current(**a)
    gen=torch.Generator(device='cpu').manual_seed(a['seed'])
    ref=torch.randn((3,3),dtype=torch.float64,generator=gen)
    assert r.xyz_bytes==ref.numpy().astype('<f8').tobytes()
    assert r.metadata.current_xyz_sha256==hashlib.sha256(r.xyz_bytes).hexdigest()
    assert torch.equal(before,torch.random.get_rng_state())
    assert r.metadata.seed==33901 and r.metadata.binding is a['binding']
    assert r.metadata.status=='unqualified_metadata_candidate'
    assert b'normal' in m.RECIPE_BYTES and b'float64' in m.RECIPE_BYTES
    assert r.metadata.recipe_sha256==hashlib.sha256(m.RECIPE_BYTES).hexdigest()
    for name in ('require_source_use','require_model_use','require_training_use'):
        with pytest.raises(ValueError):getattr(r.metadata,name)()


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_replay_copy_matches_reference_without_mutating_global_rng(device,dtype):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    m=module();r=m.construct_independent_current(**args())
    cpu=torch.random.get_rng_state().clone()
    gpu=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    x=m.replay_independent_current(r,device=device,dtype=dtype)
    g=torch.Generator().manual_seed(33901)
    ref=torch.randn((3,3),dtype=torch.float64,generator=g).to(device=device,dtype=dtype)
    assert torch.equal(x,ref) and not x.requires_grad
    x.zero_()
    assert torch.equal(m.replay_independent_current(r,device=device,dtype=dtype),ref)
    assert torch.equal(cpu,torch.random.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(gpu,torch.cuda.get_rng_state_all()))


def test_context_changes_binding_without_changing_noise_and_seed_changes_noise():
    m=module();a=args();r=m.construct_independent_current(**a)
    changed=dict(a,context_sha256='a'*64,expected_context_sha256='a'*64)
    s=m.construct_independent_current(**changed)
    assert s.xyz_bytes==r.xyz_bytes and s.metadata.context_sha256!=r.metadata.context_sha256
    assert m.construct_independent_current(**dict(a,seed=33902)).xyz_bytes!=r.xyz_bytes


@pytest.mark.parametrize('bad',['seed_bool','seed_negative','seed_large','context','split','fixed','unit','stale_axis','synthetic'])
def test_invalid_or_unsupported_inputs_refused(bad):
    m=module();a=args();b=a['binding']
    if bad=='seed_bool':a['seed']=True
    if bad=='seed_negative':a['seed']=-1
    if bad=='seed_large':a['seed']=2**63
    if bad=='context':a['context_sha256']='0'*64
    if bad=='split':a['expected_native_split']='test'
    if bad=='fixed':a['binding']=replace(b,free_atoms=(True,False,True))
    if bad=='unit':object.__setattr__(b.current,'units','nm')
    if bad=='stale_axis':a['binding']=replace(b,native_axis=b.native_axis[::-1])
    if bad=='synthetic':
        from test_current_frame_request_binding import fixture,bind
        a['binding']=bind(fixture())
    with pytest.raises(ValueError):m.construct_independent_current(**a)


@pytest.mark.parametrize('field',['xyz','seed','recipe','hash','axis','context'])
def test_replay_rejects_tampering(field):
    m=module();r=m.construct_independent_current(**args());c=r.metadata
    if field=='xyz':r=replace(r,xyz_bytes=bytes(len(r.xyz_bytes)))
    if field=='seed':r=replace(r,metadata=replace(c,seed=c.seed+1))
    if field=='recipe':r=replace(r,metadata=replace(c,recipe_sha256='0'*64))
    if field=='hash':r=replace(r,metadata=replace(c,current_xyz_sha256='0'*64))
    if field=='axis':object.__setattr__(c.binding,'native_axis_sha256','0'*64)
    if field=='context':r=replace(r,expected_context_sha256='0'*64)
    with pytest.raises(ValueError):m.replay_independent_current(r)


def test_target_coordinate_keyword_and_unsupported_dtype_refused():
    m=module()
    with pytest.raises(TypeError):m.construct_independent_current(**args(),target_xyz=torch.zeros(3,3))
    with pytest.raises(ValueError):m.replay_independent_current(m.construct_independent_current(**args()),dtype=torch.float16)


def test_valid_rebinding_cannot_reuse_old_noise_receipt():
    from mora3.data.geometry_frame_policy import axis_digest
    m=module();r=m.construct_independent_current(**args());b=r.metadata.binding
    axis=tuple(replace(a,element='N') for a in b.native_axis)
    changed=replace(b,native_axis=axis,native_axis_sha256=axis_digest(axis))
    changed.validate()
    forged=replace(r,metadata=replace(r.metadata,binding=changed))
    with pytest.raises(ValueError):m.replay_independent_current(forged)
