"""Admission fixtures exercise gates, never grant rights to real sources."""
import hashlib
import json
from dataclasses import replace
import pytest
from test_public_image_masks import build, fixture
from mora3.data.public_image_admission import PublicImageAdmission


def receipt(tmp_path, x, **changes):
    row=dict(binding_sha256=x.binding_sha256, source_id=x.image.source_id,
             parent_id=x.image.parent_id, split=x.split, encoding=x.encoding,
             training_eligible=True, rights_review='fixture-only',
             lineage_review='fixture-only', leakage_group='fixture-parent',
             allowed_objectives=['whole-cell-instance-v1'])
    row.update(changes)
    payload=dict(schema='mora3.public_image_admission.v1', records=[row])
    raw=json.dumps(payload).encode(); path=tmp_path/'admission.json'; path.write_bytes(raw)
    return path,hashlib.sha256(raw).hexdigest()


def train_image():
    return replace(build(fixture('instance_id_map')),split='train')


def test_source_and_admission_bound_snapshot_excludes_targets_from_input(tmp_path):
    x=train_image(); path,digest=receipt(tmp_path,x)
    a=PublicImageAdmission.from_file(x,path,sha256=digest)
    a.require_training()
    assert a.native_input()==x.native_input()
    assert a.instances().shape==(1,3,4)
    assert a.instances().sum()==2
    assert a.binding_sha256!=x.binding_sha256
    assert a.admission_sha256==digest
    other=PublicImageAdmission.from_file(x,path,sha256=digest)
    assert a.canonical_bytes()==other.canonical_bytes()
    a.assert_resume_matches(other)
    # Returned target storage cannot mutate the source or checkpoint binding.
    a.instances().zero_()
    assert a.instances().sum()==2


@pytest.mark.parametrize('changes',[
    {'training_eligible':False},{'training_eligible':1}, {'split':'test'},
    {'parent_id':'another-parent'},{'source_id':'another-source'},
    {'encoding':'semantic_binary'},{'binding_sha256':'0'*64},
    {'rights_review':''},{'lineage_review':''},{'leakage_group':''},
    {'allowed_objectives':['semantic-only']},
])
def test_unadmitted_or_mismatched_source_rejected(tmp_path,changes):
    x=train_image(); path,digest=receipt(tmp_path,x,**changes)
    with pytest.raises(ValueError): PublicImageAdmission.from_file(x,path,sha256=digest)


@pytest.mark.parametrize('kind',['semantic_binary','ordered_binary_instances'])
def test_unsupported_objective_representation_rejected(tmp_path,kind):
    x=replace(build(fixture(kind)),split='train'); path,digest=receipt(tmp_path,x)
    with pytest.raises(ValueError,match='instance'): PublicImageAdmission.from_file(x,path,sha256=digest)


def test_heldout_cannot_be_promoted_by_receipt(tmp_path):
    x=build(fixture('instance_id_map')); path,digest=receipt(tmp_path,x)
    with pytest.raises(ValueError,match='split'): PublicImageAdmission.from_file(x,path,sha256=digest)


def test_external_file_changes_hash_and_resume_identity(tmp_path):
    x=train_image(); path,digest=receipt(tmp_path,x)
    a=PublicImageAdmission.from_file(x,path,sha256=digest)
    path,digest2=receipt(tmp_path,x,rights_review='revised-fixture-only')
    with pytest.raises(ValueError,match='checksum'): PublicImageAdmission.from_file(x,path,sha256=digest)
    b=PublicImageAdmission.from_file(x,path,sha256=digest2)
    with pytest.raises(ValueError,match='resume'): a.assert_resume_matches(b)
    assert a.admission_sha256==digest


def test_ambiguous_duplicate_or_unversioned_receipt_rejected(tmp_path):
    x=train_image(); path,digest=receipt(tmp_path,x)
    d=json.loads(path.read_bytes()); d['records']*=2
    for payload in [d,dict(d,schema='unknown')]:
        raw=json.dumps(payload).encode();path.write_bytes(raw)
        with pytest.raises(ValueError): PublicImageAdmission.from_file(x,path,sha256=hashlib.sha256(raw).hexdigest())


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_admitted_instance_planes_match_reference_union_loss_and_gradients(tmp_path,device):
    import torch
    from mora3.model.morphology import foreground_objective
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    x=train_image();path,digest=receipt(tmp_path,x)
    admitted=PublicImageAdmission.from_file(x,path,sha256=digest)
    z=torch.linspace(-1,1,12,device=device,dtype=torch.float64).reshape(3,4).requires_grad_()
    target=admitted.instances(device=device)
    loss=foreground_objective(z,target)
    y=torch.tensor([[0,1,1,0],[0,0,0,0],[0,0,0,0]],device=device,dtype=z.dtype)
    reference=(torch.nn.functional.softplus(z)-y*z).mean()
    torch.testing.assert_close(loss,reference)
    loss.backward()
    torch.testing.assert_close(z.grad,(z.detach().sigmoid()-y)/12)
