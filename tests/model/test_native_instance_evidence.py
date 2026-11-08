"""Synthetic corpus evidence is not actual source admission."""
import hashlib
import json
import pytest
from test_color_supervised_task import source
from test_native_instance_admission import payload, make


def fixture():
    s=source('ordered_binary_instances',split='train')
    common=dict(issuer='dataset_construction',native_binding_sha256=s.binding_sha256,
                source_id=s.image.source_id,parent_id=s.image.parent_id)
    rights=dict(common,schema='mora3.native_instance_rights.v1',training_permitted=True,
                source_terms='Synthetic test terms only',source_revision='fixture-v1')
    lineage=dict(common,schema='mora3.native_instance_lineage.v1',
        image_sha256=s.image_sha256,ordered_mask_sha256=list(s.mask_sha256),
        effective_group='synthetic-parent-group',effective_split='train',
        cross_split_conflict=False,parent_audit_complete=True)
    raw=[json.dumps(x).encode() for x in (rights,lineage)]
    p=payload(s)
    for name,b in zip(('rights','lineage'),raw):p[name+'_evidence_sha256']=hashlib.sha256(b).hexdigest()
    return make(s,p),raw


def resolve(r,raw):
    from mora3.data.native_instance_admission import resolve_native_instance_evidence
    return resolve_native_instance_evidence(r,rights_bytes=raw[0],lineage_bytes=raw[1])


def test_exact_evidence_is_detached_and_never_grants_execution():
    r,raw=fixture();resolved=resolve(r,raw)
    assert resolved.receipt_sha256==r.receipt_sha256
    assert resolved.rights_sha256==hashlib.sha256(raw[0]).hexdigest()
    assert resolved.lineage_sha256==hashlib.sha256(raw[1]).hexdigest()
    with pytest.raises(ValueError,match='admission'):resolved.require_training()


@pytest.mark.parametrize('which,field,value',[
 (0,'training_permitted',1),(0,'training_permitted',False),(0,'source_terms',''),
 (0,'source_revision',''),(0,'issuer','model_worker'),(0,'native_binding_sha256','e'*64),
 (1,'parent_audit_complete',1),(1,'parent_audit_complete',False),
 (1,'cross_split_conflict',0),(1,'cross_split_conflict',True),
 (1,'effective_split','test'),(1,'effective_group','other'),(1,'ordered_mask_sha256',[]),
 (1,'image_sha256','f'*64),(1,'parent_id','other')])
def test_rehashed_but_incompatible_evidence_refuses(which,field,value):
    r,raw=fixture();d=json.loads(raw[which]);d[field]=value;raw[which]=json.dumps(d).encode()
    p=json.loads(r.receipt_bytes);name=('rights','lineage')[which]
    p[name+'_evidence_sha256']=hashlib.sha256(raw[which]).hexdigest();r=make(r.source,p)
    with pytest.raises(ValueError):resolve(r,raw)


@pytest.mark.parametrize('bad',[b'{}',b'null',b'\xff',b'{"a":1,"a":2}',b' '* (1024*1024+1)])
def test_bad_or_stale_evidence_refuses(bad):
    r,raw=fixture();raw[0]=bad
    with pytest.raises(ValueError):resolve(r,raw)
