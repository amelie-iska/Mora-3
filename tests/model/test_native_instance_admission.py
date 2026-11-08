"""Synthetic receipts exercise structure, never grant actual source admission."""
import hashlib
import json
import pytest
import torch
from test_color_supervised_task import source


def payload(s):
    return dict(schema='mora3.native_instance_receipt.v1', issuer='dataset_construction',
        native_binding_sha256=s.binding_sha256, image_sha256=s.image_sha256,
        ordered_mask_sha256=list(s.mask_sha256), source_id=s.image.source_id,
        parent_id=s.image.parent_id, source_split=s.split,
        master_sha256='a'*64, task_sha256='b'*64, rights_evidence_sha256='c'*64,
        lineage_evidence_sha256='d'*64, effective_group='synthetic-parent-group',
        effective_split='train', objective='ordered-nuclear-instance-set-BCE-v1',
        roles={'image':'condition','ordered_masks':'target'}, training_eligible=True)


def make(s, p=None, **kw):
    from mora3.data.native_instance_admission import NativeInstanceReceipt
    raw=json.dumps(payload(s) if p is None else p).encode()
    return NativeInstanceReceipt(s,raw,hashlib.sha256(raw).hexdigest(),
        expected_master_sha256=kw.get('master','a'*64),expected_task_sha256='b'*64)


def test_detached_immutable_contract_and_no_execution_grant():
    s=source('ordered_binary_instances',split='train')
    p=payload(s); r=make(s,p)
    p['roles']['ordered_masks']='condition'
    assert r.source_binding_sha256==s.binding_sha256
    assert r.receipt_sha256==hashlib.sha256(r.receipt_bytes).hexdigest()
    with pytest.raises((AttributeError,TypeError)):r.receipt_bytes=b'bad'
    with pytest.raises(ValueError,match='execution|admission'):r.require_training()
    with pytest.raises(ValueError):s.require_training()


@pytest.mark.parametrize('field,value',[
 ('schema','old'),('issuer','model_worker'),('native_binding_sha256','f'*64),
 ('image_sha256','f'*64),('ordered_mask_sha256',[]),('source_id','other'),
 ('parent_id','other'),('source_split','test'),('master_sha256','e'*64),
 ('task_sha256','e'*64),('rights_evidence_sha256',''),('lineage_evidence_sha256',None),
 ('effective_group',''),('effective_split','test'),('training_eligible',1),
 ('roles',{'image':'condition','ordered_masks':'condition'}),('objective','semantic'),
 ('unreviewed_extra',True)])
def test_mismatch_refuses_without_rng_or_source_mutation(field,value):
    s=source('ordered_binary_instances',split='train'); p=payload(s);p[field]=value
    before=torch.random.get_rng_state().clone(); binding=s.binding_sha256
    with pytest.raises(ValueError):make(s,p)
    assert torch.equal(before,torch.random.get_rng_state())
    assert binding==s.binding_sha256


@pytest.mark.parametrize('split',['test','validation','evaluation'])
def test_source_holdout_cannot_be_promoted(split):
    s=source('ordered_binary_instances',split=split)
    with pytest.raises(ValueError):make(s)


def test_missing_stale_and_reordered_receipts_refuse():
    s=source('ordered_binary_instances',split='train')
    for key in payload(s):
        p=payload(s);del p[key]
        with pytest.raises(ValueError):make(s,p)
    with pytest.raises(ValueError):make(s,master='e'*64)
    p=payload(s);p['ordered_mask_sha256']=p['ordered_mask_sha256'][::-1]
    if len(set(s.mask_sha256))>1:
        with pytest.raises(ValueError):make(s,p)


@pytest.mark.parametrize('raw',[b'{}', b'{"schema":1,"schema":2}',b'null',b'[]',b'\xff',b' '* (1024*1024+1)])
def test_malformed_bytes_refuse(raw):
    from mora3.data.native_instance_admission import NativeInstanceReceipt
    s=source('ordered_binary_instances',split='train')
    with pytest.raises(ValueError):NativeInstanceReceipt(s,raw,hashlib.sha256(raw).hexdigest(),expected_master_sha256='a'*64,expected_task_sha256='b'*64)


def test_mask_order_and_receipt_hash_are_authenticated():
    from dataclasses import replace
    import io
    import numpy as np
    from PIL import Image
    from mora3.data.native_instance_admission import NativeInstanceReceipt
    s=source('ordered_binary_instances',split='train')
    stream=io.BytesIO();Image.fromarray((~s.target_arrays()[0]).astype(np.uint8)*255).save(stream,format='PNG')
    raw=stream.getvalue()
    s=replace(s,masks_encoded=s.masks_encoded+(raw,),mask_sha256=s.mask_sha256+(hashlib.sha256(raw).hexdigest(),))
    p=payload(s);p['ordered_mask_sha256'].reverse()
    with pytest.raises(ValueError):make(s,p)
    raw=json.dumps(payload(s)).encode()
    with pytest.raises(ValueError,match='checksum'):NativeInstanceReceipt(s,raw,'f'*64,'a'*64,'b'*64)
