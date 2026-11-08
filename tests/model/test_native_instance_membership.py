"""Synthetic prospective corpus membership; never actual training admission."""
import hashlib
import io
import json
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from test_native_instance_evidence import fixture
from test_native_instance_admission import make


def digest(b):return hashlib.sha256(b).hexdigest()
def packed(rows):
    b=io.BytesIO();pq.write_table(pa.Table.from_pylist(rows),b);return b.getvalue()


def case(change=None):
    r,raw=fixture();p=json.loads(r.receipt_bytes)
    row=dict(task_id='synthetic-native-task',source_family='MoNuSeg',
        native_schema='mora3.native_ordered_instance_task.v1',
        native_binding_sha256=r.source.binding_sha256,source_id=r.source.image.source_id,
        parent_id=r.source.image.parent_id,image_sha256=r.source.image_sha256,
        ordered_mask_sha256_json=json.dumps(list(r.source.mask_sha256)),
        task_name=p['objective'],split='train',admitted_for_task=True,
        roles_json=json.dumps(p['roles']))
    group=dict(task_id=row['task_id'],effective_split='train',
        effective_groups_json=json.dumps([p['effective_group']]))
    rows=[row];groups=[group]
    if change:change(rows,groups)
    task=packed(rows);side=packed(groups)
    master=dict(schema='mora3.master_corpus_manifest.v6',shards=[dict(
        path='synthetic/tasks.parquet',rows=len(rows),bytes=len(task),sha256=digest(task),
        source_family='MoNuSeg',split='train')],effective_group_lookup={
        'synthetic/tasks.parquet':dict(path='synthetic/groups.parquet',rows=len(groups),
                                      bytes=len(side),sha256=digest(side))})
    m=json.dumps(master).encode();p.update(master_sha256=digest(m),task_sha256=digest(task))
    from mora3.data.native_instance_admission import NativeInstanceReceipt
    rb=json.dumps(p).encode();r=NativeInstanceReceipt(r.source,rb,digest(rb),digest(m),digest(task))
    return r,raw,m,task,side


def run(c,**kw):
    from mora3.data.native_instance_admission import resolve_native_instance_membership
    r,raw,m,task,side=c
    return resolve_native_instance_membership(r,rights_bytes=raw[0],lineage_bytes=raw[1],
        master_bytes=kw.get('master',m),task_bytes=kw.get('task',task),
        group_bytes=kw.get('group',side),current_master_sha256=kw.get('current',digest(m)))


def test_exact_membership_is_immutable_and_never_grants_execution():
    c=case();out=run(c)
    assert out.task_id=='synthetic-native-task'
    assert out.master_sha256==digest(c[2]) and out.task_sha256==digest(c[3])
    assert out.group_sha256==digest(c[4])
    assert out.effective_groups==('synthetic-parent-group',)
    with pytest.raises((AttributeError,TypeError)):out.task_id='changed'
    with pytest.raises(ValueError,match='admission'):out.require_training()


@pytest.mark.parametrize('field,value',[
 ('native_binding_sha256','f'*64),('source_id','other'),('parent_id','other'),
 ('image_sha256','f'*64),('ordered_mask_sha256_json','[]'),
 ('task_name','image_to_semantic'),('split','test'),('admitted_for_task',False),
 ('admitted_for_task',1),('native_schema','old'),('source_family','DeepBacs'),
 ('roles_json','{"image":"condition","ordered_masks":"condition"}')])
def test_rehashed_but_wrong_task_refuses(field,value):
    c=case(lambda rows,groups:rows[0].update({field:value}))
    with pytest.raises(ValueError):run(c)


@pytest.mark.parametrize('change',[
 lambda rows,g:g[0].update(task_id='other'),
 lambda rows,g:g[0].update(effective_split='test'),
 lambda rows,g:g[0].update(effective_groups_json='[]'),
 lambda rows,g:g[0].update(effective_groups_json='["other"]'),
 lambda rows,g:g[0].update(effective_groups_json='["synthetic-parent-group","other"]'),
 lambda rows,g:(rows.append(dict(rows[0])),g.append(dict(g[0]))),
 lambda rows,g:g.append(dict(g[0])),
])
def test_missing_duplicate_misaligned_or_unresolved_groups_refuse(change):
    with pytest.raises(ValueError):run(case(change))


@pytest.mark.parametrize('field,value',[
 ('master',b'{}'),('task',b'bad'),('group',b'bad'),('current','f'*64),
 ('master',bytearray(b'{}')),('current',None)])
def test_stale_current_pin_or_transport_refuses(field,value):
    with pytest.raises(ValueError):run(case(),**{field:value})


def test_duplicate_master_shard_refuses_even_with_recomputed_pins():
    from mora3.data.native_instance_admission import NativeInstanceReceipt
    c=list(case());m=json.loads(c[2]);m['shards']*=2;c[2]=json.dumps(m).encode()
    p=json.loads(c[0].receipt_bytes);p['master_sha256']=digest(c[2]);b=json.dumps(p).encode()
    c[0]=NativeInstanceReceipt(c[0].source,b,digest(b),digest(c[2]),digest(c[3]))
    with pytest.raises(ValueError):run(c)
