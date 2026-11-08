"""Groups affect resume identity only; source gates and predictor isolation survive."""
import hashlib
import json
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from test_uniprot_native_reader import fixture, canonical
from test_uniprot_generation_reader import generation_fixture
from mora3.data import uniprot_annotations as u
from mora3.model.annotation_trainer import AnnotationBatch


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def setup_files(root, generation=False, guard='detected_protein_guard:one'):
    e,r = generation_fixture()[:2] if generation else fixture()
    raw=root/'raw.json';raw.write_bytes(canonical({'results':[e]}))
    t=json.loads(r['native_payload']);ref=t['audit']['source_ref'];ref['raw_file_sha256']=digest(raw)
    receipt=json.loads(r['admission_receipt_json']);receipt['source_ref']=ref
    r.update(task_id='task1', source_family='UniProt-full-record-annotations',native_payload=canonical(t),
             native_payload_sha256=hashlib.sha256(canonical(t)).hexdigest(), source_sha256=digest(raw),admission_receipt_json=json.dumps(receipt))
    pq.write_table(pa.Table.from_pylist([r]),root/'tasks.parquet')
    groups=json.loads(r['group_ids_json'])+[guard]
    pq.write_table(pa.Table.from_pylist([dict(task_id='task1',effective_groups_json=json.dumps(groups))]),root/'groups.parquet')
    def receipt_for(name,**kw):
        p=root/name;return dict(path=name,bytes=p.stat().st_size,sha256=digest(p),**kw)
    shard=receipt_for('tasks.parquet',rows=1,split='train',source_family=r['source_family'])
    m=dict(source_release='2026_03',license='CC-BY-4.0',source_only_train_split=True,
           homology_and_source_parent_heldout_unresolved=True,task_shards=[shard],
           source_page_receipts=[receipt_for('raw.json',source_release='2026_03',compression='none',entries=1)])
    snapshot=root/'snapshot.json';snapshot.write_bytes(canonical(m))
    master=root/'master.json';master.write_bytes(canonical(dict(shards=[shard],effective_group_lookup={'tasks.parquet':receipt_for('groups.parquet',rows=1)})))
    return snapshot,master,r,e


def read(root,snapshot,master,generation=False,**kw):
    return list((u.read_generation_views if generation else u.read_prediction_views)(snapshot,root=root,limit=1,
        authoritative_master=master,master_sha256=kw.get('master_sha256',digest(master))))[0]


@pytest.mark.parametrize('generation',[False,True])
def test_master_groups_preserve_native_semantics_and_bind_provenance(tmp_path,generation):
    s,m,r,e=setup_files(tmp_path,generation)
    old=(u.generation_view if generation else u.prediction_view)(r,e)
    v=read(tmp_path,s,m,generation)
    assert v.inputs()==old.inputs() and v.target_bytes==old.target_bytes
    assert v.provenance()['source_groups']==old.provenance()['groups']
    assert 'detected_protein_guard:one' in v.provenance()['groups']
    assert v.provenance()['authoritative_groups']['master_sha256']==digest(m)
    assert v.family_group is None and v.heldout_eligible is False
    if not generation:
        assert AnnotationBatch((v,),(True,),(True,)).binding()!=AnnotationBatch((old,),(True,),(True,)).binding()


def test_guard_change_changes_batch_binding_without_changing_inputs(tmp_path):
    s,m,_,_=setup_files(tmp_path);a=read(tmp_path,s,m)
    setup_files(tmp_path,guard='detected_protein_guard:two');b=read(tmp_path,s,m)
    assert a.inputs()==b.inputs() and a.target_bytes==b.target_bytes
    assert AnnotationBatch((a,),(True,),(True,)).binding()!=AnnotationBatch((b,),(True,),(True,)).binding()


@pytest.mark.parametrize('bad',['master_hash','missing','shard_hash','group_hash','duplicate','wrong_task','drop_identity','not_list','escape'])
def test_master_group_integrity_fails_closed(tmp_path,bad):
    s,m,r,e=setup_files(tmp_path);d=json.loads(m.read_bytes());g=d['effective_group_lookup']['tasks.parquet'];expected=digest(m)
    if bad=='master_hash':expected='0'*64
    elif bad=='missing':d['effective_group_lookup'].clear()
    elif bad=='shard_hash':d['shards'][0]['sha256']='0'*64
    elif bad=='group_hash':g['sha256']='0'*64
    elif bad=='escape':g['path']='../groups.parquet'
    else:
        row=dict(task_id='wrong' if bad=='wrong_task' else 'task1',effective_groups_json='[]' if bad=='drop_identity' else '{}' if bad=='not_list' else r['group_ids_json'])
        pq.write_table(pa.Table.from_pylist([row,row] if bad=='duplicate' else [row]),tmp_path/'groups.parquet')
        g.update(bytes=(tmp_path/'groups.parquet').stat().st_size,sha256=digest(tmp_path/'groups.parquet'),rows=2 if bad=='duplicate' else 1)
    if bad!='master_hash':m.write_bytes(canonical(d));expected=digest(m)
    with pytest.raises(ValueError):read(tmp_path,s,m,master_sha256=expected)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_group_and_hidden_target_isolation_and_checkpoint_rejection(tmp_path,device):
    import torch
    from dataclasses import replace
    from test_annotation_multihead import model
    from mora3.model.annotation_trainer import AnnotationTrainer
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    s,m,_,_=setup_files(tmp_path);a=read(tmp_path,s,m)
    setup_files(tmp_path,guard='detected_protein_guard:two');b=read(tmp_path,s,m)
    torch.manual_seed(9601);net=model().to(device)
    hidden=replace(b,target_bytes=b'{"annotation_packet":{"hidden":"changed"}}')
    for v in [b,hidden]:
        for key,value in net([a]).items():assert torch.equal(value,net([v])[key])
    first=AnnotationTrainer(net,AnnotationBatch((a,),(True,),(True,)))
    second=AnnotationTrainer(model().to(device),AnnotationBatch((b,),(True,),(True,)))
    with pytest.raises(ValueError,match='binding'):second.load_state_dict(first.state_dict())
