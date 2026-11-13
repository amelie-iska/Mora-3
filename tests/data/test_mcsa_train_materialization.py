"""Synthetic fixtures remain test-only; actual source is verified separately."""
from pathlib import Path
import copy
import json
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
import test_mcsa_role_list_admission as fixture
from mora3.data.mcsa_train_materialization import read_mcsa_train_materialization
from mora3.data.mcsa_role_list_admission import MCSARoleListAdmission
from mora3.data.mcsa_weak_sites import collate_weak_site_candidates
from mora3.research.mcsa_campaign_route import MCSACampaignTrainRoute, MCSAFinitePassPolicy

@pytest.fixture
def source():
    f=fixture.AdmissionTests();f.setUp()
    try:
        component=json.loads(Path(f.component_ref['path']).read_text())
        contexts=[]
        for ordinal,record in enumerate(f.records):
            if record.audit['proposed_split']!='train':continue
            raw=fixture.wire(f.natives[ordinal])
            contexts.append(dict(task_id=fixture.hashlib.sha256(b'Mora3.MCSA.source-role-list/v1\0'+raw).hexdigest(),source_record_id=record.audit['record_id'],source_candidate_record_ordinal=ordinal,current_sequence_provenance=record.audit['current_uniprot_provenance'],observed_source_mapping_views=len(record.source_lists),observed_source_role_EOS_symbols=sum(len(x.role_ids)+1 for x in record.source_lists),vocabulary_axis_sha256=f.vocab.axis_sha256,grouping_keys_are_audit_only=True))
        p=f.root/'train-contexts.jsonl';p.write_bytes(b''.join(fixture.wire(x)+b'\n' for x in contexts));f.context_ref=fixture.reference(p)
        f.review_path=f.root/'review.json';f.review_ref=fixture.write(f.review_path,f.packet)
        f.component=component
        yield f
    finally:f.doCleanups()

def load(f,**kwargs):
    return read_mcsa_train_materialization(f.root,review_path=f.review_path,expected_review_sha256=f.review_ref['sha256'],contexts_reference=f.context_ref,max_input_bytes=kwargs.pop('max_input_bytes',4*2**20),max_records=kwargs.pop('max_records',8),**kwargs)

def deny_heldout(monkeypatch,f):
    forbidden={Path(f.component['source_candidate_file']['path']).resolve()}
    for shard in f.component['shards']:
        if shard['split']!='train':forbidden.add(Path(shard['path']).resolve())
    for item in f.component['group_sidecars']:
        if item['source_shard']['split']!='train':forbidden.add(Path(item['authoritative_sidecar']['path']).resolve())
    original=Path.open;seen=[]
    def safe(path,*args,**kwargs):
        q=path.resolve();assert q not in forbidden,'Protected source file was opened'
        seen.append(q);return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',safe)
    return seen,forbidden

def test_materializer_reads_only_train_and_preserves_all_native_targets(source,monkeypatch):
    seen,forbidden=deny_heldout(monkeypatch,source)
    material=load(source)
    assert material.original_ordinals==(0,)
    assert material.records()==(source.records[0],)
    assert material.summary()['train_records']==1
    assert material.summary()['observed_role_eos_symbols']==3
    assert material.summary()['training_authorized'] is False
    assert not set(seen)&forbidden
    mutated=material.records()[0];mutated.audit['record_id']='changed'
    assert material.records()[0].audit['record_id']!='changed'

def test_native_constructor_uses_genuine_admission_without_opening_heldouts(source,monkeypatch):
    legacy=MCSACampaignTrainRoute(source.review_path,source.review_ref['sha256'],vocabulary=source.vocab,source_snapshot_sha256='a'*64,pass_policy=MCSAFinitePassPolicy(3,2,29),reviewed_payload_max_tokens=8192)
    expected=tuple(legacy.iter_from(legacy.initial_cursor))
    token=MCSARoleListAdmission.from_review_packet(source.review_path,source.review_ref['sha256'],vocabulary=source.vocab)
    seen,forbidden=deny_heldout(monkeypatch,source)
    material=load(source)
    isolated=MCSACampaignTrainRoute.from_train_materialization(material,admission=token,source_snapshot_sha256='a'*64,pass_policy=MCSAFinitePassPolicy(3,2,29),reviewed_payload_max_tokens=8192)
    assert isolated.binding()==legacy.binding()
    assert isolated.initial_cursor==legacy.initial_cursor
    assert isolated.validation_policy_sha256==legacy.validation_policy_sha256
    windows=tuple(isolated.iter_from(isolated.initial_cursor))
    assert windows==expected
    assert tuple(isolated.iter_from(windows[0].cursor_after))==windows[1:]
    assert not set(seen)&forbidden

@pytest.mark.parametrize('bad',('none','object','forged'))
def test_source_object_is_not_an_admission_token(source,bad):
    material=load(source)
    token=None if bad=='none' else object() if bad=='object' else object.__new__(MCSARoleListAdmission)
    with pytest.raises(ValueError):
        MCSACampaignTrainRoute.from_train_materialization(material,admission=token,source_snapshot_sha256='a'*64,pass_policy=MCSAFinitePassPolicy(1,2,29),reviewed_payload_max_tokens=8192)

@pytest.mark.parametrize('field,value', [('max_input_bytes',True),('max_records',True),('max_records',0),('max_input_bytes',0),('max_input_bytes',100)])
def test_explicit_materialization_budgets(source,field,value):
    with pytest.raises(ValueError):load(source,**{field:value})

def test_source_drift_detected_before_native_use(source):
    material=load(source)
    p=Path(source.component['shards'][0]['path']);p.write_bytes(p.read_bytes()+b'X')
    with pytest.raises(ValueError,match='changed'):material.records()

def test_context_hash_change_refused(source):
    p=Path(source.context_ref['path']);p.write_bytes(p.read_bytes()+b'\n')
    with pytest.raises(ValueError):load(source)

@pytest.mark.parametrize('change', ('ordinal','record','count','axis','evidence','provenance','extra'))
def test_context_semantics_cross_checked_with_native_train_rows(source,change):
    p=Path(source.context_ref['path']);row=json.loads(p.read_bytes())
    if change=='ordinal':row['source_candidate_record_ordinal']=True
    elif change=='record':row['source_record_id']='wrong'
    elif change=='count':row['observed_source_role_EOS_symbols']=2
    elif change=='axis':row['vocabulary_axis_sha256']='a'*64
    elif change=='evidence':row['grouping_keys_are_audit_only']=False
    elif change=='provenance':row['current_sequence_provenance']={}
    else:row['secret_target_input']='forbidden'
    p.write_bytes(fixture.wire(row)+b'\n');source.context_ref=fixture.reference(p)
    with pytest.raises(ValueError):load(source)

@pytest.mark.parametrize('field,value', [('split','validation'),('admitted_for_task',True),('model_adapter_supported',True),('native_payload_sha256','0'*64),('record_id','wrong')])
def test_tampered_native_rows_not_promoted(source,field,value):
    component=source.component;shard=component['shards'][0];p=Path(shard['path']);rows=pq.read_table(p).to_pylist();rows[0][field]=value;pq.write_table(pa.Table.from_pylist(rows),p)
    new=fixture.reference(p)|{k:v for k,v in shard.items()if k not in ('path','bytes','sha256')};component['shards'][0]=new;component['group_sidecars'][0]['source_shard']=new
    source.component_ref=fixture.write(Path(source.component_ref['path']),component);source.packet['component_manifest']=source.component_ref;source.packet['coordinator_review']['component_manifest_sha256']=source.component_ref['sha256'];source.review_ref=fixture.write(source.review_path,source.packet)
    with pytest.raises(ValueError):load(source)

def test_no_symlink_source_references(source):
    real=Path(source.context_ref['path']);link=real.with_name('linked.jsonl');link.symlink_to(real);source.context_ref=dict(source.context_ref,path=str(link))
    with pytest.raises(ValueError):load(source)


def test_finite_source_passes_and_exclusions_preserve_original_ordinals(source):
    material=load(source);policy=MCSAFinitePassPolicy(2,2,29)
    windows=tuple(material.iter_batches(policy))
    assert len(windows)==2
    assert tuple(w.pass_index for w in windows)==(0,1)
    assert all(w.original_candidate_row_ordinals==(0,) and w.observed_targets==3 for w in windows)
    for bad in ((True,),(-1,),(1,),(0,0),(0,),[0]):
        with pytest.raises(ValueError):tuple(material.iter_batches(policy,excluded_ordinals=bad))


def test_materialization_context_is_part_of_native_executable_identity(source):
    from mora3.research.mup_campaign import executable_adapter_sha256
    legacy=MCSACampaignTrainRoute(source.review_path,source.review_ref['sha256'],vocabulary=source.vocab,source_snapshot_sha256='a'*64,pass_policy=MCSAFinitePassPolicy(3,2,29),reviewed_payload_max_tokens=8192)
    material=load(source)
    isolated=MCSACampaignTrainRoute.from_train_materialization(material,admission=legacy.admission,source_snapshot_sha256='a'*64,pass_policy=MCSAFinitePassPolicy(3,2,29),reviewed_payload_max_tokens=8192)
    root=Path(__file__).resolve().parents[2]
    assert isolated.binding()==legacy.binding()  # mathematical source/cursor semantics preserved
    assert executable_adapter_sha256(isolated.prepare,root=root)!=executable_adapter_sha256(legacy.prepare,root=root)
    semantics=isolated.train_materialization_binding()
    assert semantics['source_context_sha256']==source.context_ref['sha256']
    assert semantics['source_component_sha256']==source.component_ref['sha256']
    assert semantics['materializer_implementation_sha256']==fixture.hashlib.sha256((root/'src/mora3/data/mcsa_train_materialization.py').read_bytes()).hexdigest()
    assert legacy.train_materialization_binding() is None


def test_native_materialized_route_cannot_outlive_context_revision(source):
    token=MCSARoleListAdmission.from_review_packet(source.review_path,source.review_ref['sha256'],vocabulary=source.vocab)
    material=load(source)
    route=MCSACampaignTrainRoute.from_train_materialization(material,admission=token,source_snapshot_sha256='a'*64,pass_policy=MCSAFinitePassPolicy(1,2,29),reviewed_payload_max_tokens=8192)
    p=Path(source.context_ref['path']);p.write_bytes(p.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='changed'):tuple(route.iter_from(route.initial_cursor))


def test_prepared_window_refuses_source_context_drift_before_owner_access(source):
    token=MCSARoleListAdmission.from_review_packet(source.review_path,source.review_ref['sha256'],vocabulary=source.vocab)
    route=MCSACampaignTrainRoute.from_train_materialization(load(source),admission=token,source_snapshot_sha256='a'*64,pass_policy=MCSAFinitePassPolicy(1,2,29),reviewed_payload_max_tokens=8192)
    window=next(route.iter_from(route.initial_cursor));prepared=route.prepare(window)
    p=Path(source.context_ref['path']);p.write_bytes(p.read_bytes()+b' ')
    with pytest.raises(ValueError,match='changed'):route.prepare(window)
    with pytest.raises(ValueError,match='changed'):prepared.evaluate(object())


def test_bound_materializer_version_cannot_change_after_route_construction(source,monkeypatch):
    from mora3.data import mcsa_train_materialization as module
    token=MCSARoleListAdmission.from_review_packet(source.review_path,source.review_ref['sha256'],vocabulary=source.vocab)
    route=MCSACampaignTrainRoute.from_train_materialization(load(source),admission=token,source_snapshot_sha256='a'*64,pass_policy=MCSAFinitePassPolicy(1,2,29),reviewed_payload_max_tokens=8192)
    fake=source.root/'test-only-alternative.py';fake.write_text('# Not the original implementation')
    monkeypatch.setattr(module,'__file__',str(fake))
    with pytest.raises(ValueError,match='implementation changed'):route.train_materialization_binding()
