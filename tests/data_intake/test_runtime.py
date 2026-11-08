import importlib.util,hashlib,json,sys
from pathlib import Path
import pytest,torch
root=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists());sys.path[:0]=[str(root/'src'),str(root/'scripts')]
spec=importlib.util.spec_from_file_location('runtime',root/'scripts/data_intake_runtime.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
from mora3.data.array_tasks import ArrayTask,OrderedArray
from data_intake_molecule_profile import molecule_profile_task

def envelope(raw,schema,supported=True):return dict(task_id='t',record_id='r',native_payload=raw,native_payload_sha256=hashlib.sha256(raw).hexdigest(),native_schema=schema,admitted_for_task=True,split='train',source_family='fixture',source_revision='v1',source_sha256='0'*64,group_ids_json='["molecule:x"]',group_metadata_status='manifest_verified_effective_groups',model_adapter_supported=supported)
def profile():return dict(name='profile',axis_ids=[str(i) for i in range(977)],values=[.2]*977,axis_identity='fixture',units='source_normalized',status='observed')

def test_native_array_runtime_preserves_full_axis_masks_and_audit_exclusion():
    a=OrderedArray('profile',tuple(map(str,range(1783))),tuple([.25]*1783),'fixture','source_normalized','observed');t=ArrayTask((a,),(('profile','target'),),'0'*64);r=envelope(t.canonical_bytes(),'mora3.ArrayTask/v1')
    p=m.project_corpus_row(r,manifest_sha256='1'*64,for_training=True)
    assert not p['model_inputs'] and p['supervision_targets']['profile'].shape==(1783,)
    assert p['supervision_targets']['profile'].dtype==torch.float64
    assert p['supervision_masks']['profile'].all() and len(p['target_axes']['profile'])==1783
    r['record_id']='AUDIT_CHANGED';r['group_ids_json']='["changed"]';q=m.project_corpus_row(r,manifest_sha256='1'*64,for_training=True)
    assert m.input_fingerprint(p)==m.input_fingerprint(q) and p['resume_binding']!=q['resume_binding']

def test_custom_forward_data_projection_excludes_targets_audit_and_keeps_missingness():
    a=profile();t=molecule_profile_task(a,'CCO','L1000','molecule_to_profile',audit={'MOA':'name'});r=envelope(t.canonical_bytes(),'mora3.CorpusTask/v1',False)
    p=m.project_corpus_row(r,manifest_sha256='1'*64,allow_profile_inspection=True)
    a['values'][0]=99;t2=molecule_profile_task(a,'CCO','L1000','molecule_to_profile',audit={'MOA':'changed'});r2=envelope(t2.canonical_bytes(),'mora3.CorpusTask/v1',False);q=m.project_corpus_row(r2,manifest_sha256='1'*64,allow_profile_inspection=True)
    assert m.input_fingerprint(p)==m.input_fingerprint(q)
    assert p['supervision_targets']['profile'].shape==(977,) and p['context_masks']['physical_dose'] is False and p['context_masks']['cell_line'] is False
    assert p['model_adapter_supported'] is False
    with pytest.raises(ValueError,match='model adapter'):m.project_corpus_row(r,manifest_sha256='1'*64,allow_profile_inspection=True,for_training=True)

def test_custom_inverse_molecule_target_never_enters_profile_inputs():
    a=molecule_profile_task(profile(),'CCO','L1000','profile_to_molecule');b=molecule_profile_task(profile(),'CCN','L1000','profile_to_molecule')
    p=m.project_corpus_row(envelope(a.canonical_bytes(),'mora3.CorpusTask/v1',False),manifest_sha256='1'*64,allow_profile_inspection=True);q=m.project_corpus_row(envelope(b.canonical_bytes(),'mora3.CorpusTask/v1',False),manifest_sha256='1'*64,allow_profile_inspection=True)
    assert m.input_fingerprint(p)==m.input_fingerprint(q) and p['supervision_targets']!=q['supervision_targets']

def test_training_and_group_receipt_guards_fail_closed():
    t=ArrayTask((OrderedArray('x',('a',),(1.,),'fixture','u','observed'),),(('x','target'),),'0'*64);r=envelope(t.canonical_bytes(),'mora3.ArrayTask/v1');r['split']='test'
    with pytest.raises(ValueError,match='heldout'):m.project_corpus_row(r,manifest_sha256='1'*64,for_training=True)
    r['split']='train';r['group_metadata_status']='absent'
    with pytest.raises(ValueError,match='group'):m.project_corpus_row(r,manifest_sha256='1'*64)

def test_forward_target_only_units_do_not_enter_runtime_predictors():
    a=profile();b=profile();b['units']='different_target_metadata'
    p=m.project_corpus_row(envelope(molecule_profile_task(a,'CCO','L1000','molecule_to_profile').canonical_bytes(),'mora3.CorpusTask/v1',False),manifest_sha256='1'*64,allow_profile_inspection=True)
    q=m.project_corpus_row(envelope(molecule_profile_task(b,'CCO','L1000','molecule_to_profile').canonical_bytes(),'mora3.CorpusTask/v1',False),manifest_sha256='1'*64,allow_profile_inspection=True)
    assert m.input_fingerprint(p)==m.input_fingerprint(q)

def test_declared_flag_cannot_turn_custom_inspection_into_a_model_adapter():
    r=envelope(molecule_profile_task(profile(),'CCO','L1000','molecule_to_profile').canonical_bytes(),'mora3.CorpusTask/v1',True)
    with pytest.raises(ValueError,match='model adapter'):
        m.project_corpus_row(r,manifest_sha256='1'*64,allow_profile_inspection=True,for_training=True)

def test_model_generated_profile_cannot_masquerade_as_observational_source():
    a=profile();a['status']='model_generated'
    r=envelope(molecule_profile_task(a,'CCO','L1000','molecule_to_profile').canonical_bytes(),'mora3.CorpusTask/v1',False)
    with pytest.raises(ValueError,match='observed source'):
        m.project_corpus_row(r,manifest_sha256='1'*64,allow_profile_inspection=True)
