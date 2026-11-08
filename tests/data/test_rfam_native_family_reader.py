import hashlib
import json
import pytest
from mora3.data import rfam_family as rf


def test_prediction_excludes_target_and_audit():
    d=dict(schema='mora3.CorpusTask/v1', task='Rfam_source_family_prediction_prior',
           conditions={'source_native_sequence':'AcgTU'}, targets=dict(source_Rfam_family_ids=['RF00001'],source_clan_ids=[],family_label_policy=rf.LABEL_POLICY),audit={})
    assert rf.project_family_task(d,'prediction') == ({'source_native_sequence':'AcgTU'},('RF00001',),())
    d['audit']['secret_target_metadata']='ignored'
    d['targets']['source_Rfam_family_ids']=['RF00002']
    assert rf.project_family_task(d,'prediction')[0] == {'source_native_sequence':'AcgTU'}


@pytest.mark.parametrize('change', ['alphabet','extra_input','negative','empty','duplicate'])
def test_rejects_invalid_prediction(change):
    d=dict(schema='mora3.CorpusTask/v1', task='Rfam_source_family_prediction_prior',conditions={'source_native_sequence':'AcgTU'},targets=dict(source_Rfam_family_ids=['RF00001'],source_clan_ids=[],family_label_policy=rf.LABEL_POLICY))
    if change=='alphabet':d['conditions']['source_native_sequence']='ACZ'
    if change=='extra_input':d['conditions']['family']='RF00001'
    if change=='negative':d['targets']['family_label_policy']='all other negative'
    if change=='empty':d['targets']['source_Rfam_family_ids']=[]
    if change=='duplicate':d['targets']['source_Rfam_family_ids']*=2
    with pytest.raises(ValueError):rf.project_family_task(d,'prediction')


def test_generation_excludes_native_target():
    d=dict(schema='mora3.CorpusTask/v1',task='Rfam_source_family_generation_prior',conditions=dict(desired_source_Rfam_family_ids=['RF00001'],desired_source_clan_ids=['CL00001']),targets={'source_native_sequence':'AcgTU'})
    a=rf.project_family_task(d,'generation')
    d['targets']['source_native_sequence']='UUUUUUUU'
    assert rf.project_family_task(d,'generation')[0] == a[0]
    with pytest.raises(ValueError):rf.project_family_task(d,'invented')


def test_manifest_hash_fails_closed(tmp_path):
    p=tmp_path/'manifest.json';p.write_text('{}')
    with pytest.raises(ValueError,match='manifest'):
        rf.read_rfam_families('manifest.json',root=tmp_path,expected_manifest_sha256='0'*64,direction='prediction',limit=1)


@pytest.fixture
def admitted(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from mora3.data.uniprot_annotations import _canonical as canon
    sha=lambda b:hashlib.sha256(b).hexdigest()
    seq='AcgTU';record=sha(seq.encode())
    d=dict(schema='mora3.CorpusTask/v1',task='Rfam_source_family_prediction_prior',conditions={'source_native_sequence':seq},targets=dict(source_Rfam_family_ids=['RF00001'],source_clan_ids=[],family_label_policy=rf.LABEL_POLICY),missing=['unlisted_family_negative_truth'],audit={'test_only':True})
    tid=sha(canon([d['task'],d['conditions'],d['targets']]))
    groups=['nucleotide_native:'+record,'rfam_model:Rfam15.1:RF00001']
    r=dict(task_id=tid,record_id=record,source_family='Rfam15.1-source-family-prior',source_revision='Rfam15.1 January2026',source_sha256='a'*64,task_name=d['task'],native_schema=d['schema'],native_payload=canon(d),native_payload_sha256=sha(canon(d)),split='train',split_policy=rf.POLICY,evidence_class='source_covariance_model_assigned_reference_family_annotation;notexperimentaltruth',admitted_for_task=True,full_axis_values=len(seq),group_ids_json=json.dumps(groups),missing_context_json=json.dumps(d['missing']),condition_context_json=json.dumps(dict(segmentation_applicability=False,profile_or_image_pair=False,source_confidence_or_name_or_headers_in_predictors=False,target_length_audit_only=True)),admission_receipt_json=json.dumps(dict(exact_identifier_join=True,all_native_sequence_collisions_for_key_quarantined=True,source_license='CC0-1.0',source_family_assignments_not_experiment=True,negative_family_targets=False)))
    def write(change=None):
        row=dict(r);group=dict(task_id=tid,effective_groups_json=json.dumps(groups))
        if change:change(row,group)
        refs=[]
        for name,rows in [('task.parquet',[row]),('group.parquet',[group])]:
            p=tmp_path/name;pq.write_table(pa.Table.from_pylist(rows),p)
            refs.append(dict(path=name,sha256=sha(p.read_bytes()),bytes=p.stat().st_size,rows=1))
        m=dict(schema='mora3.RfamFamilyAnnotationAdmission/v1',source_license='CC0-1.0',source_release='Rfam15.1 January2026',negative_labels_available=False,family_ids=['RF00001'],context_manifest={'sha256':'a'*64},shards=[refs[0]],effective_group_lookup={'task.parquet':refs[1]})
        p=tmp_path/'manifest.json';p.write_bytes(canon(m))
        return rf.read_rfam_families(p.name,root=tmp_path,expected_manifest_sha256=sha(p.read_bytes()),direction='prediction',limit=1)
    return write


def test_admission_preserves_full_targets_and_provenance(admitted):
    v=admitted()[0]
    assert v.inputs()=={'source_native_sequence':'AcgTU'}
    assert v.targets()['source_Rfam_family_ids']==['RF00001']
    assert v.provenance()['heldout_eligible'] is False
    assert v.provenance()['family_axis']==['RF00001']


@pytest.mark.parametrize('field,value', [('split','test'),('admitted_for_task',False),('source_sha256','b'*64),('full_axis_values',1),('record_id','b'*64),('native_payload_sha256','0'*64),('condition_context_json','{}'),('missing_context_json','[]')])
def test_envelope_mutations_fail_closed(admitted,field,value):
    with pytest.raises(ValueError):admitted(lambda r,g:r.update({field:value}))


@pytest.mark.parametrize('change', [lambda r,g:g.update(task_id='wrong'),lambda r,g:g.update(effective_groups_json='[]'),lambda r,g:g.update(effective_groups_json='["unrelated"]')])
def test_group_mutations_fail_closed(admitted,change):
    with pytest.raises(ValueError):admitted(change)
