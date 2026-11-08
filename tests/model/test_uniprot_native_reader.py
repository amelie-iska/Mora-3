import copy
import hashlib
import json
import pytest
from mora3.data import uniprot_annotations as u


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def fixture():
    e = {'primaryAccession':'P12345','entryType':'UniProtKB reviewed (Swiss-Prot)',
         'annotationScore':5,'entryAudit':{'entryVersion':2,'sequenceVersion':1},
         'sequence':{'value':'MAGK','length':4,'molWeight':400},
         'comments':[{'commentType':'BIOPHYSICOCHEMICAL PROPERTIES','kineticParameters':{'maximumVelocities':[{'velocity':3,'unit':'umol/min/mg','evidences':[{'evidenceCode':'ECO:0000269'}]}]}}],
         'extraAttributes':{'rare':{'unknown':[None,{'unit':'mM'}]}}, 'features':[]}
    identity={'primary_accession':'P12345','sequence_sha256':hashlib.sha256(b'MAGK').hexdigest(),'sequence_length':4,'annotation_score':5,'entry_version':2,'sequence_version':1}
    ref={'raw_file':'raw.json','raw_file_sha256':'a'*64,'results_index':0,'source_release':'2026_03','entry_version':2,'sequence_version':1}
    task={'schema':'mora3.CorpusTask/v1','task':'source_full_UniProt_annotation_prediction','conditions':{'canonical_sequence':'MAGK'},'targets':{'annotation_packet':{k:v for k,v in e.items() if k not in u.IDENTITY_KEYS},'provider_computed_sequence_properties':[{'parameter_type':'sequence_length','value':4,'unit':'residue','evidence_role':'provider_computed_from_canonical_sequence','source_pointer':'/sequence/length','source_value_json':'4','measurement_context':None},{'parameter_type':'sequence_mass','value':400,'unit':'Da','evidence_role':'provider_computed_from_canonical_sequence','source_pointer':'/sequence/molWeight','source_value_json':'400','measurement_context':None}]},'audit':{'identity':identity,'source_ref':ref},'missing':['validated_homology_heldout']}
    row={'record_id':'P12345','task_name':task['task'],'native_schema':task['schema'],'native_payload':canonical(task),'native_payload_sha256':hashlib.sha256(canonical(task)).hexdigest(),'admitted_for_task':True,'split':'train','source_sha256':'a'*64,'group_ids_json':json.dumps(['accession:P12345','sequence:'+identity['sequence_sha256']]),'admission_receipt_json':json.dumps({'license':'CC-BY-4.0','reviewed':True,'canonical_sequence_exact_baseline':True,'source_ref':ref})}
    return e,row


def test_lossless_nested_targets_and_detached_views():
    e,r=fixture();v=u.prediction_view(r,e)
    assert v.inputs()=={'canonical_sequence':'MAGK'}
    assert v.targets()['annotation_packet']['extraAttributes']==e['extraAttributes']
    x=v.targets();x['annotation_packet'].clear()
    assert v.targets()['annotation_packet']['comments'][0]['kineticParameters']['maximumVelocities'][0]['unit']=='umol/min/mg'
    assert v.family_group is None and not v.heldout_eligible


def test_target_mutation_cannot_enter_sequence_inputs():
    e,r=fixture();a=u.prediction_view(r,e)
    e['extraAttributes']['rare']['unknown'].append('hidden target')
    t=json.loads(r['native_payload']);t['targets']['annotation_packet']['extraAttributes']=e['extraAttributes'];r['native_payload']=canonical(t);r['native_payload_sha256']=hashlib.sha256(r['native_payload']).hexdigest()
    b=u.prediction_view(r,e)
    assert a.inputs()==b.inputs() and a.targets()!=b.targets()


@pytest.mark.parametrize('change', ['hash','admission','sequence','targets','extra_input','groups','heldout'])
def test_fail_closed(change):
    e,r=fixture();t=json.loads(r['native_payload'])
    if change=='hash':r['native_payload_sha256']='0'*64
    elif change=='admission':r['admitted_for_task']=False
    elif change=='sequence':e['sequence']['value']='MASK'
    elif change=='targets':t['targets']['annotation_packet']['features']=[{'invented':True}]
    elif change=='extra_input':t['conditions']['name']='leaked label'
    elif change=='groups':r['group_ids_json']='[]'
    elif change=='heldout':r['split']='test'
    if change in ('targets','extra_input'):
        r['native_payload']=canonical(t);r['native_payload_sha256']=hashlib.sha256(r['native_payload']).hexdigest()
    with pytest.raises(ValueError):u.prediction_view(r,e)


def test_snapshot_io_checks_raw_and_shard_hashes(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    e,r=fixture()
    raw=canonical({'results':[e]});(tmp_path/'raw.json').write_bytes(raw)
    digest=hashlib.sha256(raw).hexdigest()
    t=json.loads(r['native_payload']);t['audit']['source_ref']['raw_file_sha256']=digest
    receipt=json.loads(r['admission_receipt_json']);receipt['source_ref']['raw_file_sha256']=digest
    r.update(native_payload=canonical(t),native_payload_sha256=hashlib.sha256(canonical(t)).hexdigest(),source_sha256=digest,admission_receipt_json=json.dumps(receipt))
    pq.write_table(pa.Table.from_pylist([r]),tmp_path/'tasks.parquet')
    shard=(tmp_path/'tasks.parquet').read_bytes()
    m={'source_release':'2026_03','license':'CC-BY-4.0','source_only_train_split':True,'homology_and_source_parent_heldout_unresolved':True,'source_page_receipts':[{'path':'raw.json','bytes':len(raw),'sha256':digest,'source_release':'2026_03','compression':'none','entries':1}],'task_shards':[{'path':'tasks.parquet','bytes':len(shard),'sha256':hashlib.sha256(shard).hexdigest(),'rows':1,'split':'train'}]}
    p=tmp_path/'manifest.json';p.write_text(json.dumps(m))
    assert list(u.read_prediction_views(p,root=tmp_path,limit=1))[0].inputs()=={'canonical_sequence':'MAGK'}
    (tmp_path/'raw.json').write_bytes(raw+b' ')
    with pytest.raises(ValueError,match='integrity'):list(u.read_prediction_views(p,root=tmp_path,limit=1))
    (tmp_path/'raw.json').write_bytes(raw)
    (tmp_path/'tasks.parquet').write_bytes(shard+b' ')
    with pytest.raises(ValueError,match='integrity'):list(u.read_prediction_views(p,root=tmp_path,limit=1))


def test_source_pointer_path_cannot_escape_project(tmp_path):
    with pytest.raises(ValueError,match='outside project'):
        u._verified_bytes(tmp_path,{'path':'../outside.json'})
