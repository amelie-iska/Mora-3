"""Catch unknown-as-negative, leakage, axis drift and admission bypasses."""
import copy
import hashlib
import json
import pytest

AXIS = ['NR-AhR','NR-AR','NR-AR-LBD','NR-Aromatase','NR-ER','NR-ER-LBD',
        'NR-PPAR-gamma','SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53']
FORWARD = 'source_Tox21_molecule_to_observed_assay_calls'
INVERSE = 'source_Tox21_observed_call_pattern_to_molecule_prior'

def seal(row, task):
    row = copy.deepcopy(row)
    row['native_payload'] = json.dumps(task, sort_keys=True).encode()
    row['native_payload_sha256'] = hashlib.sha256(row['native_payload']).hexdigest()
    return row

def fixture(direction=FORWARD, split='train'):
    vals = [0,1,None] * 4; mask = [True,True,False] * 4
    conditions = dict(canonical_isomeric_smiles='CCO', known_assay_axis=AXIS) if direction == FORWARD else dict(known_assay_axis=AXIS, desired_source_call_values=vals, desired_source_call_mask=mask)
    targets = dict(source_call_values=vals, source_call_mask=mask) if direction == FORWARD else dict(canonical_isomeric_smiles='CCO')
    ref = dict(native_jsonl='source.jsonl', native_jsonl_sha256='a'*64, source_row=0)
    role = dict(train='source_training',validation='source_leaderboard_testing',test='source_final_evaluation_structures')[split]
    task = dict(schema='mora3.CorpusTask/v1',task=direction,conditions=conditions,targets=targets,
        audit=dict(source_ref=ref,source_protocol_role=role,physical_dose=None,exposure_time=None,source_missing_not_negative=True),missing=['perassayphysicaldose_exposuretime_cellline'])
    receipt = dict(source_ref=ref,license='US-government-public-domainoriginalTox21data;officialNCATSproviderstatement',all12assayaxes_retained=True,physical_dose_time_missing=True,measuredcallannotation_prioronly=True) if split=='train' else dict(source_ref=ref,source_license='originalUSgovTox21publicdomain',source_protocol_split=True,known_chemical_train_overlap=False,prior_model_and_benchmark_exposure_unknown=True,not_final_unseen_biological_validation=True)
    row = dict(task_id='t', record_id='r',source_family='Tox21-original-source-call-prior',source_revision='pinned',source_sha256='a'*64,task_name=direction,native_schema=task['schema'],split=split,admitted_for_task=True,full_axis_values=12,group_ids_json='["molecule:m","scaffold:s"]',admission_receipt_json=json.dumps(receipt))
    return seal(row,task),task

def view(row):
    from mora3.data.tox21 import tox21_view
    return tox21_view(row,effective_groups=('molecule:m','scaffold:s'))

@pytest.mark.parametrize('direction',[FORWARD,INVERSE])
@pytest.mark.parametrize('split',['train','validation','test'])
def test_masks_roles_and_no_target_or_audit_leakage(direction,split):
    row,task=fixture(direction,split); v=view(row)
    assert v.values == (0,1,None)*4 and v.observed_mask == (True,True,False)*4
    assert v.provenance()['split']==split
    assert v.provenance()['independent_biological_validation'] is False
    task['audit']['hidden_label']='do not encode'
    if direction==FORWARD: task['targets']['source_call_values'][0]=1
    else: task['targets']['canonical_isomeric_smiles']='CCN'
    other=view(seal(row,task))
    assert v.inputs()==other.inputs() and v.binding_sha256!=other.binding_sha256
    x=v.inputs();x['extra']='mutation';assert 'extra' not in v.inputs()

@pytest.mark.parametrize('failure',['hash','axis','value','unknown_zero','bool_value','mask_int','empty','extra','admission','rights','groups','source','role'])
def test_rejects_corruption_and_unsupported_admission(failure):
    row,t=fixture()
    if failure=='hash': row['native_payload_sha256']='bad'
    elif failure=='axis': t['conditions']['known_assay_axis']=AXIS[::-1]
    elif failure=='value': t['targets']['source_call_values'][0]=2
    elif failure=='unknown_zero': t['targets']['source_call_values'][2]=0
    elif failure=='bool_value': t['targets']['source_call_values'][0]=False
    elif failure=='mask_int': t['targets']['source_call_mask'][0]=1
    elif failure=='empty': t['targets']=dict(source_call_values=[None]*12,source_call_mask=[False]*12)
    elif failure=='extra': t['conditions']['source_name']='target derived'
    elif failure=='admission': row['admitted_for_task']=False
    elif failure=='rights': row['admission_receipt_json']='{}'
    elif failure=='groups': row['group_ids_json']='["molecule:other"]'
    elif failure=='source': t['audit']['source_ref']['native_jsonl_sha256']='b'*64
    elif failure=='role': t['audit']['source_protocol_role']='source_final_evaluation_structures'
    if failure!='hash': row=seal(row,t)
    with pytest.raises(ValueError):view(row)

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_objective_tensors_mask_unknown_gradient_and_preserve_negative(device):
    import torch
    from mora3.data.tox21 import tox21_call_tensors
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA absent')
    row,_=fixture();values,mask=tox21_call_tensors([view(row)],role='target',device=device)
    assert values[0,:3].tolist()==[0.,1.,0.] and mask[0,:3].tolist()==[True,True,False]
    logits=torch.zeros_like(values,requires_grad=True)
    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,values,reduction='none')[mask].mean()
    loss.backward()
    assert logits.grad[0,:3].tolist()==[.0625,-.0625,0.]
    inverse=view(fixture(INVERSE)[0])
    with pytest.raises(ValueError):tox21_call_tensors([inverse],role='target',device=device)
    desired,dm=tox21_call_tensors([inverse],role='desired_condition',device=device)
    assert torch.equal(values,desired) and torch.equal(mask,dm)

def corpus(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    row,_=fixture();pq.write_table(pa.Table.from_pylist([row]),tmp_path/'s.parquet')
    pq.write_table(pa.Table.from_pylist([dict(task_id='t',effective_groups_json=row['group_ids_json'])]),tmp_path/'g.parquet')
    def rec(n):
        b=(tmp_path/n).read_bytes();return dict(path=n,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),rows=1)
    m=dict(shards=[dict(rec('s.parquet'),source_family=row['source_family'],split='train')],effective_group_lookup={'s.parquet':rec('g.parquet')})
    b=json.dumps(m).encode();(tmp_path/'m.json').write_bytes(b);return hashlib.sha256(b).hexdigest()

@pytest.mark.parametrize('damage',[None,'manifest','shard','sidecar'])
def test_pinned_bounded_reader(tmp_path,damage):
    from mora3.data.tox21 import read_tox21_views
    digest=corpus(tmp_path)
    if damage:
        name=dict(manifest='m.json',shard='s.parquet',sidecar='g.parquet')[damage]
        with (tmp_path/name).open('ab') as f:f.write(b'bad')
        with pytest.raises(ValueError):read_tox21_views('m.json',root=tmp_path,expected_manifest_sha256=digest,direction=FORWARD,split='train',limit=1)
    else:
        v=read_tox21_views('m.json',root=tmp_path,expected_manifest_sha256=digest,direction=FORWARD,split='train',limit=1)
        assert len(v)==1 and v[0].provenance()['manifest_sha256']==digest
        for limit in (0,True,257):
            with pytest.raises(ValueError):read_tox21_views('m.json',root=tmp_path,expected_manifest_sha256=digest,direction=FORWARD,split='train',limit=limit)

@pytest.mark.parametrize('damage',['identity','allocation','count','groups'])
def test_reader_rejects_internally_inconsistent_but_hash_valid_corpus(tmp_path,damage):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from mora3.data.tox21 import read_tox21_views
    corpus(tmp_path);m=json.loads((tmp_path/'m.json').read_bytes())
    if damage in ('identity','groups'):
        gr=dict(task_id='other' if damage=='identity' else 't',effective_groups_json='["molecule:wrong","scaffold:s"]' if damage=='groups' else '["molecule:m","scaffold:s"]')
        pq.write_table(pa.Table.from_pylist([gr]),tmp_path/'g.parquet')
        r=m['effective_group_lookup']['s.parquet'];raw=(tmp_path/'g.parquet').read_bytes();r.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    elif damage=='allocation':
        row,_=fixture(split='test');pq.write_table(pa.Table.from_pylist([row]),tmp_path/'s.parquet')
        raw=(tmp_path/'s.parquet').read_bytes();m['shards'][0].update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    else:m['shards'][0]['rows']=2
    raw=json.dumps(m).encode();(tmp_path/'m.json').write_bytes(raw)
    with pytest.raises(ValueError):read_tox21_views('m.json',root=tmp_path,expected_manifest_sha256=hashlib.sha256(raw).hexdigest(),direction=FORWARD,split='train',limit=1)
