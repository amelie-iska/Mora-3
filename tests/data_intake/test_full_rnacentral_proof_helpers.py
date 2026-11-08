from copy import deepcopy
from pathlib import Path
import hashlib,json,sys
import pytest
sys.path.insert(0,'/home/iska/Documents/amelie/bio/Mora-3/scripts')
from data_intake_full_rnacentral_tasks import full_native_prior_envelope
from data_intake_full_rnacentral_proof_helpers import verify_native_task,overlay_literal_groups
from data_intake_nucleotide_prior import sequence_groups,prior_identity
from data_intake_corpus import canonical


def fixture():
 seq='aCGTUN'*2000;g=sequence_groups(seq);state=dict(task_id=prior_identity(seq,'RNA_reference_native_symbols',None),native_sha=bytes.fromhex(g['native_sha256']),uppercase_sha=bytes.fromhex(g['uppercase_sha256']),rc_sha=bytes.fromhex(g['reverse_complement_base_order_group']),reuse_kind='new',first_file=2,first_row=7);src=dict(native_sequence=seq,native_sequence_sha256=g['native_sha256'],native_length=len(seq),sequence_alphabet_pass=True,source_record_index=100007,source_payload='/immutable/original.fasta.gz',source_identifier='URS001',native_header='URS001 original description',source_record_id='native:100007');f=dict(path='immutable/native.parquet',sha256='pinned',component='inactive');sr=dict(native_payload=f['path'],native_payload_sha256='pinned',native_row_index=7,original_record_index=100007,original_FASTA=src['source_payload'],source_identifier='URS001',native_header=src['native_header'],source_status='inactive',source_release='RNAcentral28.0',source_rights='CC0-1.0',original_source_record_id=src['source_record_id']);r=full_native_prior_envelope(seq,state,sr,['URS001','URS_ALIAS']);idx=dict(source_file=2,source_row=7,native_sha=state['native_sha'],reuse_kind='new');gr=dict(task_id=r['task_id'],effective_groups_json=r['group_ids_json']);return [r,src,f,2,7,state,idx,['URS001','URS_ALIAS'],gr]


def test_full_native_target_provenance_and_group_readback():
 a=fixture();v=verify_native_task(*a)
 assert v['native_length']==12000 and v['task_id']==a[0]['task_id']

@pytest.mark.parametrize('change',['crop','condition_leak','source_ref','group','first_occurrence','index_location','experiment_claim'])
def test_independent_readback_detects_semantic_corruption_even_with_updated_payload_hash(change):
 a=fixture();r=a[0];d=json.loads(r['native_payload'])
 if change=='crop':d['targets']['source_native_sequence']=d['targets']['source_native_sequence'][:8192]
 elif change=='condition_leak':d['conditions']['target_length']=12000
 elif change=='source_ref':d['audit']['source_ref']['source_status']='active'
 elif change=='group':a[-1]['effective_groups_json']=json.dumps(['rnacentral_native_identifier:PHANTOM'])
 elif change=='first_occurrence':a[5]['first_row']=8
 elif change=='index_location':a[6]['source_file']=3
 elif change=='experiment_claim':
  receipt=json.loads(r['admission_receipt_json']);receipt['new_independent_experiments']=1;r['admission_receipt_json']=json.dumps(receipt)
 r['native_payload']=canonical(d);r['native_payload_sha256']=hashlib.sha256(r['native_payload']).hexdigest()
 with pytest.raises(ValueError):verify_native_task(*a)


def test_group_overlay_preserves_original_unknown_fields_and_does_not_rewrite_original():
 old=dict(task_id='old',effective_groups_json=json.dumps(['prior-native','original-parent']),rare={'scope':[None,{'x':1}]});copy=deepcopy(old);new=overlay_literal_groups(old,['new-URS','prior-native'])
 assert old==copy and new['rare']==old['rare'] and new['task_id']=='old'
 assert json.loads(new['effective_groups_json'])==['new-URS','original-parent','prior-native']
