import hashlib,json,sys
from pathlib import Path
from copy import deepcopy
import pytest
sys.path.insert(0,'/home/iska/Documents/amelie/bio/Mora-3/scripts')
from data_intake_full_rnacentral_tasks import full_native_prior_envelope,native_effective_groups
from data_intake_corpus import CorpusTask
from data_intake_nucleotide_prior import sequence_groups,prior_identity


def inputs(sequence='aCGTUN'):
 g=sequence_groups(sequence);state=dict(native_sha=bytes.fromhex(g['native_sha256']),uppercase_sha=bytes.fromhex(g['uppercase_sha256']),rc_sha=bytes.fromhex(g['reverse_complement_base_order_group']),reuse_kind='new',task_id=prior_identity(sequence,'RNA_reference_native_symbols',None))
 source=dict(native_payload='immutable/native.parquet',native_payload_sha256='pinned',native_row_index=7,original_record_index=42,original_FASTA='/immutable/original.fasta.gz',source_identifier='URS001',native_header='URS001 source classification',source_status='inactive',source_release='RNAcentral28.0',source_rights='CC0-1.0',original_source_record_id='native-original:42')
 return sequence,state,source


def test_full_native_payload_and_stable_ID_recipe():
 seq,state,source=inputs('aCGTUN'*20000)
 row=full_native_prior_envelope(seq,state,source,['URS001','URS_ALIAS'])
 d=json.loads(row['native_payload']);assert d['targets']=={'source_native_sequence':seq}
 assert row['task_id']==state['task_id'] and row['full_axis_values']==len(seq)
 assert CorpusTask.from_bytes(row['native_payload']).canonical_bytes()==row['native_payload']
 assert row['record_id']=='RNAcentral28-native-sequence:'+state['native_sha'].hex()
 assert d['audit']['source_ref']['original_source_record_id']=='native-original:42'


def test_target_length_hash_headers_status_and_aliases_are_audit_only():
 a=inputs();b=inputs('ACGU')
 x=full_native_prior_envelope(*a,['URS001']);b[2].update(native_header='TARGET HINT',source_status='active',source_identifier='URS999')
 y=full_native_prior_envelope(*b,['URS999'])
 assert CorpusTask.from_bytes(x['native_payload']).input_bytes()==CorpusTask.from_bytes(y['native_payload']).input_bytes()
 assert x['native_payload']!=y['native_payload']
 assert json.loads(x['native_payload'])['conditions']=={'prior_domain':'RNA_reference_native_symbols','source_organism':None}


def test_all_literal_source_identifiers_and_case_guards_are_retained():
 seq,state,source=inputs();groups=native_effective_groups(state,['URS_ALIAS','URS001','URS001'])
 assert 'nucleotide_native:'+state['native_sha'].hex() in groups
 assert 'nucleotide_upper:'+state['uppercase_sha'].hex() in groups
 assert 'nucleotide_derived_RC_baseorder:'+state['rc_sha'].hex() in groups
 assert {x for x in groups if x.startswith('rnacentral_native_identifier:')}=={'rnacentral_native_identifier:URS001','rnacentral_native_identifier:URS_ALIAS'}


def test_reused_prior_and_inconsistent_source_identity_cannot_serialize_new_task():
 seq,state,source=inputs();state['reuse_kind']='Rfam_existing'
 with pytest.raises(ValueError,match='new'):full_native_prior_envelope(seq,state,source,['URS001'])
 seq,state,source=inputs();state['native_sha']=bytes(32)
 with pytest.raises(ValueError,match='identity'):full_native_prior_envelope(seq,state,source,['URS001'])


def test_exact_source_identifier_must_be_in_final_URS_group_ledger():
 seq,state,source=inputs()
 with pytest.raises(ValueError,match='identifier'):full_native_prior_envelope(seq,state,source,['URS_OTHER'])


def test_source_status_is_not_negative_activity_or_experiment():
 row=full_native_prior_envelope(*inputs(),['URS001']);receipt=json.loads(row['admission_receipt_json'])
 assert row['split']=='train' and row['model_adapter_supported'] is False
 assert receipt['new_independent_experiments']==0
 assert receipt['source_status_is_not_activity_label'] is True
 assert json.loads(row['condition_context_json'])['physical_material_modification_context_known'] is False


def test_native_symbol_counts_keep_T_U_and_ambiguity_distinct_from_model_tokens():
 from data_intake_full_rnacentral_tasks import native_symbol_units
 u=native_symbol_units('aCGTUNRY')
 assert u==dict(new_distinct_target_native_symbols=8,biological_ACGTU_positions=5,ambiguity_positions=3,native_T_symbols=1,native_U_symbols=1,new_native_sequences_longer_than8192=0)
 assert native_symbol_units('N'*10000)['new_native_sequences_longer_than8192']==1
