"""Real original AZ logD train-corpus qualification; not a counted training trial."""
import json
from pathlib import Path
import pytest
from mora3.research.common_source_logd_complete import (
    audit_original_logd_complete_train, reference_logd_original, declared_logd_group_census,
)

ROOT=Path('/home/iska/Documents/amelie/bio/Mora-3')

@pytest.fixture(scope='module')
def complete():
    return audit_original_logd_complete_train(ROOT)

def test_all_original_az_logd_train_shards_exhausted_and_rebound(complete):
    assert complete['original_train_rows']==4184
    assert complete['original_train_shards']==17
    assert complete['original_native_observed_targets']==4184
    assert complete['all_train_candidate_common_equal']
    assert complete['all_train_independent_numeric_equal']
    assert complete['both_native_iterators_exhausted']
    assert complete['final_cursor_position']==[17,0]
    assert complete['independent_shard_restart_count']==32
    assert len(complete['rows'])==4184
    assert len({(r['source_shard_sha256'],r['source_ordinal']) for r in complete['rows']})==4184
    assert complete['rows'][255]['next_position']==[1,0]
    assert complete['rows'][-1]['next_position']==[17,0]

def test_original_metadata_census_preserves_shared_study_not_pretend_disjoint(complete):
    group=complete['original_declared_group_census']
    assert group['records']==4184
    assert group['all_declared_study_grouped_together']
    assert group['component_count']==1
    assert group['shared_assay_groups']>=1
    assert group['shared_document_groups']>=1
    assert complete['heldout_target_payload_decoded'] is False
    assert complete['global_chemical_split_admitted'] is False
    assert complete['training_admission'] is False
    assert complete['actual_proxy_runs']==complete['actual_target_runs']==complete['optimizer_updates']==0

def test_independent_logd_numeric_ref_does_not_invent_relation_or_units(complete):
    assert {r['source_relation'] for r in complete['rows']}=={None}
    assert {r['literal_source_unit'] for r in complete['rows']}=={None}
    for row in complete['rows']:
        assert row['original_float64_bits']==row['independent_float64_bits']
        assert row['original_numeric_lexeme']==row['independent_numeric_lexeme']
        assert len(row['native_source_fingerprint'])==64

def test_reference_rejects_duplicate_json_keys_and_boolean_imputation():
    payload=dict(schema='mora3.CorpusTask/v1',task='source_reported_experimental_logD7_4',
                 conditions=dict(canonical_isomeric_smiles='CO',pH=7.4,method='shake_flask',
                                 solvent_pair=['octan-1-ol','water']),
                 targets={'source_reported_logD7_4':1.8},
                 audit={'source_relation':None,'source_units':None,
                        'source_activity':{'value':'1.8','standard_value':'1.8','relation':None,
                                           'standard_relation':None,'standard_units':None,
                                           'units':None,'type':'LogD7.4','standard_type':'LogD7.4'}},
                 missing=['source_relation_unspecified'])
    raw=json.dumps(payload).encode()
    assert reference_logd_original(raw)['float64_bits']=='3ffccccccccccccd'
    with pytest.raises(ValueError):
        reference_logd_original(raw.replace(b'"pH": 7.4',b'"pH": 7.4, "pH": 7.4'))
    payload['targets']['source_reported_logD7_4']=True
    with pytest.raises(ValueError):reference_logd_original(json.dumps(payload).encode())
    payload['targets']['source_reported_logD7_4']=1.8
    payload['audit']['source_units']='kcal/mol'
    with pytest.raises(ValueError):reference_logd_original(json.dumps(payload).encode())

def test_declared_groups_connect_indirect_aliases_without_false_independence():
    data=({'record_id':'chembl37:activity:1','task_id':'a'*64,'groups':('assay:X','molecule:1')},
          {'record_id':'chembl37:activity:2','task_id':'b'*64,'groups':('molecule:1','source_document:Y')},
          {'record_id':'chembl37:activity:3','task_id':'c'*64,'groups':('source_document:Y','molecule:3')})
    assert declared_logd_group_census(data)['component_count']==1
    with pytest.raises(ValueError):declared_logd_group_census(data+(data[0],))
