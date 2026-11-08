import copy,sys,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'scripts'))
import pytest
from data_intake_admet_endpoint_review import review_expansion_endpoint,unresolved_asap_endpoint,apply_endpoint_review,_SCHEMA
from data_intake_admet_observations import native_cell,task_view

CARD='\n'.join(x['line'] for x in _SCHEMA.values())
REF=dict(path='source-schema',bytes=99,sha256='a'*64)
def rec(endpoint='MPPB'):
 return dict(schema='mora3.ADMETNativeObservation/v1',entity=dict(molecule_class='small_molecule',representation_type='native_SMILES',representation='CCO'),endpoint=dict(endpoint_key='ExpansionRx:'+endpoint,source_endpoint=endpoint,assay_protocol_context=dict(hidden_target='unsafe',source_ref='source-audit')),cell=native_cell('>90'),known_context={},source_ref=dict(native_entity_namespace='ExpansionRx'),native_source_record=dict(secret='retained'),evidence_role='source_curated_measurement',source_split=dict(protocol='train'),admission=dict(source_rights=True,exact_identity=False,endpoint_definition=False,task_units_or_declared_native_coordinate=False,split_group_review=False))
def test_percent_unbound_not_percent_bound_and_no_conversion():
 r=review_expansion_endpoint('MPPB',card_text=CARD,card_ref=REF)
 assert r['unit_from_official_schema']=='% Unbound' and 'unbound' in r['source_readout']
 q=apply_endpoint_review(rec(),r);assert q['cell']['decimal_literal']=='90' and q['cell']['relation']=='>'
 assert q['endpoint']['unit']=='% Unbound' and q['cell']['native_literal']=='>90'
 assert q['endpoint_source_review_audit']['conversions_performed']==[]
 assert q['admission']['exact_identity'] is False and q['admission']['split_group_review'] is False
 with pytest.raises(ValueError):task_view(q,direction='entity_to_measurement')
def test_logd_and_ratio_remain_unknown_physical_context():
 for name in ['LogD','Caco-2 Permeability Efflux']:
  r=review_expansion_endpoint(name,card_text=CARD,card_ref=REF);assert r['unit_from_official_schema'] is None and r['native_coordinate_only']
  q=apply_endpoint_review(rec(name),r);assert q['endpoint']['unit_basis']=='unresolved' and q['endpoint']['unit'] is None
  assert 'pH' not in q['known_context'] and 'source_measurement_coordinate' in q['known_context']
def test_papp_scaled_units_are_literal_schema_not_converted():
 r=review_expansion_endpoint('Caco-2 Permeability Papp A>B',card_text=CARD,card_ref=REF)
 assert r['unit_from_official_schema']=='10^-6 cm/s' and r['conversions_performed']==[]
 assert r['source_context']==dict(assay_system='Caco-2',assay_direction='A>B')
def test_microsomal_clearance_not_in_vivo_and_RLM_scope_stays_extra():
 r=review_expansion_endpoint('RLM CLint',card_text=CARD,card_ref=REF)
 assert r['clinical_or_in_vivo_PK_validation'] is False and r['source_context']['species']=='Rat'
 assert 'raw-only' in r['source_protocol_scope']
def test_proof_pointers_and_hidden_audit_fields_cannot_enter_conditions():
 a=review_expansion_endpoint('MPPB',card_text=CARD,card_ref=REF);b=copy.deepcopy(a);b['source_schema_ref']['path']='secret-target-reference'
 qa=apply_endpoint_review(rec(),a);qb=apply_endpoint_review(rec(),b)
 for q in [qa,qb]:q['admission'].update(exact_identity=True,split_group_review=True)
 for direction in ['entity_to_measurement','measurement_to_entity']:
  ta=task_view(qa,direction=direction);tb=task_view(qb,direction=direction);assert ta.input_bytes()==tb.input_bytes()
  assert b'secret-target-reference' not in ta.input_bytes() and b'unsafe' not in ta.input_bytes()
 assert qa['original_endpoint_projection_audit']==rec()['endpoint']
def test_source_review_cannot_cross_families_or_invent_missing_units():
 r=unresolved_asap_endpoint('KSOL',card_ref=REF)
 assert not r['task_units_or_declared_native_coordinate_gate'] and r['unit_from_official_schema'] is None
 with pytest.raises(ValueError):apply_endpoint_review(rec('KSOL'),r)
 with pytest.raises(ValueError):review_expansion_endpoint('MPPB',card_text=CARD.replace('% Unbound','% Bound'),card_ref=REF)

def test_forward_target_mutation_leaves_all_predictor_bytes_unchanged():
 q=apply_endpoint_review(rec(),review_expansion_endpoint('MPPB',card_text=CARD,card_ref=REF));q['admission'].update(exact_identity=True,split_group_review=True)
 altered=copy.deepcopy(q);altered['cell']=native_cell('<5');altered['native_source_record']={'MPPB':'<5','target_name':'hidden'}
 a=task_view(q,direction='entity_to_measurement');b=task_view(altered,direction='entity_to_measurement')
 assert a.input_bytes()==b.input_bytes() and json.loads(a.canonical_bytes())['targets']!=json.loads(b.canonical_bytes())['targets']
def test_inverse_native_identity_and_audit_mutations_leave_predictors_unchanged():
 q=apply_endpoint_review(rec(),review_expansion_endpoint('MPPB',card_text=CARD,card_ref=REF));q['admission'].update(exact_identity=True,split_group_review=True)
 altered=copy.deepcopy(q);altered['entity']['representation']='CCCC';altered['source_ref']['hidden_entity_id']='target-secret'
 a=task_view(q,direction='measurement_to_entity');b=task_view(altered,direction='measurement_to_entity')
 assert a.input_bytes()==b.input_bytes() and json.loads(a.canonical_bytes())['targets']!=json.loads(b.canonical_bytes())['targets']
