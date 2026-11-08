import copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'scripts'))
import pytest
from data_intake_openadmet_admission import source_allocation,admitted_observation,source_task_views
from data_intake_admet_observations import native_cell
from data_intake_admet_endpoint_review import review_expansion_endpoint,_SCHEMA

def fixture(parent='source_train'):
 graph=dict(source_family='ExpansionRx',native_entity_id='native-1',native_row_index=2,native_parent_protocol=parent,source_graph_scope_json=json.dumps(dict(native_representation='CCO',native_predictor_identity_candidate=True,legacy_plain_SMILES_exact_comparison_eligible=True)),known_current_heldout_group_matches_json='[]',frozen_pilot_reservation_matches_json='[]',oldtrain_group_matches_json='[]')
 joint={k:graph[k] for k in ('source_family','native_entity_id','native_row_index','native_parent_protocol')};joint.update(any_benchmark_heldout_parent_source_families_json='[]' if parent=='source_train' else '["ExpansionRx"]',candidate_train_after_joint_known_source_guards=parent=='source_train')
 rec=dict(schema='mora3.ADMETNativeObservation/v1',entity=dict(molecule_class='small_molecule',representation_type='native_SMILES',representation='CCO'),endpoint=dict(endpoint_key='ExpansionRx:MPPB',source_endpoint='MPPB'),cell=native_cell('>90'),known_context={},source_ref=dict(native_entity_namespace='ExpansionRx',native_entity_id='native-1',native_row_index=2),native_source_record={'MPPB':'>90'},evidence_role='source_curated_measurement',source_split=dict(parent_protocol={'source_train':'native_train','source_test':'native_test','source_unallocated':'raw_only_unallocated'}[parent]),admission=dict(source_rights=True,exact_identity=False,endpoint_definition=False,task_units_or_declared_native_coordinate=False,split_group_review=False),source_native_axis_cell_multiplicity=1,independent_experimental_measurement_count=None)
 review=review_expansion_endpoint('MPPB',card_text='\n'.join(x['line'] for x in _SCHEMA.values()),card_ref={'path':'card'})
 return graph,joint,rec,review

def test_source_test_cannot_be_reclassified_train_by_candidate_boolean():
 g,j,r,e=fixture('source_test');j['candidate_train_after_joint_known_source_guards']=True
 assert source_allocation(g,j)[0]=='test'
def test_train_original_or_cross_source_heldout_parent_always_blocks():
 for sources in ['["ExpansionRx"]','["ASAP"]']:
  g,j,r,e=fixture();j['any_benchmark_heldout_parent_source_families_json']=sources
  assert source_allocation(g,j)[0] is None
def test_current_protected_and_frozen_groups_override_candidate_boolean():
 for key in ['known_current_heldout_group_matches_json','frozen_pilot_reservation_matches_json']:
  g,j,r,e=fixture();g[key]='["scaffold:protected"]';assert source_allocation(g,j)[0] is None
def test_known_old_train_exposure_disqualifies_cold_test_task():
 g,j,r,e=fixture('source_test');g['oldtrain_group_matches_json']='["scaffold:known"]';assert source_allocation(g,j)[0] is None
def test_unallocated_source_entities_and_ASAP_not_admitted():
 g,j,r,e=fixture('source_unallocated');assert source_allocation(g,j)[0] is None
 g,j,r,e=fixture();g['source_family']=j['source_family']='ASAP';assert source_allocation(g,j)[0] is None
def test_native_parent_identity_or_row_mismatch_never_silently_joins():
 for key,value in [('native_entity_id','other'),('native_row_index',3)]:
  g,j,r,e=fixture();g[key]=value
  with pytest.raises(ValueError):admitted_observation(r,g,j,e)
 g,j,r,e=fixture();r['source_split']['parent_protocol']='native_test'
 with pytest.raises(ValueError):admitted_observation(r,g,j,e)
def test_two_directions_reuse_one_censored_cell_without_independent_assay_claim():
 g,j,r,e=fixture();before=copy.deepcopy(r);out,_=admitted_observation(r,g,j,e);views=source_task_views(out)
 assert r==before and len(views)==2 and out['cell']['native_literal']=='>90' and out['independent_experimental_measurement_count'] is None
 assert out['admission']['model_use_authorized'] is False and out['admission']['global_novelty_or_study_disjointness_proven'] is False
 for task in views.values():
  d=json.loads(task.canonical_bytes());assert b'native-1' not in task.input_bytes();assert d['audit']['directions_reuse_one_source_observation']
def test_missing_or_computed_native_cells_not_experimental_objectives():
 g,j,r,e=fixture();r['cell']=native_cell('');assert admitted_observation(r,g,j,e)[0] is None
 g,j,r,e=fixture();r['evidence_role']='predicted';assert admitted_observation(r,g,j,e)[0] is None

def test_component_approval_binds_scope_script_and_no_master_or_training():
 from data_intake_openadmet_component import approved
 plan={'sha256':'plan'};script={'sha256':'script'};ok=dict(approved=True,scope='ExpansionRx-complete-eligible-native-observation-component-only',plan_sha256='plan',script_sha256='script',master_publication=False,model_training=False)
 approved(ok,plan,script)
 for key,bad in [('approved',False),('scope','other'),('plan_sha256','changed'),('script_sha256','changed'),('master_publication',True),('model_training',True)]:
  a=dict(ok);a[key]=bad
  with pytest.raises(ValueError):approved(a,plan,script)

def test_independent_wire_check_rejects_hidden_roles_or_changed_native_censor():
 import hashlib
 from data_intake_corpus import canonical
 from data_intake_openadmet_component import independent_envelope_check
 g,j,r,e=fixture();q,_=admitted_observation(r,g,j,e);groups={'molecule:source','source_native_axis_cell:source'}
 for direction,task in source_task_views(q).items():
  payload=task.canonical_bytes();row=dict(native_payload=payload,native_payload_sha256=hashlib.sha256(payload).hexdigest(),group_ids_json=json.dumps(sorted(groups)),split='train')
  assert independent_envelope_check(row,q,direction,groups)
  raw=json.loads(payload)
  if direction=='entity_to_measurement':raw['targets']['native_measurement']['relation']='='
  else:raw['conditions']['native_entity']=r['entity']
  changed=canonical(raw);row.update(native_payload=changed,native_payload_sha256=hashlib.sha256(changed).hexdigest())
  with pytest.raises(ValueError):independent_envelope_check(row,q,direction,groups)
