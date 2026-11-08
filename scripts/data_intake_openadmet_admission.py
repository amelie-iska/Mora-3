"""Task-scoped ExpansionRx admission; source cells never become extra assays."""
import json
from copy import deepcopy
from data_intake_admet_endpoint_review import apply_endpoint_review
from data_intake_admet_observations import task_view


def source_allocation(graph, joint):
    for key in ('source_family','native_entity_id','native_row_index','native_parent_protocol'):
        if graph[key]!=joint[key]:
            raise ValueError('exact source-native parent/group review concordance required')
    if graph['source_family']!='ExpansionRx':
        return None,'source_endpoint_definition_or_transform_not_admitted'
    scope=json.loads(graph['source_graph_scope_json'])
    if not scope['native_predictor_identity_candidate'] or not scope['legacy_plain_SMILES_exact_comparison_eligible']:
        return None,'native_graph_representation_scope'
    protected=bool(json.loads(graph['known_current_heldout_group_matches_json']))
    frozen=bool(json.loads(graph['frozen_pilot_reservation_matches_json']))
    if protected or frozen:
        return None,'existing_protected_group_or_frozen_reservation'
    parent=graph['native_parent_protocol']
    if parent=='source_train':
        if json.loads(joint['any_benchmark_heldout_parent_source_families_json']):
            return None,'known_official_source_heldout_parent_graph_group'
        if joint['candidate_train_after_joint_known_source_guards'] is not True:
            return None,'joint_source_train_guard_not_qualified'
        return 'train','guarded_original_source_train'
    if parent=='source_test':
        if json.loads(graph['oldtrain_group_matches_json']):
            return None,'source_test_known_prior_train_graph_group_exposure'
        return 'test','known_group_screened_original_source_test_subset_not_global_blind_evaluation'
    return None,'raw_only_original_source_parent_unallocated'


def admitted_observation(record, graph, joint, endpoint_review):
    """Return a task-specific source-readout candidate, or an explicit exclusion.

    This is source/corpus evidence only. Publication in a master and native model
    reader verification are separate; no full-ready or optimizer use is implied.
    """
    if record['source_ref']['native_entity_id']!=graph['native_entity_id'] or record['source_ref']['native_row_index']!=graph['native_row_index']:
        raise ValueError('observation and graph must be the exact same source row')
    if record['entity']['representation']!=json.loads(graph['source_graph_scope_json'])['native_representation']:
        raise ValueError('native source representation changed before admission')
    expected={'source_train':'native_train','source_test':'native_test','source_unallocated':'raw_only_unallocated'}[graph['native_parent_protocol']]
    actual=record['source_split']['parent_protocol']
    if actual!=expected:
        raise ValueError('original source parent split cannot be reallocated silently')
    allocation,reason=source_allocation(graph,joint)
    if allocation is None:
        return None,reason
    if record['evidence_role']!='source_curated_measurement' or record['admission'].get('source_rights') is not True:
        return None,'source_curated_measurement_or_rights_gate'
    if not record['cell']['observation_present'] or not record['cell']['numeric_loss_eligible']:
        return None,'missing_or_unparsed_source_native_axis_cell'
    out=apply_endpoint_review(record,endpoint_review)
    if not out['admission']['endpoint_definition'] or not out['admission']['task_units_or_declared_native_coordinate']:
        return None,'source_endpoint_or_native_coordinate_gate'
    out['admission'].update(exact_identity=True,split_group_review=True,
        exact_identity_scope='exact released native chemical reference graph; full material/batch/protonation context not established',
        allocation=allocation,allocation_reason=reason,
        global_novelty_or_study_disjointness_proven=False,
        original_source_protocol_parent_preserved=True,
        full_dataset_readiness=False,model_use_authorized=False)
    out['graph_scope_review_audit']=json.loads(graph['source_graph_scope_json'])
    out['source_protocol_group_review_audit']=deepcopy(graph)
    out['joint_source_benchmark_guard_audit']=deepcopy(joint)
    out['inverse_design_scope_audit']='source-supported observational property-to-reference-graph task; not clinical efficacy/safety or causal optimized-design validation'
    return out,reason


def source_task_views(record):
    """Both views reuse one native source axis cell and preserve its censor."""
    return {direction:task_view(record,direction=direction) for direction in
            ('entity_to_measurement','measurement_to_entity')}
