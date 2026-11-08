"""Pure completion-aware owned RNA process accounting, no process control."""
import json

class PhaseCoexistenceError(ValueError):
    pass

def owned_phase_RSS(snapshot,expected,normal_completion_verified):
    if snapshot is None:
        if normal_completion_verified is True:return 0,'verified_normal_completion'
        raise PhaseCoexistenceError('owned_RNA_phase_disappeared_without_verified_normal_completion')
    for key in ('pid','start_ticks'):
        if snapshot.get(key)!=expected[key]:raise PhaseCoexistenceError('owned_RNA_phase_identity_mismatch:'+key)
    if snapshot.get('state')=='Z':
        if normal_completion_verified is True:return 0,'verified_normal_completion_zombie'
        raise PhaseCoexistenceError('owned_RNA_phase_disappeared_without_verified_normal_completion')
    for key in ('argv',):
        if snapshot.get(key)!=expected[key]:raise PhaseCoexistenceError('owned_RNA_phase_identity_mismatch:'+key)
    if snapshot.get('script_sha256')!=expected['script_sha256']:
        raise PhaseCoexistenceError('owned_RNA_phase_script_bytes_changed')
    if snapshot.get('CPU_affinity')!=[26,27] or snapshot.get('nice',0)<10:
        raise PhaseCoexistenceError('owned_RNA_phase_resource_identity_changed')
    return snapshot['RSS_bytes'],'verified_live_owner'

def normal_materialization(manifest,log_tail,expected):
    if manifest.get('schema')!='mora3.FullRNAcentralNativeMaterializedComponent/v1' or manifest.get('status')!='all_eligible_new_native_tasks_serialized_pending_independent_full_source_and_master_union':return False
    for key in ('census','build_plan','clock','phase_execution_approval'):
        if manifest.get(key)!=expected[key]:return False
    if manifest.get('counts',{}).get('new_task_views_serialized')!=54732469:return False
    marker='COMPLETE_FULL_RNACENTRAL_MATERIALIZATION '
    try:final=json.loads(log_tail.rsplit(marker,1)[1].strip())
    except (IndexError,ValueError,TypeError):return False
    return final.get('counts')==manifest['counts'] and final.get('task_shard_bytes')==manifest['task_shard_bytes'] and final.get('effective_group_bytes')==manifest['effective_group_bytes']

def normal_source_proof(receipt,log_tail,expected):
    if receipt.get('schema')!='mora3.FullRNAcentralIndependentSourceAndTaskProof/v1' or receipt.get('status')!='verified_source_and_new_tasks_pending_inherited_master_group_union':return False
    for key in ('build_plan','census','component','clock','phase_execution_approval'):
        if receipt.get(key)!=expected[key]:return False
    if receipt.get('all_source_occurrences')!=61026468 or receipt.get('all_new_task_payload_source_target_ID_location_axis_group_readback')!=54732469:return False
    marker='COMPLETE_FULL_RNA_SOURCE_TASK_PROOF '
    try:final=json.loads(log_tail.rsplit(marker,1)[1].strip())
    except (IndexError,ValueError,TypeError):return False
    return final==receipt and receipt.get('old_indices_and_completed_census_and_materializer_index_unchanged') is True
