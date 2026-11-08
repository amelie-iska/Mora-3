"""Stage an exact ExpansionRx union only after separate, hash-bound review.

Preparation imports are side-effect free. The future finite main compares every
v22 task ID literally, then queries candidate IDs against the complete immutable
RNA task index. The 6.27M reuse witness is never substituted for either domain.
No old source/task/group/index/pointer or model file is modified.
"""
from pathlib import Path
import argparse
import collections
import copy
import datetime
import hashlib
import json
import os
import resource
import sqlite3
import sys
import time

ROOT=Path('/home/iska/Documents/amelie/bio/Mora-3')
FAMILY='ExpansionRx-source-native-ADMET-readouts'
SCHEMA='mora3.CorpusTask/v1'
SCOPE='ExpansionRx-post-RNA-exact-master-union-staging-only'
DIRECTIONS={'entity_to_measurement','measurement_to_entity'}


def canonical_task_digest(value):
    if not isinstance(value,str) or len(value)!=64:
        raise ValueError('candidate task ID must be exact lowercase SHA256 text')
    try:raw=bytes.fromhex(value)
    except ValueError as exc:raise ValueError('candidate task ID is not hexadecimal') from exc
    if len(raw)!=32 or raw.hex()!=value:
        raise ValueError('candidate task ID does not match native RNA index representation')
    return raw


def compare_old_id_batch(ids,candidates):
    hits=set();count=0
    for value in ids:
        if not isinstance(value,str) or not value:
            raise ValueError('every inherited task ID must be present literal text')
        count+=1
        if value in candidates:hits.add(value)
    return count,hits


def rna_membership(db,candidate_ids,chunk_size=400):
    if not 1<=chunk_size<=400:raise ValueError('bounded SQLite parameter batch required')
    info={row[1]:row for row in db.execute('PRAGMA table_info(tasks)')}
    if 'task_id' not in info or info['task_id'][2].upper()!='BLOB' or not info['task_id'][5] or 'reuse_kind' not in info or info['reuse_kind'][2].upper()!='TEXT':
        raise ValueError('actual RNA index schema/canonical binary key is not proven')
    values=list(candidate_ids)
    if len(values)!=len(set(values)):raise ValueError('duplicate candidate ID query')
    encoded=[canonical_task_digest(t) for t in values];hits={}
    for first in range(0,len(encoded),chunk_size):
        chunk=encoded[first:first+chunk_size]
        rows=db.execute('SELECT task_id,reuse_kind FROM tasks WHERE task_id IN ('+','.join('?' for _ in chunk)+')',chunk)
        for digest,kind in rows:
            text=bytes(digest).hex()
            if text not in values[first:first+chunk_size]:raise ValueError('RNA index returned an unrequested literal ID')
            hits[text]=kind
    return hits


def verify_index_binding(pin,closure):
    if not any(row['expected_full_SHA_reference']==pin for row in closure['giant_index_open_close_hash_evidence']):
        raise ValueError('query index is not the exact C2 closing-proof index')


def verify_coverage(expected_v22,scanned_v22,new_RNA_views,expected_v23,candidate_count,queries,old_collisions,rna_collisions):
    if scanned_v22!=expected_v22 or expected_v22+new_RNA_views!=expected_v23:
        raise ValueError('incomplete exact inherited ID-domain coverage; reuse witness is not all v22')
    if queries!=candidate_count or old_collisions or rna_collisions:
        raise ValueError('candidate query coverage incomplete or exact task ID collision')
    return dict(v22_literal_task_IDs_scanned=scanned_v22,new_RNA_domain_task_views_from_closed_C1_C2=new_RNA_views,exact_prior_master_domain_task_views=expected_v23,full_RNA_index_candidate_ID_queries=queries,candidate_task_IDs=candidate_count,old_task_ID_collisions=0,RNA_task_ID_collisions=0,reuse_witness_used_as_full_ID_index=False)


def validate_direction_pairs(rows):
    groups={};ids=set();splits={}
    for r in rows:
        tid=r['task_id'];canonical_task_digest(tid)
        if tid in ids:raise ValueError('duplicate candidate task ID')
        ids.add(tid);obs=r['record_id'];direction=r['direction'];split=r['split']
        if direction not in DIRECTIONS or split not in {'train','test'}:
            raise ValueError('invalid native direction or original allocation')
        if obs in splits and splits[obs]!=split:raise ValueError('same native cell crosses allocations')
        splits[obs]=split
        if direction in groups.setdefault(obs,set()):raise ValueError('duplicate direction of native cell')
        groups[obs].add(direction)
    if any(d!=DIRECTIONS for d in groups.values()):raise ValueError('native cell lacks exactly two distinct directions')
    return groups


def validate_candidate_row(row,group,split):
    canonical_task_digest(row['task_id'])
    if row['source_family']!=FAMILY or row['native_schema']!=SCHEMA or row['split']!=split or split not in {'train','test'} or row['admitted_for_task'] is not True or row['model_adapter_supported'] is not False or row['direction'] not in DIRECTIONS:
        raise ValueError('native source/schema/allocation/admission role changed')
    if hashlib.sha256(row['native_payload']).hexdigest()!=row['native_payload_sha256']:
        raise ValueError('native payload exact checksum changed')
    groups=json.loads(group['effective_groups_json'])
    if group['task_id']!=row['task_id'] or groups!=json.loads(row['group_ids_json']):
        raise ValueError('candidate source groups changed')
    return groups


def candidate_counts(base,component):
    counts=component['source_native_axis_cell_counts'];n=component['serialized_tasks'];directions=component['serialized_directions']
    if n!=61426 or counts['qualified_source_axis_cells']!=30713 or directions!={k:30713 for k in DIRECTIONS}:
        raise ValueError('source-axis-cell/task direction conservation failed')
    if counts['serialized_task_views:train']!=38442 or counts['serialized_task_views:test']!=22984:
        raise ValueError('original source allocations cannot be silently changed')
    if n!=sum(counts['serialized_task_views:'+k] for k in ['train','test']):
        raise ValueError('component split counts do not conserve tasks')
    if component['new_train_test_known_group_intersections']!=0:
        raise ValueError('known component group intersection')
    splits=dict(base['split_counts'])
    for k in ['train','test']:splits[k]+=counts['serialized_task_views:'+k]
    return dict(task_views=base['serialized_tasks']+n,shards=len(base['shards'])+len(component['shards']),task_bytes=base['total_shard_bytes']+component['total_task_shard_bytes'],splits=splits)


def extend_master(base,component,component_ref,source_proof_ref,union_proof_ref,source_policy_ref=None):
    totals=candidate_counts(base,component);m=copy.deepcopy(base)
    oldpaths={s['path'] for s in m['shards']};newpaths={s['path'] for s in component['shards']}
    if len(newpaths)!=len(component['shards']) or oldpaths&newpaths or FAMILY in m['serialized_distinct_record_counts_by_source']:
        raise ValueError('component already present or shard paths overlap')
    m['shards'].extend(copy.deepcopy(component['shards']));m['effective_group_lookup'].update(copy.deepcopy(component['effective_group_lookup']))
    m['serialized_tasks']=totals['task_views'];m['total_shard_bytes']=totals['task_bytes'];m['split_counts']=totals['splits']
    m['native_schema_counts'][SCHEMA]=m['native_schema_counts'].get(SCHEMA,0)+component['serialized_tasks']
    adapter=SCHEMA+';model_adapter_supported=False';m['adapter_support_counts'][adapter]=m['adapter_support_counts'].get(adapter,0)+component['serialized_tasks']
    m['serialized_distinct_record_counts_by_source'][FAMILY]=component['serialized_tasks']
    m['source_ledger'].append(dict(source_family=FAMILY,component=component_ref,independent_original_source_task_proof=source_proof_ref,master_union_proof=union_proof_ref,task_scoped_axis_cells=30713,source_train_axis_cells=19221,source_test_axis_cells=11492,serialized_tasks=61426,admitted_records=30713,admitted_records_unit='qualified source-native endpoint readouts; two directions do not create independent assays',independent_experimental_measurement_count=None,source_native_entities_with_tasks={'train':4152,'test':2228},source_native_entity_count_unit='source compounds, not independently validated material identities',source_observation_and_ML_views_nonadditive=True,full_native_literals_censor_missingness_unknown_fields_preserved=True,original_source_test_known_group_screened_not_global_blind=True,physical_source_schema_units_endpoints=8,source_coordinate_only_endpoints=['LogD','Caco-2 Permeability Efflux'],ASAP_admitted_axis_cells=0,raw_only_unallocated_source_entities_not_admitted=10,material_assay_or_clinical_equivalence_proven=False,source_component_row_global_admission_false_is_immutable_historical_component_scope=True))
    m['contexts'][FAMILY]=dict(rows=61426,source_axis_cells=30713,directions=component['serialized_directions'],source_train_axis_cells=19221,source_test_axis_cells=11492,axis='one endpoint-native scalar or censored bound per task; full original multi-endpoint records remain immutable audit',native_units_and_context=component['inputs']['joint_endpoint_review'],chemical_and_protocol_guards=component['inputs']['chemical_group_review'],source_handoff=component['inputs']['source_handoff'],official_source_test_not_training=True,ASAP_all5_units_transform_gated=True,actual_model_tokens=None,actual_optimizer_use=None)
    m['verification_layers'].append(union_proof_ref['path']);m['source_curation_manifests'].extend([component_ref,source_proof_ref])
    m['mixture_proposal']['ExpansionRx_source_native_ADMET']=dict(status='provisional unactivated; source-specific task union staged only',weight=None,source_axis_cells=30713,task_views=61426,train_task_views=38442,source_test_task_views=22984,independent_experiments=None)
    if source_policy_ref is not None:
        overlays=m.setdefault('active_source_policy_overlays',[])
        if source_policy_ref not in overlays:overlays.append(source_policy_ref)
        exclusion=dict(source_policy=source_policy_ref,selector={'source_family':'RGD1-computed-energy','release':'Figshare21066901.v6'},energy_objective_supervision=False,numerical_energy_conditioning_as_known_truth=False,energy_quality_or_physical_accuracy_claims=False,not_merely_zero_loss_mask=True,audit_or_codec_only=True,runtime_enforcement_proof_pending=True,historical_serialized_rows_retained_not_current_energy_permission=True)
        m.setdefault('source_readiness_overrides',{})['RGD1-v6-energy-objectives']=exclusion
        for entry in m['source_ledger']:
            if entry.get('source_family')=='RGD1-computed-energy':entry['active_source_version_eligibility_override']=copy.deepcopy(exclusion)
        if 'RGD1-computed-energy' in m['contexts']:m['contexts']['RGD1-computed-energy']['active_source_version_eligibility_override']=copy.deepcopy(exclusion)
    m['open_gates'].append('ACTIVE RGD1v6 energy source policy closes oldCSV-derived energy supervision, known-truth conditioning and physical-quality evaluation regardless historical admitted flags; audit/codecs remain allowed. Runtime registration proof is pending. No silent source repair/inversion, and originalHDFenergy semantics are independently unverified.')
    m['open_gates'].append('ExpansionRx is source-bound endpoint readout supervision, not clinical safety/causal design or global blind evaluation. ASAP5units/transform, other peptide/protein/RNA/DNA ADMET, ProCyon/X-Pert/retrosynthesis, global study/material/homology and mixture/token/runtime/readiness gates remain unresolved.')
    m.update(build_id='20261101-ExpansionRx-native-union-v24',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),additional_ExpansionRx_native_component=component_ref,additional_ExpansionRx_original_source_task_proof=source_proof_ref,verification=union_proof_ref['path'],final_full_training_dataset_complete=False,full_dataset_readiness_signature=None,substantive_training_authorized=False,global_leakage_safe=False,actual_whole_model_token_totals=None)
    return m


def validate_source_policy(policy):
    if policy.get('id')!='RGD1-V6-ENERGY-QUARANTINE-01' or policy.get('source_family')!='RGD1':
        raise ValueError('exact inherited RGD1 source-version quarantine required')
    restriction=policy['restriction']
    for key in ['old_v6_energy_objective_supervision','old_v6_energy_quality_evaluation_or_physical_accuracy_claims','old_v6_numerical_energy_conditioning_as_known_truth']:
        if restriction.get(key) is not False:raise ValueError('RGD1 energy role cannot reopen by historical admitted flag')
    if restriction.get('not_merely_zero_loss_mask') is not True or {x['id'] for x in policy['affected_original_v6_assets']}!={39521815,38170374}:
        raise ValueError('source-version gate selector/refusal scope incomplete')


def validate_publication_binding(binding,progress,queue,receipt):
    if not isinstance(binding,dict) or not isinstance(binding.get('master'),dict) or not isinstance(binding.get('publication_closure'),dict):
        raise ValueError('final published v23 binding is required before execution')
    master=binding['master']
    if receipt.get('status')!='normal_complete' or receipt.get('master')!=master or any(receipt.get(k) is not False for k in ['full_dataset_ready','model_use','training']):
        raise ValueError('pointer publication normal closure/gates missing')
    if progress.get('current_master_corpus_manifest')!=master['path'] or progress.get('running_jobs')!=[] or queue.get('current_corpus_manifest')!=master['path']:
        raise ValueError('current master/owner changed after publication')
    if queue.get('full_dataset_ready') is not False or queue.get('substantive_training_before_full_dataset_ready') is not False:
        raise ValueError('full-readiness/training gate changed')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def ref(path,**extra):
    return dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=sha(path),**extra)


def checked(pin):
    p=Path(pin['path']);p=p if p.is_absolute() else ROOT/p
    if p.stat().st_size!=pin['bytes'] or sha(p)!=pin['sha256']:raise ValueError('pinned immutable input changed: '+str(p))
    return p


def load(p):return json.loads(p.read_text())


def write_new(p,value):
    with p.open('x') as f:f.write(json.dumps(value,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--approval',type=Path,required=True);a=parser.parse_args()
    os.chdir(ROOT);start=time.monotonic();plan=load(a.plan);approval=load(a.approval)
    for k,v in dict(approved=True,scope=SCOPE,plan_sha256=sha(a.plan),script_sha256=sha(Path(__file__)),stage_only=True,pointer_publication=False,model_use=False,training=False,full_dataset_ready=False).items():
        if approval.get(k)!=v or (isinstance(v,bool) and approval.get(k) is not v):raise ValueError('separate exact union review missing: '+k)
    if plan.get('status')!='FROZEN_FINAL_EXECUTION_PLAN':raise ValueError('preparation plan is not executable')
    for pin in plan['input_pins']:checked(pin)
    binding=plan.get('publication_binding');validate_publication_binding(binding,load(ROOT/'planning/data_intake_progress.json'),load(ROOT/'configs/dataset_acquisition_queue.json'),load(checked(binding['publication_closure'])) if isinstance(binding,dict) else {})
    sys.path.insert(0,str(ROOT/'scripts'))
    from data_intake_publish_full_rnacentral_guard_v1 import verify_index_stats,require_no_other_RNA_owner
    import pyarrow.parquet as pq
    import pyarrow as pa
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    closure=load(checked(plan['RNA_C2_normal_closure']));verify_index_stats(closure['giant_index_open_close_hash_evidence']);require_no_other_RNA_owner()
    out=ROOT/plan['output_namespace']
    if out.exists():raise FileExistsError('new staging namespace required; no retry or overwrite')
    if sorted(os.sched_getaffinity(0))!=[26,27] or os.getpriority(os.PRIO_PROCESS,0)<10:raise RuntimeError('CPU/nice resource mismatch')
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
    def guard():
        free=os.statvfs(ROOT);freebytes=free.f_bavail*free.f_frsize;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;elapsed=time.monotonic()-start
        available=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
        if elapsed>1800 or freebytes<65*1024**3 or peak>2*1024**3 or available<8*1024**3:
            raise RuntimeError('finite union guard: '+json.dumps(dict(elapsed=elapsed,max_seconds=1800,peak_RSS=peak,max_peak_RSS=2*1024**3,free=freebytes,min_free=65*1024**3,host_MemAvailable=available,min_host_MemAvailable=8*1024**3)))
        return dict(elapsed_seconds=elapsed,peak_RSS_bytes=peak,free_bytes=freebytes,host_MemAvailable_bytes=available)
    guard();base=load(checked(plan['published_master']));old=load(checked(plan['remaining_v22_domain']));component=load(checked(plan['component']));proof=load(checked(plan['source_task_proof']));validate_source_policy(load(checked(plan['active_inherited_source_policy'])));counts=candidate_counts(base,component)
    if counts!=plan['expected_candidate_counts']:raise ValueError('conditional candidate totals changed')
    if proof['component']!=plan['component'] or proof['counts']['verified_task_views']!=61426 or proof['new_train_test_known_group_intersections']!=0:raise ValueError('independent source/task proof binding changed')
    bypath={s['path']:s for s in base['shards']}
    if any(bypath.get(s['path'])!=s for s in old['shards']):raise ValueError('v22 ID-domain metadata not exactly retained in published master')
    oldpaths={s['path'] for s in old['shards']};rna_new=[s for s in base['shards'] if s['path'] not in oldpaths]
    if sum(s['rows'] for s in rna_new)!=54732469 or any(s.get('source_family')!='RNAcentral-reference-priors' or s.get('split')!='train' for s in rna_new):raise ValueError('published v23 has an unaccounted inherited ID domain')
    out.mkdir();state=dict(stage='candidate_component',candidate_rows=0,old_ID_rows=0,RNA_candidate_queries=0)
    try:
        (out/'code').mkdir()
        for pin in plan['code_pins']:
            (out/'code'/Path(pin['path']).name).open('xb').write(checked(pin).read_bytes())
        records=[];candidate_ids=set();splitcounts=collections.Counter();directioncounts=collections.Counter();seen_groups={};newbytes=0
        for sh in component['shards']:
            guard();p=checked(sh);gp=checked(component['effective_group_lookup'][sh['path']]);pf=pq.ParquetFile(p);gt=pq.read_table(gp).to_pylist()
            if pf.metadata.num_rows!=sh['rows'] or len(gt)!=sh['rows']:raise ValueError('candidate shard/group row conservation failed')
            rows=pf.read().to_pylist()
            for r,g in zip(rows,gt):
                if r['task_id'] in candidate_ids:raise ValueError('duplicate component task ID')
                candidate_ids.add(r['task_id']);canonical_task_digest(r['task_id'])
                groups=validate_candidate_row(r,g,sh['split'])
                splitcounts[r['split']]+=1;directioncounts[r['direction']]+=1
                records.append({k:r[k] for k in ['task_id','record_id','direction','split']})
                for group in groups:seen_groups.setdefault(group,set()).add(r['split'])
            if ref(p)!= {k:sh[k] for k in ['path','bytes','sha256']} or ref(gp)!={k:component['effective_group_lookup'][sh['path']][k] for k in ['path','bytes','sha256']}:raise ValueError('candidate files changed during read')
            newbytes+=p.stat().st_size;state['candidate_rows']+=len(rows)
        pairs=validate_direction_pairs(records)
        if len(pairs)!=30713 or splitcounts!={'train':38442,'test':22984} or directioncounts!={k:30713 for k in DIRECTIONS} or newbytes!=17161141:raise ValueError('component source-cell/direction/allocation conservation failed')
        if any(len(v)>1 for v in seen_groups.values()):raise ValueError('candidate train/test known-group intersection')
        state['stage']='all_remaining_v22_literal_IDs';oldhits=set();oldcount=0
        with (out/'v22-complete-ID-domain-shard-ledger.jsonl').open('x') as ledger:
            for number,sh in enumerate(old['shards']):
                guard();p=checked(sh);pf=pq.ParquetFile(p)
                if pf.metadata.num_rows!=sh['rows'] or 'task_id' not in pf.schema_arrow.names:raise ValueError('inherited task-ID column or footer changed')
                ids_hash=hashlib.sha256();n=0
                for batch in pf.iter_batches(columns=['task_id'],batch_size=8192):
                    ids=batch.column(0).to_pylist();c,h=compare_old_id_batch(ids,candidate_ids);n+=c;oldcount+=c;oldhits.update(h)
                    for value in ids:ids_hash.update(json.dumps(value,ensure_ascii=False).encode()+b'\n')
                    guard()
                if n!=sh['rows'] or ref(p)!={k:sh[k] for k in ['path','bytes','sha256']}:raise ValueError('inherited IDs incomplete or source changed')
                ledger.write(json.dumps(dict(shard=sh,all_literal_IDs_read=n,ordered_literal_IDs_sha256=ids_hash.hexdigest(),candidate_ID_collisions=sorted(oldhits)))+'\n');state['old_ID_rows']=oldcount
                if oldhits:raise ValueError('exact new/old task-ID collision; retained partial proof')
                if number%1000==0:print('EXPANSIONRX_V24_OLD_ID_PROGRESS',number+1,oldcount,guard(),flush=True)
        state['stage']='full_RNA_targeted_candidate_ID_queries';verify_index_stats(closure['giant_index_open_close_hash_evidence'])
        verify_index_binding(plan['RNA_task_index'],closure)
        index=ROOT/plan['RNA_task_index']['path'];db=sqlite3.connect('file:'+str(index)+'?mode=ro&immutable=1',uri=True);db.execute('PRAGMA query_only=ON');db.execute('PRAGMA cache_size=-65536')
        all_rna_hits={};ordered=sorted(candidate_ids)
        for first in range(0,len(ordered),400):
            guard();chunk=ordered[first:first+400];all_rna_hits.update(rna_membership(db,chunk));state['RNA_candidate_queries']+=len(chunk)
        db.close();verify_index_stats(closure['giant_index_open_close_hash_evidence'])
        coverage=verify_coverage(old['serialized_tasks'],oldcount,54732469,base['serialized_tasks'],len(candidate_ids),state['RNA_candidate_queries'],len(oldhits),len(all_rna_hits))
        for pin in plan['input_pins']:checked(pin)
        validate_publication_binding(binding,load(ROOT/'planning/data_intake_progress.json'),load(ROOT/'configs/dataset_acquisition_queue.json'),load(checked(binding['publication_closure'])))
        verify_index_stats(closure['giant_index_open_close_hash_evidence']);state['stage']='staged_master_write';g=guard()
        proofout=out/'receipt.json';write_new(proofout,dict(schema='mora3.ExpansionRxExactInheritedUnionProof/v1',status='verified_staging_only_not_pointer_published',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),prior_master=plan['published_master'],component=plan['component'],independent_source_task_proof=plan['source_task_proof'],publication_binding=binding,coverage=coverage,source_axis_cells=30713,task_views=61426,original_source_train_views=38442,official_source_test_views=22984,known_source_group_exposure_review_inherited=component['inputs']['chemical_group_review'],known_group_screened_source_test_is_not_global_blind_evaluation=True,native_payload_censor_source_unknown_fields_and_groups_unchanged=True,active_inherited_source_policy=plan['active_inherited_source_policy'],RGD1_v6_energy_roles_hard_closed_regardless_historical_admission=True,RGD1_runtime_enforcement_proof_pending=True,affected_RGD1_metadata_inventory=plan['inherited_RGD1_affected_inventory'],remaining_v22_ID_ledger=ref(out/'v22-complete-ID-domain-shard-ledger.jsonl'),giant_RNA_index_complete_hash_authority=plan['RNA_C2_normal_closure'],giant_RNA_index_targeted_readonly_membership_and_current_stat_checks=True,giant_RNA_index_rehashed_here=False,all_old_task_or_group_files_unchanged=True,source_cell_views_not_independent_assays=True,model_use=False,actual_optimizer_use=None,actual_model_tokens=None,mixture_activated=False,full_dataset_ready=False,training=False,guard=g))
        staged=extend_master(base,component,plan['component'],plan['source_task_proof'],ref(proofout),plan['active_inherited_source_policy']);staged['inherited_RGD1_affected_scope_inventory']=plan['inherited_RGD1_affected_inventory'];staged.update(prior_manifest_sha256=plan['published_master']['sha256'],supersedes_for_consumption=plan['published_master']['path'],source_components_base_manifest=plan['published_master']['path'])
        target=out/'master-corpus-manifest-v24-ExpansionRx-native-v1.json';write_new(target,staged)
        if load(target)!=staged:raise ValueError('staged manifest JSON roundtrip failed')
        guard();write_new(out/'staging-receipt.json',dict(master=ref(target),proof=ref(proofout),counts=counts,current_pointer_remains=plan['published_master'],pointer_publication=False,full_dataset_ready=False,model_use=False,training=False));print('COMPLETE_EXPANSIONRX_V24_STAGED_UNION',json.dumps(dict(master=ref(target),counts=counts)),flush=True)
    except BaseException as exc:
        try:write_new(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),state=state,guard_sample=dict(seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),partial_outputs_preserved=True,no_retry_reset_rollback_or_cleanup=True))
        except BaseException:pass
        raise


if __name__=='__main__':main()
