"""Exact committed-prefix accounting for a NEW audit continuation only."""
import json
from data_intake_master_GO_scan_v2 import subject_scope

def committed_prefix(shards,records):
    bypath={r[0]:tuple(r[1:]) for r in records};n=len(records)
    if len(bypath)!=n or n>len(shards):raise ValueError('duplicate/oversized committed shard registry')
    if set(bypath)!={s['path'] for s in shards[:n]}:raise ValueError('registry is not exact contiguous master prefix')
    for s in shards[:n]:
        if bypath[s['path']]!=(s['sha256'],s['bytes'],s['rows'],1):raise ValueError('committed shard source/hash/count/verification mismatch')
    return n,sum(s['rows'] for s in shards[:n])

def accumulate(rollups,counts,subjects,schemas,row,exposure,*,count_native):
    family=row['source_family'];schema=row['native_schema'];split=row['split'];roll=rollups[json.dumps([family,schema,split])]
    if count_native:
        roll['task_views']+=1;schemas[schema]+=1;counts['all_admitted_master_task_views']+=1
    if exposure is None:return
    for role,IDs in exposure['GO_IDs_by_native_role'].items():
        if IDs:roll['views_with_any_GO_ID_'+role]+=1
    for kind,prefix in [('audit_role_counts','audit_'),('training_proposal_counts','training_proposal_')]:
        for field,value in exposure[kind].items():roll[prefix+field]+=value;counts[prefix+field]+=value
    if family=='UniProt-GOA-assertion-packets':
        counts['all_admitted_GOA_packet_views']+=1;subject=subjects[subject_scope(row['record_id'])];subject['packet_views']+=1
        for kind,prefix in [('audit_role_counts','audit_'),('training_proposal_counts','training_proposal_')]:
            for field,value in exposure[kind].items():subject[prefix+field]+=value

def overlay_record(row,shard,local,exposure,role_ref):
    policy=exposure['training_mixture_policy_proposal']
    if row['split']!='train' or not policy['coarse_entire_view_exclusion_proposed']:raise ValueError('no validation/test or unaffected overlay proposal')
    return dict(schema='mora3.NONAPPLYINGTrainingOnlyHeldoutGOOverlayProposal/v2',task_id=row['task_id'],native_source_shard=shard,shard_row=local,native_payload_SHA256=row['native_payload_sha256'],source_family=row['source_family'],source_split=row['split'],literal_record_id=row['record_id'],native_official_role_sets=role_ref,training_mixture_only=True,official_validation_test_unchanged=True,coarse_entire_view_exclusion_proposed=True,selective_atomic_GOA_source_annotation_IDs=policy.get('atomic_source_annotation_IDs_to_project_out',[]),protected_condition_requires_view_exclusion_or_separate_admission=exposure['protected_condition_exposure'],non_GOA_fine_projection_unreviewed=row['source_family']!='UniProt-GOA-assertion-packets',proposal_only=True,policy_activation=False,task_admission=False,model_use=False)

def match_protected(stored,row,shard,local,exposure):
    want=dict(task_id=row['task_id'],source_family_json=json.dumps(row['source_family']),native_schema=row['native_schema'],source_split=row['split'],literal_record_id=row['record_id'],subject_scope=subject_scope(row['record_id']),source_shard=shard['path'],source_shard_SHA=shard['sha256'],source_shard_row=local,native_payload_SHA=row['native_payload_sha256'],audit_protected_target=int(exposure['protected_target_exposure']),audit_protected_condition=int(exposure['protected_condition_exposure']),training_exclusion_proposed=int(exposure['training_mixture_policy_proposal']['coarse_entire_view_exclusion_proposed']),training_selective_GOA_projection_proposed=int(exposure['training_mixture_policy_proposal']['selective_GOA_projection_proposed']))
    if any(stored[k]!=v for k,v in want.items()) or json.loads(stored['proposal_json'])!=exposure:raise ValueError('inherited protected metadata/proposal differs from bounded native replay')


def match_overlay_line(line,expected):
    if not line.endswith(b'\n'):raise ValueError('missing/truncated inherited overlay line')
    if json.loads(line)!=expected:raise ValueError('inherited overlay proposal differs from native replay')

def finish_overlay(stream):
    if stream.read(1):raise ValueError('unproven extra or uncommitted inherited overlay tail')


def copy_verified_database(source,target,expected_sha,guard):
    import hashlib
    from pathlib import Path
    if any(Path(str(source)+suffix).exists() for suffix in ['-journal','-wal','-shm']):raise ValueError('unreviewed original journal/WAL; do not recover it')
    h=hashlib.sha256()
    with source.open('rb') as old,target.open('xb') as new:
        while chunk:=old.read(4*2**20):new.write(chunk);h.update(chunk);guard()
    if h.hexdigest()!=expected_sha:raise ValueError('copied immutable database checksum mismatch; retain partial')
