"""FreeSolv heldout versus AZ/FreeSolv train: sealed group SIDECARS ONLY.

No heldout source shard is opened: in particular native_payload, targets,
condition_context_json and full_axis_values are not accessible to this reader.
These hashes are connectivity/scaffold *leakage guards*, not stereochemical
identity or validation admission. The prior authenticated source receipts
provide their expected sidecar SHA256 and original source-file identity.
"""
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re

from mora3.data.finite_union_native_rows import read_exact
from mora3.research.cross_source_az_freesolv import (
    _readback, _AZ_RECEIPT, _AZ_SHA, _FS_RECEIPT, _FS_SHA,
    _ORIGINAL, _CANDIDATE, _COMMON, _POLICY,
)

_COLUMNS=['task_id','source_file','source_group_ids_json','effective_groups_json']
_KINDS=('molecule','scaffold')


def _check_record(r,source,role,seen):
    if (type(r) is not dict or set(r)!={'source','role','task_id','molecule_group','scaffold_group'}
            or r['source']!=source or r['role']!=role
            or type(r['task_id']) is not str or re.fullmatch(r'[0-9a-f]{64}',r['task_id']) is None
            or r['task_id'] in seen):
        raise ValueError('Source/role/unique task ID policy violated')
    seen.add(r['task_id'])
    for kind in _KINDS:
        g=r[kind+'_group']
        if type(g) is not str or re.fullmatch(kind+r':[0-9a-f]{64}',g) is None:
            raise ValueError('Missing or invalid source chemical group')


def group_overlap(train,heldout):
    """Count only exact source-native hash-string intersections, not identity proofs."""
    if (type(train) is not dict or set(train)!={'AZ','FreeSolv'}
            or type(heldout) is not dict or set(heldout)!={'validation','test'}):
        raise ValueError('Fixed source and heldout roles required')
    seen=set()
    for source in ('AZ','FreeSolv'):
        if type(train[source]) is not tuple or not train[source]:
            raise ValueError('Missing train source records')
        for r in train[source]:_check_record(r,source,'train',seen)
    for role in ('validation','test'):
        if type(heldout[role]) is not tuple or not heldout[role]:
            raise ValueError('Missing heldout role records')
        for r in heldout[role]:_check_record(r,'FreeSolv',role,seen)

    result={}
    pairs=[('validation_vs_AZ_train',heldout['validation'],train['AZ']),
           ('validation_vs_FreeSolv_train',heldout['validation'],train['FreeSolv']),
           ('test_vs_AZ_train',heldout['test'],train['AZ']),
           ('test_vs_FreeSolv_train',heldout['test'],train['FreeSolv']),
           ('validation_vs_test',heldout['validation'],heldout['test'])]
    for label,left,right in pairs:
        observation={'left_task_count':len(left),'right_task_count':len(right)}
        for kind in _KINDS:
            key=kind+'_group'
            lmap=defaultdict(list);rmap=defaultdict(list)
            for r in left:lmap[r[key]].append(r['task_id'])
            for r in right:rmap[r[key]].append(r['task_id'])
            shared=sorted(set(lmap)&set(rmap))
            observation[kind+'_hash_collision_count']=len(shared)
            observation[kind+'_left_tasks_colliding']=len({x for g in shared for x in lmap[g]})
            observation[kind+'_right_tasks_colliding']=len({x for g in shared for x in rmap[g]})
            observation[kind+'_hash_collisions']=[{
                'group':g,'left_task_ids':sorted(lmap[g]),'right_task_ids':sorted(rmap[g])}
                for g in shared]
        result[label]=observation
    return dict(heldout_comparisons=result,
        all_scoped_group_hashes_disjoint=all(
            all(v[kind+'_hash_collision_count']==0 for kind in _KINDS) for v in result.values()),
        group_protocol='original non-stereo molecule hash + original Murcko scaffold hash; no stereochemical/tautomer/salt equivalence asserted',
        global_split_admitted=False)


def _sidecar_records(root,descriptors,source,role):
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows=[];proofs=[]
    for entry in descriptors:
        shard=entry['source_shard']
        side=entry['original_group_sidecar'] if source=='AZ' else entry['group_sidecar']
        path=shard['path']
        if (type(path) is not str or not path.startswith('outputs/data-intake/')
                or source=='AZ' and '/shards/train/' not in path
                or source=='FreeSolv' and ('/FreeSolv/'+role+'/') not in path
                or source=='FreeSolv' and entry['role']!=role
                or type(side['path']) is not str
                or '/effective-groups/' not in side['path']
                or not side['path'].startswith('outputs/data-intake/')
                or side['path'].endswith('.parquet') is False):
            raise ValueError('Original source role or sidecar path is not allowed')
        # No reading even a single byte from the original heldout data shard.
        src=(root/path).resolve();safebase=(root/'outputs/data-intake').resolve()
        if safebase not in src.parents or not src.is_file() or src.stat().st_size!=shard['bytes']:
            raise ValueError('Original source shard pointer/stat mismatch')
        raw=read_exact(side,root=root)  # SHA256 authenticated SIDE-CAR bytes only.
        columns=['task_id','effective_groups_json'] if source=='AZ' else _COLUMNS
        table=pq.read_table(pa.BufferReader(raw),columns=columns,use_threads=False)
        if table.num_rows!=side['rows'] or table.num_rows!=shard['rows']:
            raise ValueError('Group sidecar and original shard row-count mismatch')
        for r in table.to_pylist():
            if source=='FreeSolv' and r['source_file']!=path:
                raise ValueError('Original source pointer differs from effective sidecar')
            effective=json.loads(r['effective_groups_json'])
            declared=json.loads(r['source_group_ids_json']) if source=='FreeSolv' else effective
            if (type(effective) is not list or type(declared) is not list
                    or len(effective)!=len(set(effective))
                    or set(declared)!=set(effective)):
                raise ValueError('Source groups not preserved in sealed effective sidecar')
            typed={}
            for kind in _KINDS:
                matches=[x for x in effective if type(x) is str and x.startswith(kind+':')]
                if len(matches)!=1:
                    raise ValueError('Exactly one original molecule/scaffold hash required')
                typed[kind+'_group']=matches[0]
            rows.append(dict(source=source,role=role,task_id=r['task_id'],**typed))
        proofs.append(dict(source=source,role=role,sidecar_path=side['path'],
            sidecar_sha256=side['sha256'],rows=table.num_rows,
            prior_receipt_source_shard_path=path,
            prior_receipt_source_shard_sha256=shard['sha256'],
            source_shard_bytes_read=0))
    return tuple(rows),proofs


def audit_original_freesolv_heldout_group_exposure(root):
    """Pin source-native group leakage guards; never load heldout target columns."""
    root=Path(root).resolve()
    if root!=_ORIGINAL or Path.cwd().resolve()!=root:
        raise ValueError('Exact original checkout cwd required for immutable source-sidecar access')
    az=_readback(_AZ_RECEIPT,_AZ_SHA)
    fs=_readback(_FS_RECEIPT,_FS_SHA)
    if (az['original_train_rows']!=4184 or az['original_train_shards']!=17
            or az['no_heldout_role_shards_for_this_route'] is not True
            or fs['native_train_rows']!=556
            or fs['original_role_rows']!={'train':556,'validation':42,'test':44}
            or fs['heldout_target_payload_decoded'] is not False
            or az['heldout_target_payload_decoded'] is not False):
        raise ValueError('Authenticated source role or heldout-access contract mismatch')
    fs_by_role={role:[x for x in fs['source_files'] if x['role']==role]
                for role in ('train','validation','test')}
    if (len(az['original_source_files']), *(len(fs_by_role[r]) for r in ('train','validation','test')))!=(17,3,1,1):
        raise ValueError('Original source sidecar inventory changed')
    az_rows,az_proofs=_sidecar_records(root,az['original_source_files'],'AZ','train')
    fs_train,fs_train_proofs=_sidecar_records(root,fs_by_role['train'],'FreeSolv','train')
    heldout={};heldout_proofs=[]
    for role in ('validation','test'):
        heldout[role],proofs=_sidecar_records(root,fs_by_role[role],'FreeSolv',role)
        heldout_proofs+=proofs
    if (len(az_rows),len(fs_train),len(heldout['validation']),len(heldout['test']))!=(4184,556,42,44):
        raise ValueError('Original role counts cannot be inferred from partial metadata')
    overlap=group_overlap({'AZ':az_rows,'FreeSolv':fs_train},heldout)
    proofs=az_proofs+fs_train_proofs+heldout_proofs
    provenance_digest=sha256(json.dumps(
        [(r['source'],r['role'],r['task_id'],r['molecule_group'],r['scaffold_group'])
         for r in az_rows+fs_train+heldout['validation']+heldout['test']],
        separators=(',',':')).encode()).hexdigest()
    return dict(schema='Mora3.OriginalFreeSolvHeldoutTargetInaccessibleGroupHashExposure/v1',
        train_rows={'AZ':len(az_rows),'FreeSolv':len(fs_train)},
        heldout_rows={role:len(heldout[role]) for role in ('validation','test')},
        immutable_complete_train_source_readbacks={_AZ_RECEIPT:_AZ_SHA,_FS_RECEIPT:_FS_SHA},
        candidate_source_snapshot_sha256=_CANDIDATE,
        common_source_snapshot_sha256=_COMMON,
        native_view_policy_sha256=_POLICY,
        original_source_sidecar_proofs=proofs,
        group_sidecars_authenticated=len(proofs),
        ordered_task_role_group_commitment_sha256=provenance_digest,
        read_columns_heldout=list(_COLUMNS),read_columns_AZ_train=['task_id','effective_groups_json'],
        heldout_native_shards_opened=False,
        heldout_target_payload_decoded=False,
        heldout_source_shard_bytes_hashed_this_run=False,
        heldout_structural_isomeric_identity_proven=False,
        provenance_limit='Only prior authenticated original source pointers and current exact-byte group sidecars; cannot infer exact stereochemistry or global validation eligibility',
        **overlap,
        training_admission=False,optimizer_updates=0,actual_proxy_runs=0,actual_target_runs=0)
