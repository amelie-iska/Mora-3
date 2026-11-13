"""Original AZ/FreeSolv *training* chemical-exposure overlap; no model/trial.

The input evidence is the separately authenticated, full original native train
readbacks. This module independently reopens every SHA-bound original train
shard and group sidecar, without opening held-out role shards or using targets.
RDKit identity is a bounded two-source structural link, NOT a global split.
"""
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re

from mora3.data.finite_union_native_rows import full9, read_exact

_ORIGINAL=Path('/home/iska/Documents/amelie/bio/Mora-3')
_WORKTREE=Path(__file__).resolve().parents[3]
_AZ_RECEIPT='outputs/common-source-logd-complete-20261111/actual-original-az-logd-complete.json'
_FS_RECEIPT='outputs/common-source-freesolv-complete-20261111/original-complete-freesolv-readback.json'
_AZ_SHA='1a6e1dfdb26ccfe2693e974b79870ea3d1763771206dd1a366eba8a02f8151d7'
_FS_SHA='f0820098583bd37155b55f36716b8734649613faa1d897250c2d89759d7e51a9'
_CANDIDATE='1091e3f27d57cb70203ab9fce1136836d0cd96eef955b6682a3e6264abadc733'
_COMMON='79d328f542d1158581875ef325e56f213b407992f95db6ee0c7352a3459669d4'
_POLICY='c3812ecfcfea8a01f81769bf5ef6fb3880e04a0ac054a6349e9272db14765393'


def _unique(pairs):
    out={}
    for key,value in pairs:
        if key in out:
            raise ValueError('Duplicate source-native JSON keys not accepted')
        out[key]=value
    return out


def _readback(name,digest):
    raw=(_WORKTREE/name).read_bytes()
    if sha256(raw).hexdigest()!=digest:
        raise ValueError('Immutable original-source readback digest mismatch')
    d=json.loads(raw,object_pairs_hook=_unique)
    if (d['candidate_source_snapshot_sha256']!=_CANDIDATE
            or d['common_source_snapshot_sha256']!=_COMMON
            or d['native_view_policy_sha256']!=_POLICY
            or d['training_admission'] is not False
            or d['heldout_target_payload_decoded'] is not False
            or d['optimizer_updates']!=0):
        raise ValueError('Original full-source receipt source/role guard failed')
    return d


def chemical_exposure_pairs(az,fs):
    """Link exact stereochemical structures and separate stereo-unspecified leads.

    No salts, charge states, tautomer or assay semantics are normalized. A
    connectivity-only match is *not* an exact chemical identity assertion.
    """
    from rdkit import Chem
    if (type(az) is not tuple or type(fs) is not tuple or not az or not fs):
        raise ValueError('Both immutable nonempty source collections required')
    sources={'AZ':az,'FreeSolv':fs}
    byiso={};byconn={}; total={}; names={};
    for label,records in sources.items():
        byiso[label]=defaultdict(list);byconn[label]=defaultdict(list)
        seen_ids=set();seen_tasks=set()
        for r in records:
            if (type(r) is not dict or set(r)!={'source','record_id','task_id','smiles','groups'}
                    or r['source']!=label or type(r['record_id']) is not str or not r['record_id']
                    or type(r['task_id']) is not str or not re.fullmatch('[0-9a-f]{64}',r['task_id'])
                    or type(r['smiles']) is not str or not r['smiles']
                    or type(r['groups']) is not tuple or len(r['groups'])!=len(set(r['groups']))
                    or any(type(g) is not str or ':' not in g for g in r['groups'])
                    or r['record_id'] in seen_ids or r['task_id'] in seen_tasks):
                raise ValueError('Original source row/role/identity/group must not be synthesized or deduplicated')
            seen_ids.add(r['record_id']);seen_tasks.add(r['task_id'])
            m=Chem.MolFromSmiles(r['smiles'])
            if m is None or m.GetNumAtoms()==0:
                raise ValueError('Cannot silently accept unparsed original chemical structure')
            iso=Chem.MolToSmiles(m,canonical=True,isomericSmiles=True)
            conn=Chem.MolToSmiles(m,canonical=True,isomericSmiles=False)
            byiso[label][iso].append(r);byconn[label][conn].append((r,iso))
        total[label]=len(records);names[label]=len(byiso[label])
    exact=[];connect_only=[]
    for identity in sorted(set(byiso['AZ'])&set(byiso['FreeSolv'])):
        for a in byiso['AZ'][identity]:
            for f in byiso['FreeSolv'][identity]:
                exact.append(dict(az_record_id=a['record_id'],freesolv_record_id=f['record_id'],
                                  canonical_isomeric_smiles=identity))
    for identity in sorted(set(byconn['AZ'])&set(byconn['FreeSolv'])):
        for a,ai in byconn['AZ'][identity]:
            for f,fi in byconn['FreeSolv'][identity]:
                if ai!=fi:
                    connect_only.append(dict(az_record_id=a['record_id'],
                        freesolv_record_id=f['record_id'],canonical_connectivity_smiles=identity,
                        az_isomeric_smiles=ai,freesolv_isomeric_smiles=fi))
    exact=sorted(exact,key=lambda r:(r['az_record_id'],r['freesolv_record_id']))
    connect_only=sorted(connect_only,key=lambda r:(r['az_record_id'],r['freesolv_record_id']))
    all_exact_ids={r['freesolv_record_id'] for r in exact}
    all_connectivity_ids=all_exact_ids|{r['freesolv_record_id'] for r in connect_only}
    # Source-local original group names are deliberately namespaced. Only
    # *exact* cross-source isomeric chemical matches add cross-source edges;
    # the connectivity-only/stereochemical candidates are never unioned.
    combined=list(az)+list(fs)
    parents=list(range(len(combined)))
    def find(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]]
            i=parents[i]
        return i
    def union(i,j):
        parents[find(i)]=find(j)
    owner={}
    for i,r in enumerate(combined):
        for group in r['groups']:
            key=(r['source'],group)
            if key in owner:union(i,owner[key])
            else:owner[key]=i
    local_components=len({find(i) for i in range(len(combined))})
    coordinate={(r['source'],r['record_id']):i for i,r in enumerate(combined)}
    for item in exact:
        union(coordinate[('AZ',item['az_record_id'])],
              coordinate[('FreeSolv',item['freesolv_record_id'])])
    components=defaultdict(list)
    for i,r in enumerate(combined):components[find(i)].append(r)
    cross_components=[v for v in components.values() if {r['source'] for r in v}=={'AZ','FreeSolv'}]
    scoped_closure=dict(original_source_local_component_count=local_components,
        two_source_exact_component_count=len(components),
        cross_source_component_count=len(cross_components),
        freesolv_train_rows_in_cross_source_components=sum(
            sum(r['source']=='FreeSolv' for r in c) for c in cross_components),
        cross_source_component_record_counts=sorted(len(c) for c in cross_components),
        grouping_protocol='source-namespaced original groups + exact isomeric cross-source edges only',
        global_split_admitted=False)
    return dict(train_row_count=total,unique_isomeric_structures=names,
        exact_isomeric_chemical_count=len(set(byiso['AZ'])&set(byiso['FreeSolv'])),
        connectivity_chemical_count=len(set(byconn['AZ'])&set(byconn['FreeSolv'])),
        exact_cross_source_record_pair_count=len(exact),
        connectivity_only_record_pair_count=len(connect_only),
        direct_exact_freesolv_record_count=len(all_exact_ids),
        direct_connectivity_freesolv_record_count=len(all_connectivity_ids),
        exact_matches=exact,connectivity_only_matches=connect_only,
        chemical_identity_protocol='RDKit canonical isomeric and nonisomeric SMILES; no other normalization',
        two_source_exact_group_closure=scoped_closure,
        same_experimental_assay_claimed=False,global_split_admitted=False)


def _original_train_rows(root,source,descriptors,receipt_rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows=[];file_proofs=[]
    for entry in descriptors:
        shard=entry['source_shard']
        if (source=='AZ' and '/shards/train/' not in shard['path']
                or source=='FreeSolv' and ('/FreeSolv/train/' not in shard['path'] or entry['role']!='train')):
            raise ValueError('Only original AZ and FreeSolv TRAIN shards permitted')
        if full9(root/shard['path'])!=tuple(entry['source_full9']):
            raise ValueError('Original shard device/stat/byte fingerprint changed')
        raw=read_exact(shard,root=root)
        group_entry=entry['original_group_sidecar'] if source=='AZ' else entry['group_sidecar']
        group_raw=read_exact(group_entry,root=root)
        fields=['task_id','record_id','source_family','source_revision','task_name',
                'native_schema','native_payload','native_payload_sha256','split','group_ids_json']
        table=pq.read_table(pa.BufferReader(raw),columns=fields,use_threads=False)
        groups=pq.read_table(pa.BufferReader(group_raw),columns=['task_id','effective_groups_json'],use_threads=False)
        if table.num_rows!=shard['rows'] or groups.num_rows!=table.num_rows:
            raise ValueError('Source and effective-group row cardinality mismatch')
        for ordinal,(row,g) in enumerate(zip(table.to_pylist(),groups.to_pylist(),strict=True)):
            original=receipt_rows[len(rows)]
            if (row['split']!='train' or row['native_schema']!='mora3.CorpusTask/v1'
                    or row['source_family']!=('ChEMBL37-AZ-experimental-logD7_4' if source=='AZ' else 'FreeSolv')
                    or row['task_name']!=('source_reported_experimental_logD7_4' if source=='AZ' else 'experimental_hydration_free_energy')
                    or row['record_id']!=original['record_id'] or row['native_payload_sha256']!=original['native_payload_sha256']
                    or sha256(row['native_payload']).hexdigest()!=row['native_payload_sha256']
                    or original['source_shard_sha256']!=shard['sha256']
                    or original['source_ordinal']!=ordinal or g['task_id']!=row['task_id']):
                raise ValueError('Original train row does not reproduce frozen full-source readback')
            declared=json.loads(g['effective_groups_json'],object_pairs_hook=_unique)
            source_groups=json.loads(row['group_ids_json'],object_pairs_hook=_unique)
            if (type(declared) is not list or type(source_groups) is not list
                    or len(source_groups)!=len(set(source_groups)) or
                    set(declared)!=set(source_groups)):
                raise ValueError('Actual original effective groups differ from source native groups')
            native=json.loads(row['native_payload'],object_pairs_hook=_unique)
            if (native['schema']!='mora3.CorpusTask/v1' or native['task']!=row['task_name']
                    or type(native['conditions']) is not dict):
                raise ValueError('Original source-native condition contract changed')
            smiles=native['conditions'].get('canonical_isomeric_smiles' if source=='AZ' else 'smiles')
            rows.append(dict(source=source,record_id=row['record_id'],task_id=row['task_id'],
                             smiles=smiles,groups=tuple(declared)))
        file_proofs.append(dict(path=shard['path'],sha256=shard['sha256'],
                                rows=table.num_rows,original_group_sidecar_sha256=group_entry['sha256']))
    if len(rows)!=len(receipt_rows):
        raise ValueError('Complete readback source row count differs from original observed train roles')
    return tuple(rows),file_proofs


def audit_original_az_freesolv_train(root):
    """Reopen and verify 4740 original training records without touching heldout."""
    root=Path(root).resolve()
    if root!=_ORIGINAL or Path.cwd().resolve()!=root:
        raise ValueError('Exact original source checkout cwd required for verified read-only replay')
    az=_readback(_AZ_RECEIPT,_AZ_SHA);fs=_readback(_FS_RECEIPT,_FS_SHA)
    if (az['original_train_rows']!=4184 or az['original_train_shards']!=17
            or fs['native_train_rows']!=556 or fs['original_role_rows']!={'train':556,'validation':42,'test':44}
            or az['all_train_candidate_common_equal'] is not True
            or fs['candidate_common_all_train_rows_equal'] is not True
            or az['both_native_iterators_exhausted'] is not True
            or fs['both_train_iterators_exhausted'] is not True):
        raise ValueError('Complete source/owner binding receipt changed')
    az_descriptors=az['original_source_files']
    fs_descriptors=[f for f in fs['source_files'] if f['role']=='train']
    if len(az_descriptors)!=17 or len(fs_descriptors)!=3:
        raise ValueError('Expected original 17+3 training shards; no test or validation scan')
    a,afile=_original_train_rows(root,'AZ',az_descriptors,az['rows'])
    f,ffile=_original_train_rows(root,'FreeSolv',fs_descriptors,fs['rows'])
    overlap=chemical_exposure_pairs(a,f)
    return dict(schema='Mora3.OriginalAZFreeSolvTwoSourceTrainChemicalExposure/v1',
        source_scope='only original AZ/FreeSolv train records, not all 62 routes or heldout',
        immutable_source_receipts={_AZ_RECEIPT:_AZ_SHA,_FS_RECEIPT:_FS_SHA},
        source_snapshots=dict(candidate=_CANDIDATE,common=_COMMON,view_policy=_POLICY),
        train_rows={'AZ':len(a),'FreeSolv':len(f)},
        train_shards={'AZ':len(afile),'FreeSolv':len(ffile)},
        original_files={'AZ':afile,'FreeSolv':ffile},
        chemical_exposure=overlap,
        source_rows_fully_checked=True,source_original_payloads_verified=True,
        candidate_common_bindings_preserved=True,
        heldout_source_files_read=False,heldout_target_payload_decoded=False,
        original_az_declared_study_single_component=az['original_declared_group_census']['component_count']==1,
        original_freesolv_local_groups_disjoint=fs['declared_group_census']['source_local_declared_groups_disjoint'],
        global_chemical_grouping_completed=False,global_split_admitted=False,
        training_admission=False,optimizer_updates=0,actual_proxy_runs=0,actual_target_runs=0)
