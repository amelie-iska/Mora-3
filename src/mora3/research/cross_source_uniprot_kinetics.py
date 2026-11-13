"""Actual original train-side UniProt kinetics-to-sequence prior group exposure.

Read SHA-authenticated original train shards and source-effective protein group
sidecars; never decode protein targets or any heldout source. The group join is
a conservative split-risk relation, NOT experimental, same-molecule or causal
identity and not global source admission or μP execution.
"""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json
import re

from mora3.data.admet import canonical
from mora3.data.finite_union_native_rows import decode_disposition, full9, read_exact
from mora3.research.primary_native_routes import finite_view_policy_sha256
from mora3.research.remaining_eight_registry import load_assets

_POLICY='c3812ecfcfea8a01f81769bf5ef6fb3880e04a0ac054a6349e9272db14765393'
_KINETIC_READBACK='outputs/common-source-uniprot-kinetics-complete-20261111/original-complete-uniprot-kinetics-readback-v2.json'
_KINETIC_READBACK_SHA='6d9922b1af68973836cac5dac63c10b2e9ab656014156219b4ea3dcbf4ec1c0d'
_ALLOWED=('accession','sequence','detected_protein_guard','pubmed')
_TASKS={'kinetics':('UniProt-kinetics','curated_kinetic_annotation_value'),
        'sequence':('UniProt','protein_sequence_modeling')}
_HEX=re.compile(r'[0-9a-f]{64}\Z')


def _sha(x):return sha256(x).hexdigest()


def _strict_json(raw):
    def unique(pairs):
        d={}
        for k,v in pairs:
            if k in d:raise ValueError('Repeated original source group JSON key')
            d[k]=v
        return d
    return json.loads(raw,object_pairs_hook=unique)


def _valid_groups(groups):
    return (type(groups)is tuple and len(groups)>0 and len(groups)==len(set(groups))
            and all(type(g)is str and ':' in g and g.split(':',1)[0]in _ALLOWED
                    and bool(g.split(':',1)[1]) for g in groups))


def _component_stats(records, *, namespaced, class_filter):
    parent=list(range(len(records))); by_group={}
    def find(k):
        while parent[k]!=k:
            parent[k]=parent[parent[k]]
            k=parent[k]
        return k
    for i,r in enumerate(records):
        for g in r['groups']:
            if g.split(':',1)[0] not in class_filter:continue
            key=(r['source'],g) if namespaced else g
            if key in by_group:
                parent[find(i)]=find(by_group[key])
            else:by_group[key]=i
    members={}
    for i,r in enumerate(records):
        members.setdefault(find(i),[]).append(r)
    cross=[rows for rows in members.values() if len({r['source'] for r in rows})==2]
    cross_sizes=Counter(len(rows) for rows in cross)
    exposed=Counter(r['source'] for rows in cross for r in rows)
    commitment=_sha(canonical(sorted(
        _sha(canonical(sorted(r['task_id'] for r in rows))) for rows in cross)))
    return dict(components=len(members),cross_components=len(cross),
                cross_rows=dict(exposed),cross_size_histogram=dict(sorted(cross_sizes.items())),
                largest_cross_component=max((len(rows)for rows in cross),default=0),
                cross_component_commitment=commitment)


def protein_group_closure(kinetics,sequence):
    """Intersect *original effective train groups*, keeping study vs identity distinct."""
    if (type(kinetics)is not tuple or type(sequence)is not tuple
            or not kinetics or not sequence):raise ValueError('Two complete nonempty source collections required')
    all_rows=kinetics+sequence
    seen=set()
    for source,rows in (('kinetics',kinetics),('sequence',sequence)):
        for r in rows:
            if (type(r)is not dict or set(r)!={'source','task_id','groups'}
                    or r['source']!=source or type(r['task_id'])is not str
                    or _HEX.fullmatch(r['task_id'])is None or r['task_id']in seen
                    or not _valid_groups(r['groups'])):
                raise ValueError('Unqualified, repeated or incomplete original train/group identity')
            seen.add(r['task_id'])
    groups={s:set(g for r in rows for g in r['groups'])
            for s,rows in (('kinetics',kinetics),('sequence',sequence))}
    shared=groups['kinetics'] & groups['sequence']
    classes=dict(sorted(Counter(g.split(':',1)[0] for g in shared).items()))
    full=_component_stats(all_rows,namespaced=False,class_filter=_ALLOWED)
    identity=_component_stats(all_rows,namespaced=False,
                              class_filter=('accession','sequence','detected_protein_guard'))
    independent=_component_stats(all_rows,namespaced=True,class_filter=_ALLOWED)
    kinetics_independent=_component_stats(kinetics,namespaced=True,class_filter=_ALLOWED)
    return dict(source_rows={'kinetics':len(kinetics),'sequence':len(sequence)},
                original_source_local_components=independent['components'],
                kinetics_source_local_components=kinetics_independent['components'],
                combined_components=full['components'],
                cross_source_components=full['cross_components'],
                kinetics_rows_cross_exposed=full['cross_rows'].get('kinetics',0),
                sequence_rows_cross_exposed=full['cross_rows'].get('sequence',0),
                cross_source_component_size_histogram=full['cross_size_histogram'],
                largest_cross_component_rows=full['largest_cross_component'],
                cross_component_commitment=full['cross_component_commitment'],
                direct_cross_source_group_classes=classes,
                cross_source_identity_only_components=identity['cross_components'],
                identity_only_kinetics_rows_exposed=identity['cross_rows'].get('kinetics',0),
                shared_declared_group_count=len(shared),
                grouping_policy='literal original groups; cross-source accession/sequence/homology and separate publication/study exposure; no inferred aliases',
                no_heldout_source_payloads=True,all_original_values_absent=True,
                same_protein_measurement_claimed=False,
                global_split_admitted=False,training_admission=False)


def audit_original_uniprot_kinetic_sequence_exposure(root):
    """Scan only pinned original TRAIN groups of two native families, no targets."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    root=Path(root).resolve()
    if not root.is_dir()or Path.cwd().resolve()!=root:
        raise ValueError('Use the original live source root, not a copied source')
    from mora3.research.primary_campaign_source_rebind import load_inactive_common_source_rebind
    from mora3.research.primary_source_factory_preflight import document
    ctx=load_inactive_common_source_rebind(root)
    doc=document(root,ctx.evidence)
    if (ctx.common_snapshot.sha256!='79d328f542d1158581875ef325e56f213b407992f95db6ee0c7352a3459669d4'
            or doc['inactive_candidate_snapshot']['sha256']!='1091e3f27d57cb70203ab9fce1136836d0cd96eef955b6682a3e6264abadc733'):
        raise ValueError('Reviewed candidate/common source binding changed')
    readback_path=Path(__file__).resolve().parents[3]/_KINETIC_READBACK
    raw=readback_path.read_bytes()
    if _sha(raw)!=_KINETIC_READBACK_SHA:raise ValueError('Complete original kinetic readback changed')
    prior=_strict_json(raw)
    if (prior['native_train_rows']!=618 or prior['original_source_rows']!=622
            or prior['training_admission'] is not False
            or prior['original_view_policy_sha256']!=_POLICY
            or prior['both_original_train_iterators_exhausted'] is not True):
        raise ValueError('Unqualified kinetic prior readback')
    v=load_assets(root)[0]
    if finite_view_policy_sha256(v)!=_POLICY:raise ValueError('Changed original source study filter')
    expected={(r['source_shard_sha256'],r['source_ordinal']):r for r in prior['rows']}
    if len(expected)!=618:raise ValueError('Duplicate original kinetic prior coordinates')
    selected={'kinetics':[],'sequence':[]};files=[];original=Counter();excluded=Counter()
    descriptors=[]
    for d in v.descriptors:
        path=d['source_shard']['path']
        label=('kinetics' if '/UniProt-kinetics/train/' in path else
               'sequence' if '/UniProt/train/' in path else None)
        if label is not None:descriptors.append((label,d))
    if len(descriptors)!=410:
        raise ValueError('Original 407 protein prior + 3 kinetic source shards changed')
    for label,d in descriptors:
        shard=d['source_shard'];path=shard['path'];side=d['original_group_sidecar']
        if full9(root/path)!=tuple(d['source_full9']):
            raise ValueError('Original full9 source shard changed: '+path)
        raw_source=read_exact(shard,root=root)
        raw_group=read_exact(side,root=root)
        cols=['task_id','source_family','source_revision','task_name','native_schema','split','group_ids_json']
        native=pq.read_table(pa.BufferReader(raw_source),columns=cols,use_threads=False)
        groups=pq.read_table(pa.BufferReader(raw_group),use_threads=False)
        if (native.num_rows!=shard['rows'] or groups.num_rows!=shard['rows']
                or set(groups.column_names)!={'effective_groups_json','evidence_scope',
                     'source_file','source_group_ids_json','task_id'}):
            raise ValueError('Original source and grouping row/cardinality mismatch')
        disposition=decode_disposition(d,root=root)
        if len(disposition)!=shard['rows']:raise ValueError('Original disposition length mismatch')
        fam,task=_TASKS[label]
        for ordinal,(r,g,code) in enumerate(zip(native.to_pylist(),groups.to_pylist(),disposition,strict=True)):
            original[label]+=1
            if (r['split']!='train' or r['source_family']!=fam or r['task_name']!=task
                    or r['task_id']!=g['task_id'] or g['source_file']!=path
                    or g['evidence_scope']!='identity/leakagegroupsonly;notpredictorfeaturesorsamecell/effectproof'):
                raise ValueError('Original protein family/study/role/sidecar changed')
            source_groups=_strict_json(r['group_ids_json'])
            if source_groups!=_strict_json(g['source_group_ids_json']):
                raise ValueError('Original native and source-group identities disagree')
            effective=_strict_json(g['effective_groups_json'])
            if (type(source_groups)is not list or type(effective)is not list
                    or not set(source_groups).issubset(effective)
                    or not _valid_groups(tuple(effective))):
                raise ValueError('Malformed or unsupported original biological groups')
            if code!=0 or (path,ordinal) in v._ordinal_exclusions:
                excluded[label]+=1
                if label=='kinetics' and (shard['sha256'],ordinal) in expected:
                    raise ValueError('Previously retained kinetic record is excluded')
                continue
            if label=='kinetics':
                key=(shard['sha256'],ordinal)
                old=expected.pop(key,None)
                if old is None or old['effective_group_sha256']!=_sha(canonical(effective)) or old['source_group_sha256']!=_sha(canonical(source_groups)):
                    raise ValueError('Kinetic source group mismatch against independently completed source audit')
            selected[label].append(dict(source=label,task_id=r['task_id'],groups=tuple(effective)))
        files.append(dict(label=label,path=path,source_sha256=shard['sha256'],
                          group_sha256=side['sha256'],source_rows=shard['rows']))
    if (expected or original['kinetics']!=622 or len(selected['kinetics'])!=618
            or original['sequence']!=104176):
        raise ValueError('Full original kinetic or UniProt sequence source census differs')
    proof=protein_group_closure(tuple(selected['kinetics']),tuple(selected['sequence']))
    if proof['kinetics_source_local_components']!=prior['effective_group_census']['connected_components']:
        raise ValueError('Independent original source-local kinetic grouping mismatch')
    return dict(schema='Mora3.UniProtKineticsSequenceOriginalGroupExposure/v1',
                candidate_source_snapshot_sha256=doc['inactive_candidate_snapshot']['sha256'],
                common_source_snapshot_sha256=ctx.common_snapshot.sha256,
                source_native_view_policy_sha256=_POLICY,
                prior_full_kinetics_readback_sha256=_KINETIC_READBACK_SHA,
                original_source_rows=dict(original),excluded_original_rows=dict(excluded),
                original_train_shards=len(files),original_shard_manifest_commitment=_sha(canonical(files)),
                source_scoped_group_audit=proof,source_targets_decoded=False,
                heldout_source_shards_opened=False,model_forward_calls=0,
                optimizer_updates=0,actual_campaign_runs=0,
                same_owner_validation_reference=False,global_split_admitted=False,
                training_admission=False)
