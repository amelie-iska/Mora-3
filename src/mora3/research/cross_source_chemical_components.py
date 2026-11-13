"""Original chemical group connected components, without heldout payload access.

This source-only closure uses *literal* authenticated molecule/scaffold sidecar
hashes. A common scaffold is a conservative grouping signal, not proof that two
molecules are the same, and cross-source hash domains have not been independently
standardized. Never admit this diagnostic as a final blind evaluation split.
"""
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import json

from mora3.data.admet import canonical
from mora3.research.cross_source_chemical_roles import (
    ROLES, audit_original_chemical_roles, compare_scoped_role_groups,
)


def digest(obj):
    return sha256(canonical(obj)).hexdigest()


def derive_literal_components(roles):
    """Close verified sidecar group keys, separately by axis and in their union.

    A component can connect different observations by original study/group keys.
    It is *not* a new chemical identity, an assay-level biological replicate, or
    a globally admitted partition. Training and heldout roles remain separate.
    """
    # The exact original routine independently validates all nine roles, exact
    # row schema, unique task IDs and axis-tagged source hashes before unioning.
    checked=compare_scoped_role_groups(roles)
    flat=sorted(((r['task_id'],source,role,r['molecule'],r['scaffold'])
                 for source in ROLES for role in ROLES[source]
                 for r in roles[source][role]),key=lambda x:x[0])
    if len(flat)!=len({r[0] for r in flat}):
        raise ValueError('Repeated original task identity')
    computed={}
    for axis in ('molecule','scaffold','combined'):
        parent=list(range(len(flat)));sizes=[1]*len(flat);owners={}
        def find(i):
            while parent[i]!=i:
                parent[i]=parent[parent[i]]
                i=parent[i]
            return i
        def union(i,j):
            i=find(i);j=find(j)
            if i==j:return
            if sizes[i]<sizes[j]:i,j=j,i
            parent[j]=i;sizes[i]+=sizes[j]
        for i,(_,_,_,molecule,scaffold) in enumerate(flat):
            keys=((molecule,) if axis=='molecule' else (scaffold,) if axis=='scaffold'
                  else (molecule,scaffold))
            for key in keys:
                if key in owners:union(i,owners[key])
                else: owners[key]=i
        by_component=defaultdict(list)
        for i,r in enumerate(flat):by_component[find(i)].append(r)
        groups=list(by_component.values())
        mixed_train=0;cross_source_heldout=0;cross_source=0
        biggest_training=0;summary=[];family_pair_counts=Counter()
        for group in groups:
            sources={r[1] for r in group}
            train=[r for r in group if r[2]=='train']
            held=[r for r in group if r[2]!='train']
            if len(sources)>1:cross_source+=1
            if train and held:mixed_train+=1
            if len({r[1] for r in held})>1:cross_source_heldout+=1
            biggest_training=max(biggest_training,len(train))
            for left in sorted(sources):
                for right in sorted(sources):
                    if left<right:family_pair_counts[left+'|'+right]+=1
            summary.append((sorted(r[0] for r in group),
                            sorted(sources),sorted({r[1]+'.'+r[2] for r in group})))
        summary.sort(key=lambda item:item[0])
        computed[axis]={
            'total_components':len(groups),
            'largest_component_rows':max(map(len,groups)),
            'largest_training_component_rows':biggest_training,
            'multiple_source_components':cross_source,
            'heldout_train_mixed_components':mixed_train,
            'cross_source_heldout_components':cross_source_heldout,
            'source_pair_component_counts':dict(sorted(family_pair_counts.items())),
            'full_component_membership_sha256':digest(summary),
            'source_group_hash_domains_independently_equated':False,
            'exact_stereochemical_identity_proven':False,
            'global_split_admitted':False,
        }
    return {
        'schema':'Mora3.OriginalFourFamilyLiteralGroupClosure/v1',
        'total_original_roles':sum(len(r) for r in ROLES.values()),
        'total_original_rows':len(flat),
        'original_role_counts':checked['role_rows'],
        'original_nine_role_ordered_id_commitment_sha256':digest(
            [(source,role,[r['task_id'] for r in roles[source][role]])
             for source in ROLES for role in ROLES[source]]),
        'axes':computed,
        'original_source_shards_replayed':False,
        'heldout_targets_or_conditions_opened':False,
        'actual_training_runs':0,
        'optimizer_updates':0,
        'training_admission':False,
    }


def audit_original_four_family_components(root):
    """Execute on the real source root, retaining independent original proof IDs."""
    verified=audit_original_chemical_roles(root,include_verified_roles=True)
    rows=verified.pop('_in_memory_verified_original_roles')
    comparison=verified['comparison']
    # A new graph audit is anchored to the previously reviewed exact sidecar
    # inventory and source-role controls; do not reinterpret target evidence.
    closure=derive_literal_components(rows)
    if (closure['total_original_rows']!=sum(comparison['role_rows'].values())
            or closure['original_role_counts']!=comparison['role_rows']):
        raise ValueError('Exact original role cardinality changed during closure')
    return {
        'schema':'Mora3.OriginalFourFamilyLiteralGroupClosureReadback/v1',
        'parent_sidecar_readback_sha256':digest(verified),
        'parent_sidecar_commitment_sha256':verified['source_role_file_commitment'],
        'parent_ordered_retained_commitment_sha256':verified['ordered_retained_group_commitment'],
        'common_source_snapshot_sha256':verified['common_source_snapshot_sha256'],
        'candidate_source_snapshot_sha256':verified['candidate_source_snapshot_sha256'],
        'original_sidecar_shards':verified['original_sidecar_shards'],
        'original_source_proofs':verified['original_source_proofs'],
        'original_excluded_role_rows':verified['original_excluded_role_rows'],
        'original_numerical_source_shards_read':False,
        'original_heldout_numerical_shards_read':False,
        'original_heldout_targets_decoded':False,
        'closure':closure,
        'global_chemical_identity_adjudicated':False,
        'independent_blind_split_validated':False,
        'training_admission':False,
        'actual_campaign_runs':0,
        'optimizer_updates':0,
    }


if __name__=='__main__':
    import sys
    if len(sys.argv)!=3:
        raise SystemExit('Usage: python -m mora3.research.cross_source_chemical_components ACTIVE_ROOT EXCLUSIVE_OUTPUT')
    p=Path(sys.argv[2]);
    if p.exists():raise SystemExit('Refusing to overwrite original output')
    data=audit_original_four_family_components(sys.argv[1])
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:
        json.dump(data,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({'readback_sha256':sha256(p.read_bytes()).hexdigest(),
                      'source_shards':data['original_sidecar_shards'],
                      'rows':data['closure']['total_original_rows'],
                      'components':{a:{k:v for k,v in r.items() if k in (
                          'total_components','largest_component_rows','heldout_train_mixed_components',
                          'multiple_source_components','cross_source_heldout_components')}
                          for a,r in data['closure']['axes'].items()}},sort_keys=True,indent=2))
