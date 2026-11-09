"""Independent within-release native-wire and parent-closure conservation checks."""
from collections import Counter
from .curation import strict_json
from .contracts import canonical,number
from .grh_intake import GRHRelease,CONFIGURATIONS,MANIFEST_SHA256


def _same(a,b):
    # Exact wire values/types including zero signs: no coercive bool/int equality.
    return canonical(a)==canonical(b)


def verify_native_export(row,configuration):
    if type(row)is not dict or configuration not in CONFIGURATIONS:raise ValueError('native release row/configuration required')
    payload=strict_json(row['payload_json'])
    if type(payload)is not dict:raise ValueError('complete native payload object required')
    c=strict_json(row['condition_json'])if 'condition_json'in row else payload['conditions']
    t=strict_json(row['target_json'])if 'target_json'in row else payload['targets']
    if type(c)is not dict or type(t)is not dict:raise ValueError('complete native condition/target objects required')
    if not _same(c,payload['conditions'])or not _same(t,payload['targets']):
        raise ValueError('JSON/payload native conservation mismatch')
    if 'atomic_numbers'in c and 'bonds'in c:
        atoms=c['atomic_numbers'];bonds=c['bonds']
        if type(atoms)is not list or not atoms or any(type(z)is not int or not 1<=z<=118 for z in atoms):
            raise ValueError('invalid native atom axis')
        if type(bonds)is not list:raise ValueError('native bond list required')
        seen=set()
        for b in bonds:
            if (type(b)is not dict or not {'source','target','order','aromatic'}<=set(b)
                    or any(type(b[k])is not int or not 0<=b[k]<len(atoms)for k in ('source','target'))
                    or b['source']==b['target']or type(b['aromatic'])is not bool):
                raise ValueError('invalid native bond endpoints/roles')
            number(b['order'],'native bond order')
            key=tuple(sorted((b['source'],b['target'])))
            if key in seen or b['order']<=0:raise ValueError('duplicate or nonpositive native bond')
            seen.add(key)
            if 'stereo_atoms'in b and (type(b['stereo_atoms'])is not list or any(type(i)is not int or not 0<=i<len(atoms)for i in b['stereo_atoms'])):
                raise ValueError('invalid native stereo atom map')
    expected={}
    if configuration=='data':
        expected=dict(condition_atomic_numbers=c['atomic_numbers'],condition_coordinates_angstrom=c['coordinates_angstrom'],
            condition_bond_sources=[b['source']for b in c['bonds']],condition_bond_targets=[b['target']for b in c['bonds']],
            condition_bond_orders=[b['order']for b in c['bonds']],target_forces_kcal_mol_angstrom=t['forces_kcal_per_mol_angstrom'],
            target_energy_kcal_mol=t['energy_kcal_per_mol'],target_partial_charges_e=t['forcefield_partial_charges_e'],
            target_VR_H0_angstrom=t['heavy_atom_VR']['H0'],target_VR_H1_angstrom=t['heavy_atom_VR']['H1'],
            target_reference_coordinates_angstrom=payload['inverse_task']['targets']['reference_coordinates_angstrom'],
            evaluation_canonical_smiles=payload['canonical_smiles'],evaluation_reference_energy_kcal_mol=payload['reference_energy_kcal_per_mol'])
    elif configuration=='electronic_dynamics':
        edges=t['bond_edges']
        expected=dict(time=t['time'],target_occupation=t['occupation'],bond_sources=[e[0]for e in edges],bond_targets=[e[1]for e in edges],
            target_bond_current=t['bond_current'],target_charge_modulation=t['charge_modulation'],target_pair_source=t['pair_source'],
            target_connected_correlation=t['connected_density_correlation_with_orbital0'])
    elif configuration=='nuclear_md':
        expected=dict(time_fs=[f['time_fs']for f in t['frames']],coordinates_angstrom=[f['coordinates_angstrom']for f in t['frames']],
            total_energy_kcal_mol=t['total_energy_kcal_mol'])
    for field,value in expected.items():
        if field not in row or not _same(row[field],value):raise ValueError('native column conservation mismatch: '+field)
    return dict(native_columns_checked=len(expected),json_payloads_agree=True,bond_mapping_checked='bonds'in c,
        scientific_method_accuracy_established=False)


def group_closure(records):
    """Join declared group, chemical parent, and explicit data-parent record links.

    This does not infer geometric/similarity/homology relationships or inspect
    any external corpus. Output includes counts only, never source coordinates.
    """
    rows=tuple(records);parents=list(range(len(rows)));first={};by_id={};duplicates=0
    def root(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    def join(i,j):
        i,j=root(i),root(j)
        if i!=j:parents[max(i,j)]=min(i,j)
    for i,(config,rid,group,split,parent,parent_id)in enumerate(rows):
        if (config,rid)in by_id:duplicates+=1
        by_id[(config,rid)]=i
        keys=[('group',group)]
        if parent is not None:keys.append(('chemical-parent',config,parent))
        for key in keys:
            if key in first:join(i,first[key])
            else:first[key]=i
    resolved=unresolved=0
    for i,row in enumerate(rows):
        if row[5]is None:continue
        target=by_id.get(('data',row[5]))
        if target is None:unresolved+=1
        else:join(i,target);resolved+=1
    splits={}
    for i,row in enumerate(rows):splits.setdefault(root(i),set()).add(row[3])
    return dict(components=len(splits),conflicting_components=sum(len(s)>1 for s in splits.values()),
        resolved_parent_record_links=resolved,unresolved_parent_record_links=unresolved,duplicate_ids=duplicates,
        inferred_similarity_or_homology_edges=0,global_corpus_exposure_qualified=False)


def audit_conservation(root):
    import pyarrow.parquet as pq
    release=GRHRelease(root);records=[];failure=Counter();counts=Counter();columns=0
    for cfg in CONFIGURATIONS:
        names=sorted(n for n in release.files if n.startswith(cfg+'/')and n.endswith('.parquet'))
        for name in names:
            path=release._file(name)
            for batch in pq.ParquetFile(path).iter_batches(batch_size=16,use_threads=False):
                for row in batch.to_pylist():
                    counts[cfg]+=1;p=strict_json(row['payload_json']);s=p.get('source',{})
                    records.append((cfg,row['record_id'],row['group_id'],row['split'],p.get('parent')or s.get('parent_group'),p.get('parent_record_id')))
                    try:columns+=verify_native_export(row,cfg)['native_columns_checked']
                    except (ValueError,KeyError,TypeError)as exc:failure[type(exc).__name__+':'+str(exc)[:160]]+=1
            release._file(name)
    return dict(schema='Mora3.GRHIndependentConservation/v1',configuration_counts=dict(counts),source_rows=len(records),
        duplicate_native_columns_checked=columns,rejected_rows=sum(failure.values()),failure_reasons=dict(failure),
        exact_parent_closure=group_closure(records),source_manifest_sha256=MANIFEST_SHA256,
        native_value_transforms=0,source_mutations=0,training_admitted_rows=0,
        model_or_physical_quality_established=False)
