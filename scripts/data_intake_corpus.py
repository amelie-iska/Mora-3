"""Data-only corpus envelopes and conservative identity allocation utilities.

No source reader, model architecture or existing dataset is mutated. Envelopes
keep complete targets/audit evidence but expose only explicit conditions to an
adapter. Exact grouping does not establish homology or cross-assay equivalence.
"""
import hashlib
import json
import math


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


_corpus_consumption_guard = None


def _require_corpus_consumption(row, *, purpose):
    """Lazy repository-local guard; standalone scripts need no PYTHONPATH.

    Canonical audit serialization does not import a model-use guard. Review
    candidates may be nested inside the repository, so find only the fixed
    src/mora3/data/rgd1_consumption.py path among this file's ancestors.
    """
    global _corpus_consumption_guard
    if _corpus_consumption_guard is None:
        import importlib
        import sys
        from pathlib import Path
        relative_guard = Path('mora3/data/rgd1_consumption.py')
        source = next((parent/'src' for parent in Path(__file__).resolve().parents
                       if (parent/'src'/relative_guard).is_file()), None)
        if source is None:
            raise ImportError('repository source consumption guard unavailable')
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        guard = importlib.import_module('mora3.data.rgd1_consumption')
        if Path(guard.__file__).resolve() != (source/relative_guard).resolve():
            raise ImportError('unexpected source consumption guard module origin')
        _corpus_consumption_guard = guard
    _corpus_consumption_guard.require_corpus_consumption(row, purpose=purpose)


class CorpusTask:
    def __init__(self,task,conditions,targets,audit,missing):
        if not isinstance(task,str) or not task or not targets:
            raise ValueError('nonempty task and targets required')
        if any(not isinstance(x,dict) for x in (conditions,targets,audit)):
            raise ValueError('explicit field dictionaries required')
        if set(conditions)&set(targets) or set(conditions)&set(audit):
            raise ValueError('conditions cannot also be targets or audit')
        self._raw=canonical(dict(schema='mora3.CorpusTask/v1',task=task,conditions=conditions,targets=targets,audit=audit,missing=list(missing)))

    def canonical_bytes(self):
        return self._raw

    def input_bytes(self):
        d=json.loads(self._raw)
        _require_corpus_consumption(d, purpose='conditioning')
        return canonical(dict(task=d['task'],conditions=d['conditions']))

    @classmethod
    def from_bytes(cls,raw):
        d=json.loads(raw)
        if set(d)!={'schema','task','conditions','targets','audit','missing'} or d['schema']!='mora3.CorpusTask/v1':
            raise ValueError('unsupported envelope')
        return cls(d['task'],d['conditions'],d['targets'],d['audit'],d['missing'])


def connected_groups(records):
    parent={}
    def find(x):
        parent.setdefault(x,x)
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        parent[max(a,b)]=min(a,b)
    normalized=[]
    for record,groups in records:
        groups=sorted(set(groups)) or ['record:'+record]
        normalized.append((record,groups))
        for g in groups:union(groups[0],g)
    components={}
    for g in list(parent):components.setdefault(find(g),[]).append(g)
    keys={k:'component:'+hashlib.sha256(canonical(sorted(v))).hexdigest() for k,v in components.items()}
    return {r:keys[find(g[0])] for r,g in normalized}


def group_split(group,*,seed=20261031):
    value=int.from_bytes(hashlib.sha256(f'{seed}:{group}'.encode()).digest()[:8],'big')/2**64
    return 'train' if value<.8 else ('validation' if value<.9 else 'test')


def exclude_heldout_collisions(rows):
    heldout={g for r in rows if r['split'] in ('validation','test') for g in r['groups']}
    kept=[];rejected=[]
    for r in rows:
        collision=sorted(heldout&set(r['groups'])) if r['split']=='train' else []
        if collision:rejected.append(dict(r,conflicting_groups=collision))
        else:kept.append(r)
    return kept,rejected


def annotation_score(value):
    score=float(value)
    if not math.isfinite(score) or score!=int(score) or not 4<=score<=5:
        raise ValueError('reviewed annotation score4/5 required')
    return int(score)


def source_link(raw):
    link=json.loads(raw)
    if link is None or link==[]:
        return None
    if not isinstance(link,list) or any(not isinstance(x,str) or not x for x in link):
        raise ValueError('source linkage must be a nonempty string tuple or absent')
    return tuple(link)


def read_corpus_tasks(manifest,*,root=None,split='train',source=None,supported_only=True,limit=None):
    """Yield immutable canonical payloads;never infer training exposure/admission.

    Manifest-relative paths use the repository root,not the manifest directory.
    Each consumed complete shard is hash-checked before yielding any row. Native
    Consumption is source-policy guarded, including evaluation. Canonical
    envelope serialization remains available separately for native audits.
    Projection remains the model caller's responsibility; audit/source metadata
    must not be silently turned into predictor inputs.
    """
    from pathlib import Path
    import pyarrow.parquet as pq
    root=Path(root) if root is not None else Path(__file__).resolve().parents[1]
    if split not in ('train','validation','test') or (limit is not None and (type(limit) is not int or limit<1)):
        raise ValueError('explicit split and positive bound required')
    d=json.loads(Path(manifest).read_text());count=0
    for shard in d['shards']:
        if shard.get('split') not in (None,split) or (source is not None and shard.get('source_family') not in (None,source)):
            continue
        path=root/shard['path'];raw=path.read_bytes()
        if len(raw)!=shard['bytes'] or hashlib.sha256(raw).hexdigest()!=shard['sha256']:
            raise ValueError('corpus shard integrity mismatch: '+str(path))
        rows=pq.read_table(path).to_pylist()
        if len(rows)!=shard['rows']:
            raise ValueError('corpus shard row count mismatch')
        effective=None
        entry=d.get('effective_group_lookup',{}).get(shard['path'])
        if 'effective_group_lookup' in d and entry is None:
            raise ValueError('missing effective group metadata for consumed shard')
        if entry is not None:
            group_path=root/entry['path'];group_raw=group_path.read_bytes()
            if len(group_raw)!=entry['bytes'] or hashlib.sha256(group_raw).hexdigest()!=entry['sha256']:
                raise ValueError('group metadata integrity mismatch')
            group_rows=pq.read_table(group_path).to_pylist()
            effective={r['task_id']:r['effective_groups_json'] for r in group_rows}
            if len(group_rows)!=entry['rows'] or len(effective)!=len(group_rows) or set(effective)!={r['task_id'] for r in rows}:
                raise ValueError('group/task allocation mismatch')
        for row in rows:
            if effective is not None:
                row=dict(row,source_group_ids_json=row.get('group_ids_json'),group_ids_json=effective[row['task_id']],group_metadata_status='manifest_verified_effective_groups')
            if row['split']!=split or (source is not None and row['source_family']!=source):
                continue
            supported=row.get('model_adapter_supported',row['native_schema']=='mora3.ObservationalSciPlexTask/v1')
            if supported_only and not supported:
                continue
            if hashlib.sha256(row['native_payload']).hexdigest()!=row['native_payload_sha256']:
                raise ValueError('native payload integrity mismatch')
            _require_corpus_consumption(row, purpose='objective' if split=='train' else 'evaluation')
            yield row
            count+=1
            if limit is not None and count>=limit:
                return



def chemical_identity(smiles):
    """Derived leakage identity only; caller must retain original source SMILES.

    Atom-map labels/explicit hydrogen syntax are not compound identity. Source
    isomeric stereo stays in the derived isomeric string; only the leakage group
    removes stereo. No tautomer,protonation,salt or assay equivalence is implied.
    """
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    mol=Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('unparseable source molecule')
    for atom in mol.GetAtoms():atom.SetAtomMapNum(0)
    mol=Chem.RemoveHs(mol)
    iso=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True)
    copy=Chem.Mol(mol);Chem.RemoveStereochemistry(copy)
    base=Chem.MolToSmiles(copy,canonical=True,isomericSmiles=False)
    scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=copy,includeChirality=False) or base
    return dict(canonical_isomeric_smiles=iso,molecule_group='molecule:'+hashlib.sha256(base.encode()).hexdigest(),scaffold_group='scaffold:'+hashlib.sha256(scaffold.encode()).hexdigest())


def masked_coordinate_task(atoms,masked_indices,*,audit=None):
    """Known deposited atom-set reconstruction,not unknown-topology generation.

    The mask is supplied independently of coordinates. Source atom identities
    and unmasked positions are conditions. B-factors,occupancy,validation,source
    hashes and entry IDs cannot become conditions through this allowlist.
    """
    indices=tuple(masked_indices);n=len(atoms)
    if not indices or len(set(indices))!=len(indices) or len(indices)>=n or any(type(i) is not int or not 0<=i<n for i in indices):
        raise ValueError('nonempty proper subset of known atom-axis indices required')
    hidden=set(indices);conditions=[];positions=[];targets=[]
    fields=('type_symbol','label_atom_id','label_comp_id','label_asym_id','label_seq_id','label_alt_id')
    for i,atom in enumerate(atoms):
        xyz=[float(atom[k]) for k in ('x_angstrom','y_angstrom','z_angstrom')]
        if not all(math.isfinite(v) for v in xyz):raise ValueError('finite source coordinates required')
        identity={key:atom[key] for key in fields}
        if not all(isinstance(v,str) and v for v in identity.values()):raise ValueError('source atom identity strings required')
        conditions.append(identity);positions.append(None if i in hidden else xyz);targets.append(xyz if i in hidden else None)
    return CorpusTask('source_structure_masked_coordinate_reconstruction',dict(known_atom_axis=conditions,positions_angstrom=positions,coordinate_unit='angstrom'),dict(masked_positions_angstrom=targets),audit or {},['experimental_diffraction_data','independent_source_family_heldout','unknown_atom_topology'])


def gene_reference_ids(value):
    values=sorted({x.strip() for x in value.split(';') if x.strip()},key=lambda x:(len(x),x))
    if any(not x.isdigit() or int(x)<1 for x in values):
        raise ValueError('unverified nonnumeric source GeneID')
    return tuple(values)


def protein_function_task(row):
    available={}
    for source,key in [('Function [CC]','source_curated_function'),('Catalytic activity','source_curated_catalytic_activity')]:
        if row[source].strip():available[key]=row[source]
    go=tuple(x.strip() for x in row['Gene Ontology IDs'].split(';') if x.strip())
    if go:available['source_GO_identifiers']=list(go)
    if not available:raise ValueError('no available functional annotation;retain separately eligible sequence task')
    if row['Protein names'].strip():available['source_protein_name']=row['Protein names']
    conditions=dict(amino_acid_sequence=row['Sequence'],organism_taxon=int(row['Organism (ID)']))
    missing=[key for source,key in [('Function [CC]','function_annotation'),('Catalytic activity','catalytic_activity_annotation'),('Gene Ontology IDs','GO_annotation')] if not row[source].strip()]
    missing+=['experimental_assay_context','cellular_state','protein_3D_structure']
    return CorpusTask('sequence_to_curated_function_annotations',conditions,available,dict(original_source_row=row,evidence='source_curated_mixed_ECO_annotations;not blanket experimental measurements',gene_reference_scope='sourceasserted_gene/protein_reference;not samecell orassayedisoform'),missing)


def integer_gene_id(value):
    import re
    if not isinstance(value,str):raise ValueError('original source string required')
    value=value.strip()
    if value in ('','-666'):return None
    if not re.fullmatch(r'[0-9]+(?:[.]0+)?',value):raise ValueError('nonintegral or nonnumeric source GeneID')
    result=str(int(value.split('.')[0]))
    if result=='0':raise ValueError('positive source GeneID required')
    return result


def rosetta_gene_reference(metadata):
    """Known genetic-source fields only;never infer a GeneID from a symbol.

    This is a reference identity,not the sequence of an assayed mutant/clone.
    Caller must restrict to the documented genetic studies.
    """
    fields={key:metadata[key] for key in ('Metadata_GeneID','Metadata_NCBIGeneID','NCBIGeneID','pert_univ_id') if key in metadata}
    try:values={integer_gene_id(v) for v in fields.values()}-{None}
    except ValueError:return dict(gene_id=None,status='invalid_source_gene_reference',source_fields=fields)
    if len(values)>1:return dict(gene_id=None,status='conflicting_source_gene_fields',source_fields=fields)
    return dict(gene_id=next(iter(values),None),status='source_numeric_reference' if values else 'missing_source_gene_reference',source_fields=fields)


def audit_table(rows):
    """Preserve every flat audit field across heterogeneous source schemas.

    Arrow from_pylist otherwise takes field names from only the first row.
    Null means a field is absent for that source,not fabricated context.
    """
    import pyarrow as pa
    fields=sorted({key for row in rows for key in row})
    return pa.Table.from_pylist([{key:row.get(key) for key in fields} for row in rows])

def main():
    import argparse
    p=argparse.ArgumentParser(description='Read-only corpus shard integrity/probe;no training or data mutation')
    p.add_argument('--manifest',required=True);p.add_argument('--split',choices=['train','validation','test'],default='train');p.add_argument('--source');p.add_argument('--limit',type=int,default=2);p.add_argument('--include-unsupported',action='store_true');a=p.parse_args()
    result=[]
    for row in read_corpus_tasks(a.manifest,split=a.split,source=a.source,supported_only=not a.include_unsupported,limit=a.limit):
        result.append({k:row[k] for k in ['record_id','source_family','native_schema','split','native_payload_sha256']})
    print(json.dumps(dict(probe_rows=len(result),source=a.source,split=a.split,rows=result,optimizer_steps=0),indent=2))


if __name__=='__main__':
    main()
