"""Read-only pinned GRH candidate intake, never experimental target admission.

No repository code/pickle execution, unit conversion, coordinate alignment,
implicit conformer averaging, or whole-payload encoding is performed. Within-
release groups do not establish cross-corpus or protein-family blind evaluation.
"""
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import math
from .contracts import canonical,digest,text
from .curation import file_digest,regular_file,strict_json

REVISION='53a6b66acaa69ef79584648f3f6dc068363c4837'
MANIFEST_SHA256='3610fcb330ee2b2b89037a38a39bf8c0509190b52f61a92313422f07ea9c5cad'
CONFIGURATIONS=('data','source_anchors','electronic_dynamics','nuclear_md')


@dataclass(frozen=True)
class GRHCandidate:
    configuration: str
    record_id: str
    group_id: str
    split: str
    condition_json: str
    target_json: str
    audit_json: str
    evidence_kind: str='computed'

    def __post_init__(self):
        if self.configuration not in CONFIGURATIONS or self.split not in ('train','validation','test'):
            raise ValueError('exact released GRH configuration and split required')
        for value in (self.record_id,self.group_id):text(value,'native identity')
        if self.evidence_kind!='computed':raise ValueError('GRH computed evidence cannot be relabeled observed')
        for raw in (self.condition_json,self.target_json,self.audit_json):
            if type(raw)is not str or type(strict_json(raw))is not dict:
                raise ValueError('exact finite native JSON object required')

    @property
    def training_admitted(self):return False

    def require_training_admission(self):
        raise ValueError('GRH candidates require original-source, cross-corpus and coordinator admission')


def project_candidate(row,*,configuration,expected_split):
    """Use only source-designated conditions; audit/targets never become predictors."""
    if type(row)is not dict or row.get('split')!=expected_split or configuration not in CONFIGURATIONS:
        raise ValueError('exact source row, native configuration and original split required')
    for name in ('record_id','group_id','family','origin_class','label_origin'):text(row.get(name),name)
    if configuration in ('data','source_anchors'):
        conditions=strict_json(row['condition_json']);targets=strict_json(row['target_json'])
    else:
        payload=strict_json(row['payload_json'])
        if type(payload)is not dict:raise ValueError('native dynamics object required')
        conditions=payload['conditions'];targets=payload['targets']
    if type(conditions)is not dict or type(targets)is not dict:
        raise ValueError('complete native condition/target objects required')
    # Preserve representation: missing fields are not filled or imputed.
    payload=strict_json(row['payload_json'])
    if type(payload)is not dict:raise ValueError('native audit payload must be an object')
    source=payload.get('source',{})
    if type(source)is not dict:raise ValueError('native source provenance must be an object')
    audit=dict(revision=REVISION,family=row['family'],origin_class=row['origin_class'],
        label_origin=row['label_origin'],planned_source_dataset=row.get('planned_source_dataset'),
        planned_source_record_id=row.get('planned_source_record_id'),original_group=row['group_id'],
        parent=payload.get('parent'),parent_record_id=payload.get('parent_record_id'),conformer=payload.get('conformer'),
        source_parent_group=source.get('parent_group'),source_revision=source.get('revision'),
        source_license=source.get('source_license'),native_units=payload.get('units'),
        payload_sha256=sha256(row['payload_json'].encode()).hexdigest(),
        parent_identity_and_global_exposure='separate-source-owner-review-required')
    return GRHCandidate(configuration,row['record_id'],row['group_id'],expected_split,
        canonical(conditions).decode(),canonical(targets).decode(),canonical(audit).decode())


def _matrix(value,n,cols,label):
    if type(value)is not list or len(value)!=n or any(type(r)is not list or len(r)!=cols for r in value):
        raise ValueError(f'complete native {label} axis required')
    if any(type(x) not in (int,float) or not math.isfinite(x) for r in value for x in r):
        raise ValueError(f'finite native {label} required')
    return tuple(tuple(x for x in row)for row in value)


def native_ef_view(task):
    """Method-specific MMFF94 energy/force view; no common experimental-array bypass."""
    if type(task)is not GRHCandidate:raise ValueError('exact GRH candidate required')
    task.__post_init__();c=strict_json(task.condition_json);t=strict_json(task.target_json)
    if c.get('method')!='MMFF94' or task.configuration not in ('data','source_anchors'):
        raise ValueError('only declared native MMFF94 static energy/force view is implemented')
    atoms=c.get('atomic_numbers')
    if type(atoms)is not list or not atoms or any(type(z)is not int or not 1<=z<=118 for z in atoms):
        raise ValueError('complete ordered native atomic numbers required')
    xyz=_matrix(c.get('coordinates_angstrom'),len(atoms),3,'condition coordinates')
    ekey='energy_kcal_per_mol' if task.configuration=='data' else 'energy_kcal_mol'
    fkey='forces_kcal_per_mol_angstrom' if task.configuration=='data' else 'forces_kcal_mol_angstrom'
    energy=t.get(ekey)
    if type(energy)not in(int,float)or not math.isfinite(energy):raise ValueError('finite native energy required')
    forces=_matrix(t.get(fkey),len(atoms),3,'target forces')
    return dict(atomic_numbers=tuple(atoms),condition_coordinates=xyz,target_forces=forces,target_energy=energy,
        coordinate_unit='angstrom',energy_unit='kcal/mol',force_unit='kcal/mol/angstrom',
        force_convention='negative-gradient-in-same-native-frame',method='MMFF94',
        evidence_kind='computed',absolute_energy_cross_molecule_comparable=False,training_admitted=False)


class GRHRelease:
    """Pinned files, bounded streaming, exact source hashes and original row ordinals."""
    def __init__(self,root):
        self.root=Path(root).absolute()
        p=regular_file(self.root/'manifest.json',max_bytes=1024*1024)
        if file_digest(p)!=MANIFEST_SHA256:raise ValueError('GRH final release manifest mismatch')
        self.manifest=strict_json(p.read_bytes())
        self.files=self.manifest['files']
        if type(self.files)is not dict:raise ValueError('complete final file manifest required')

    def _file(self,name):
        rel=Path(name)
        if rel.is_absolute() or '..'in rel.parts or name not in self.files:raise ValueError('unlisted release path')
        pin=self.files[name]
        p=regular_file(self.root/rel,max_bytes=256*1024*1024)
        if p.stat().st_size!=pin['bytes']or file_digest(p)!=pin['sha256']:
            raise ValueError('pinned original source bytes changed')
        return p

    def iter_candidates(self,configuration,split,*,batch_size=16):
        import pyarrow.parquet as pq
        if configuration not in CONFIGURATIONS or split not in ('train','validation','test'):
            raise ValueError('native configuration/split required')
        if type(batch_size)is not int or not 1<=batch_size<=64:raise ValueError('bounded native batch required')
        names=sorted(n for n in self.files if n.startswith(configuration+'/'+split+'-')and n.endswith('.parquet'))
        if not names:raise ValueError('requested original split is absent; not an empty admitted partition')
        for name in names:
            path=self._file(name);count=0;reader=pq.ParquetFile(path)
            for batch in reader.iter_batches(batch_size=batch_size,use_threads=False):
                for row in batch.to_pylist():
                    yield name,count,project_candidate(row,configuration=configuration,expected_split=split)
                    count+=1
            if count!=reader.metadata.num_rows or file_digest(path)!=self.files[name]['sha256']:
                raise ValueError('native row count or post-read source digest mismatch')

    def audit(self):
        """Full within-release candidate checks; no source coordinate values are emitted."""
        import pyarrow.parquet as pq
        counts={};groups={};ids=set();conflicts=set();duplicate_ids=0;ef=0;unsupported=0
        failures=Counter();families=Counter();origins=Counter();files=[]
        for config in CONFIGURATIONS:
            counts[config]={}
            for split in ('train','validation','test'):
                names=sorted(n for n in self.files if n.startswith(config+'/'+split+'-')and n.endswith('.parquet'))
                if not names:continue
                counts[config][split]=0
                for name in names:
                    path=self._file(name);reader=pq.ParquetFile(path);observed=0
                    for batch in reader.iter_batches(batch_size=16,use_threads=False):
                        for row in batch.to_pylist():
                            observed+=1;counts[config][split]+=1
                            key=(config,row['record_id'])
                            if key in ids:duplicate_ids+=1
                            ids.add(key)
                            old=groups.setdefault(row['group_id'],split)
                            if old!=split:conflicts.add(row['group_id'])
                            families[(config,row['family'])]+=1;origins[(config,row['origin_class'])]+=1
                            try:
                                task=project_candidate(row,configuration=config,expected_split=split)
                                c=strict_json(task.condition_json)
                                if c.get('method')=='MMFF94' and config in ('data','source_anchors'):
                                    native_ef_view(task);ef+=1
                                else:unsupported+=1
                            except (ValueError,TypeError,KeyError,OverflowError)as exc:
                                failures[type(exc).__name__+':'+str(exc)[:160]]+=1
                    if observed!=reader.metadata.num_rows:raise ValueError('footer/native original count mismatch')
                    self._file(name)
                    files.append(dict(path=name,bytes=path.stat().st_size,sha256=self.files[name]['sha256'],rows=observed,
                                      schema_sha256=sha256(str(reader.schema_arrow).encode()).hexdigest()))
        return dict(schema='Mora3.GRHIndependentCandidateAudit/v1',revision=REVISION,manifest_sha256=MANIFEST_SHA256,
            counts=counts,source_rows=sum(sum(s.values())for s in counts.values()),files=files,
            duplicate_ids_within_configuration=duplicate_ids,declared_group_cross_split_conflicts=len(conflicts),
            declared_groups=len(groups),native_mmff94_ef_candidates=ef,other_method_or_dynamic_candidates=unsupported,
            rejected_schema_rows=sum(failures.values()),rejection_reasons=dict(failures),
            families=[dict(configuration=k[0],family=k[1],rows=v)for k,v in sorted(families.items())],
            origin_classes=[dict(configuration=k[0],origin_class=k[1],rows=v)for k,v in sorted(origins.items())],
            newly_training_admitted_rows=0,global_exposure_qualified=False,source_rights_qualified=False,
            biological_experimental_rows=0,raw_coordinates_in_report=False,
            remaining_gates=['original anchors rights/provenance','cross-configuration parent closure beyond declared group IDs',
                'global master exposure and heldouts','full native objectives and masks','same-snapshot coordinator admission'])
