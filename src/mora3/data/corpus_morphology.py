"""Consume corpus-owned DeepBacs admission without promoting new source rights.

Full source raster and effective groups bind native checkpoints. Test records
remain development-exposed evaluation only. No perturbation/expression join.
"""
from dataclasses import dataclass, asdict
import hashlib
import io
import json
from pathlib import Path
from .raster_morphology import RasterMorphologyTask, _canonical
from .uniprot_annotations import _verified_bytes

EVIDENCE = 'source_observed_image_manual_ellipse_instance_annotation'
POLICY = 'official-source-train-test;test-development-exposed-not-blind'


@dataclass(frozen=True)
class CorpusMorphologyTask(RasterMorphologyTask):
    corpus_admission: bytes = b''

    def __post_init__(self):
        super().__post_init__()
        if type(self.corpus_admission) is not bytes or not self.corpus_admission:
            raise ValueError('immutable corpus admission required')
        metadata=json.loads(self.corpus_admission)
        if metadata.get('schema')!='mora3.corpus_morphology_admission.v1':
            raise ValueError('unsupported corpus morphology admission')
        payload=asdict(self);payload.pop('corpus_admission')
        if hashlib.sha256(_canonical(payload)).hexdigest()!=metadata.get('raster_sha256'):
            raise ValueError('raster/admission binding mismatch')

    def admission(self):
        return json.loads(self.corpus_admission)

    def canonical_bytes(self):
        payload=asdict(self);payload['corpus_admission']=self.admission()
        return _canonical(payload)


def corpus_morphology_task(row, *, effective_groups, corpus_binding):
    """Validate one source envelope; audit/target fields never enter image input."""
    raw=row['native_payload']
    if hashlib.sha256(raw).hexdigest()!=row['native_payload_sha256']:
        raise ValueError('native payload integrity mismatch')
    task=RasterMorphologyTask.from_bytes(raw)
    receipt=json.loads(row['admission_receipt_json'])
    groups=tuple(effective_groups)
    if (row['source_family']!='DeepBacs-Staph' or
            row['source_revision']!='Zenodo5550933.v1' or
            row['native_schema']!='mora3.RasterMorphologyTask/v1' or
            row['task_name']!='image_to_mask' or task.direction!='image_to_mask' or
            row['admitted_for_task'] is not True or row['evidence_class']!=EVIDENCE or
            row['split_policy']!=POLICY or receipt.get('policy')!=POLICY or
            receipt.get('source_quality')!='source-specific documented QC' or
            receipt.get('physical_assay_equivalence') is not False):
        raise ValueError('source/evidence/direction/admission gate failed')
    if (row['split']!=task.split or task.split not in ('train','test') or
            row['record_id']!=task.source_id or row['source_sha256']!=task.archive_sha256 or
            task.source!='https://zenodo.org/records/5550933' or task.revision!='v1' or
            task.license!='CC-BY-4.0' or
            task.annotation!='manual ellipses drawn on Nile Red fluorescence; whole-cell instance labels' or
            row['full_axis_values']!=task.height*task.width):
        raise ValueError('source identity/rights/shape mismatch')
    missing=json.loads(row['missing_context_json'])
    if (not {'intervention','paired_expression','donor','independent_heldout_status'}<=set(missing)
            or json.loads(row['condition_context_json'])!={}):
        raise ValueError('missingness/input context mismatch')
    if (not groups or len(set(groups))!=len(groups) or
            any(not isinstance(g,str) or not g for g in groups)):
        raise ValueError('distinct effective groups required')
    if (set(corpus_binding)!={'manifest_sha256','shard_sha256','groups_sha256'} or
            any(not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v)
                for v in corpus_binding.values())):
        raise ValueError('pinned corpus source/group bindings required')
    envelope={k:v for k,v in row.items() if k!='native_payload'}
    metadata=dict(schema='mora3.corpus_morphology_admission.v1',envelope=envelope,
        effective_groups=groups,corpus_binding=corpus_binding,
        independent_heldout=False,paired_expression=False,paired_intervention=False,
        raster_sha256=hashlib.sha256(task.canonical_bytes()).hexdigest())
    return CorpusMorphologyTask(**asdict(task),corpus_admission=_canonical(metadata))


def read_corpus_morphology(manifest, *, root, expected_manifest_sha256, split, limit):
    """Bounded pinned reader of admitted full images; no axis/pixel truncation."""
    import pyarrow.parquet as pq
    if split not in ('train','test') or type(limit) is not int or not 1<=limit<=32:
        raise ValueError('explicit source split and bounded limit 1..32 required')
    root=Path(root).resolve();path=(root/manifest).resolve()
    if not path.is_relative_to(root):raise ValueError('manifest outside project')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_manifest_sha256:
        raise ValueError('manifest hash mismatch')
    m=json.loads(raw);result=[];seen=set()
    for shard in m['shards']:
        if shard.get('source_family')!='DeepBacs-Staph' or shard.get('split')!=split:continue
        side=m['effective_group_lookup'][shard['path']]
        rows=pq.read_table(io.BytesIO(_verified_bytes(root,shard))).to_pylist()
        group_rows=pq.read_table(io.BytesIO(_verified_bytes(root,side))).to_pylist()
        if len(rows)!=shard['rows'] or len(group_rows)!=side['rows'] or len(rows)!=len(group_rows):
            raise ValueError('shard/group row counts mismatch')
        for row,group in zip(rows,group_rows):
            if row['task_id']!=group['task_id'] or row['split']!=split:
                raise ValueError('group/source split alignment mismatch')
            if row['task_name']!='image_to_mask':continue
            if row['task_id'] in seen:raise ValueError('duplicate task')
            seen.add(row['task_id'])
            result.append(corpus_morphology_task(row,effective_groups=json.loads(group['effective_groups_json']),
                corpus_binding=dict(manifest_sha256=expected_manifest_sha256,
                    shard_sha256=shard['sha256'],groups_sha256=side['sha256'])))
            if len(result)==limit:return tuple(result)
    return tuple(result)
