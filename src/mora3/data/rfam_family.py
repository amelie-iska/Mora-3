"""Pinned Rfam source-prior tasks; unknown families are never negative truth.

Consumes corpus-owner admission, not independent source re-admission. No image,
response pairing, held-out eligibility, or experimental function is inferred.
"""
from dataclasses import dataclass
import io
import json
from pathlib import Path
import re
from .nucleotide_prior import ALPHABET, digest
from .uniprot_annotations import _canonical as canonical, _verified_bytes

LABEL_POLICY='positive_source_assignments_only;all_unlisted_families_unknown'
POLICY='boundedfamilydiversesourceprior_trainonly;fullalloccurrenceparentgroups_fromnucleotidecomponent;homology_unknown'


def _ids(values, prefix, required=False):
    if (not isinstance(values,list) or (required and not values)
            or any(not isinstance(x,str) or re.fullmatch(prefix+r'[0-9]{5}',x) is None for x in values)
            or values!=sorted(set(values))):
        raise ValueError('ordered unique source family/clan accessions required')
    return tuple(values)


def project_family_task(d, direction):
    if direction not in ('prediction','generation'):
        raise ValueError('explicit prediction/generation direction required')
    if d.get('schema')!='mora3.CorpusTask/v1' or d.get('task')!='Rfam_source_family_'+direction+'_prior':
        raise ValueError('family task schema/direction mismatch')
    c=d['conditions'];t=d['targets']
    if direction=='prediction':
        if set(c)!={'source_native_sequence'} or set(t)!={'source_Rfam_family_ids','source_clan_ids','family_label_policy'} or t['family_label_policy']!=LABEL_POLICY:
            raise ValueError('prediction allowlist/unknown policy mismatch')
        fs=_ids(t['source_Rfam_family_ids'],'RF',True);cs=_ids(t['source_clan_ids'],'CL');seq=c['source_native_sequence']
    else:
        if set(c)!={'desired_source_Rfam_family_ids','desired_source_clan_ids'} or set(t)!={'source_native_sequence'}:
            raise ValueError('generation condition/target allowlist mismatch')
        fs=_ids(c['desired_source_Rfam_family_ids'],'RF',True);cs=_ids(c['desired_source_clan_ids'],'CL');seq=t['source_native_sequence']
    if not isinstance(seq,str) or not seq or set(seq)-set(ALPHABET):
        raise ValueError('complete native nucleotide symbols required')
    return json.loads(canonical(c)),fs,cs


@dataclass(frozen=True)
class RfamFamilyView:
    condition_bytes: bytes
    target_bytes: bytes
    provenance_bytes: bytes

    def inputs(self):return json.loads(self.condition_bytes)
    def targets(self):return json.loads(self.target_bytes)
    def provenance(self):return json.loads(self.provenance_bytes)


def _project_original_row(r, g, *, m, axis, shard, side, expected_manifest_sha256, direction):
    if r["task_id"] != g["task_id"]:
        raise ValueError("task/group alignment mismatch")
    if digest(r['native_payload'])!=r['native_payload_sha256']:raise ValueError('payload hash mismatch')
    d=json.loads(r['native_payload']);c,fs,cs=project_family_task(d,direction)
    seq=c['source_native_sequence'] if direction=='prediction' else d['targets']['source_native_sequence']
    receipt=json.loads(r['admission_receipt_json']);eg=json.loads(g['effective_groups_json']);original=json.loads(r['group_ids_json'])
    if (r['admitted_for_task'] is not True or r['split']!='train' or r['split_policy']!=POLICY
            or r['source_family']!='Rfam15.1-source-family-prior' or r['source_revision']!=m['source_release']
            or r['source_sha256']!=m['context_manifest']['sha256'] or r['native_schema']!=d['schema']
            or r['evidence_class']!='source_covariance_model_assigned_reference_family_annotation;notexperimentaltruth'
            or receipt!={'exact_identifier_join':True,'all_native_sequence_collisions_for_key_quarantined':True,'source_license':'CC0-1.0','source_family_assignments_not_experiment':True,'negative_family_targets':False}
            or r['record_id']!=digest(seq.encode()) or r['full_axis_values']!=len(seq)
            or r['task_id']!=digest(canonical([d['task'],d['conditions'],d['targets']]))
            or not set(fs)<=set(axis) or json.loads(r['missing_context_json'])!=d['missing']):
        raise ValueError('family source/train-only admission mismatch')
    if (not isinstance(eg,list) or not eg or any(not isinstance(x,str) or not x for x in eg)
            or len(set(eg))!=len(eg) or not set(original)<=set(eg)
            or 'nucleotide_native:'+r['record_id'] not in eg
            or any('rfam_model:Rfam15.1:'+f not in eg for f in fs)):
        raise ValueError('effective source groups mismatch')
    if json.loads(r['condition_context_json'])!={'segmentation_applicability':False,'profile_or_image_pair':False,'source_confidence_or_name_or_headers_in_predictors':False,'target_length_audit_only':True}:
        raise ValueError('applicability mismatch')
    p=dict(manifest_sha256=expected_manifest_sha256,shard_sha256=shard['sha256'],groups_sha256=side['sha256'],
           family_axis=axis,effective_groups=eg,envelope={k:v for k,v in r.items() if k!='native_payload'},
           source_audit=d['audit'],missing=d['missing'],heldout_eligible=False,paired_cellular_observation=False)
    return RfamFamilyView(canonical(c),canonical(d['targets']),canonical(p))


def read_rfam_families(manifest, *, root, expected_manifest_sha256, direction, limit):
    """Verify consumed shards/groups; retain complete targets and all admitted axes."""
    import pyarrow.parquet as pq
    if direction not in ('prediction','generation') or type(limit) is not int or not 1<=limit<=256:
        raise ValueError('bounded limit 1..256 and explicit direction required')
    root=Path(root).resolve();path=(root/manifest).resolve()
    if not path.is_relative_to(root):raise ValueError('manifest outside project')
    raw=path.read_bytes()
    if digest(raw)!=expected_manifest_sha256:raise ValueError('manifest hash mismatch')
    m=json.loads(raw)
    if (m['schema']!='mora3.RfamFamilyAnnotationAdmission/v1' or m['source_license']!='CC0-1.0'
            or m['source_release']!='Rfam15.1 January2026' or m['negative_labels_available'] is not False):
        raise ValueError('source registry admission mismatch')
    axis=_ids(m['family_ids'],'RF',True);out=[];seen=set()
    for shard in m['shards']:
        side=m['effective_group_lookup'][shard['path']]
        rows=pq.read_table(io.BytesIO(_verified_bytes(root,shard))).to_pylist()
        groups=pq.read_table(io.BytesIO(_verified_bytes(root,side))).to_pylist()
        if len(rows)!=shard['rows'] or len(groups)!=side['rows'] or len(rows)!=len(groups):
            raise ValueError('task/group row counts mismatch')
        for r,g in zip(rows,groups):
            if r['task_id']!=g['task_id']:raise ValueError('task/group alignment mismatch')
            if r['task_name']!='Rfam_source_family_'+direction+'_prior':continue
            if r['task_id'] in seen:raise ValueError('duplicate task')
            seen.add(r['task_id'])
            out.append(_project_original_row(r,g,m=m,axis=axis,shard=shard,side=side,
                expected_manifest_sha256=expected_manifest_sha256,direction=direction))
            if len(out)==limit:return tuple(out)
    return tuple(out)
