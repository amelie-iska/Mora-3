"""Native observational Rosetta molecule/profile tasks; no image or causal joins.

The immutable corpus admission is consumed, never promoted. Both directions are
input/target interfaces; this reader itself implements no molecular generator.
"""
from dataclasses import dataclass, replace
import hashlib
import io
import json
import math
from pathlib import Path

from .uniprot_annotations import _canonical, _verified_bytes


@dataclass(frozen=True)
class MoleculeProfileView:
    direction: str
    modality: str
    axis: tuple[str, ...]
    input_bytes: bytes
    target_bytes: bytes
    provenance_bytes: bytes
    binding_sha256: str

    def inputs(self):
        return json.loads(self.input_bytes)

    def targets(self):
        return json.loads(self.target_bytes)

    def provenance(self):
        return json.loads(self.provenance_bytes)


def molecule_profile_view(row, *, effective_groups):
    """Fail closed on payload/admission/axis drift; audit IDs stay outside inputs."""
    payload = row['native_payload']
    if hashlib.sha256(payload).hexdigest() != row['native_payload_sha256']:
        raise ValueError('native payload hash mismatch')
    task = json.loads(payload)
    direction = row['task_name']
    if (direction not in ('molecule_to_profile', 'profile_to_molecule') or
            task.get('schema') != 'mora3.CorpusTask/v1' or
            row['native_schema'] != task['schema'] or
            task.get('task') != 'source_observational_' + direction):
        raise ValueError('unsupported source observational task')
    receipt = json.loads(row['admission_receipt_json'])
    groups = tuple(effective_groups)
    if (row['admitted_for_task'] is not True or row['split'] != 'train' or
            receipt.get('source_rights') != 'CPG0003CC0' or
            receipt.get('full_feature_axis') is not True or
            receipt.get('no_same_cell_join') is not True or
            not groups or any(not isinstance(x, str) or not x for x in groups) or
            len(set(groups)) != len(groups)):
        raise ValueError('train-only source admission/groups gate failed')
    conditions, targets = task['conditions'], task['targets']
    modality = conditions.get('assay_modality')
    if modality not in ('CellPainting', 'L1000') or row['source_family'] != 'Rosetta-CDRP-molecule-profile-' + modality:
        raise ValueError('source modality mismatch')
    forward = direction == 'molecule_to_profile'
    expected = {'assay_modality', 'cell_line'} | ({'canonical_isomeric_smiles', 'known_profile_axis'} if forward else {'source_profile_array'})
    if set(conditions) != expected or set(targets) != ({'source_profile_array'} if forward else {'canonical_isomeric_smiles'}):
        raise ValueError('unexpected input/target fields: possible target leakage')
    cell = conditions['cell_line']
    if cell is not None and (not isinstance(cell, str) or not cell):
        raise ValueError('explicit cell line or missing required')
    missing = task['missing']
    if not {'physical_dose', 'exposure_time'} <= set(missing) or ((cell is None) != ('cell_line' in missing)):
        raise ValueError('source missingness mismatch')
    smiles = (conditions if forward else targets)['canonical_isomeric_smiles']
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError('source molecular representation required')
    array = (targets if forward else conditions)['source_profile_array']
    axis, values = array['axis_ids'], array['values']
    n = 1783 if modality == 'CellPainting' else 977
    if (len(axis) != n or len(values) != n or row['full_axis_values'] != n or
            any(not isinstance(x, str) or not x for x in axis) or len(set(axis)) != n or
            any(type(x) not in (int, float) or not math.isfinite(x) for x in values) or
            array.get('status') != 'observed' or not array.get('axis_identity') or
            array.get('units') != 'source_normalized_or_image_computed_feature_units' or
            (forward and conditions['known_profile_axis'] != axis)):
        raise ValueError('complete finite source axis/array required')
    if hashlib.sha256(_canonical(array)).hexdigest() != row['original_array_sha256']:
        raise ValueError('original source array hash mismatch')
    provenance = dict(task_id=row['task_id'], record_id=row['record_id'],
        source_revision=row['source_revision'], source_sha256=row['source_sha256'],
        payload_sha256=row['native_payload_sha256'], original_array_sha256=row['original_array_sha256'],
        effective_groups=groups, admission=receipt, audit=task['audit'], missing=missing,
        split='train', heldout_eligible=False, image_available=False, mask_available=False,
        observational_only=True, molecular_generation_implemented=False)
    pb = _canonical(provenance)
    ib, tb = _canonical(conditions), _canonical(targets)
    binding = hashlib.sha256(_canonical(dict(schema='mora3.molecule_profile_view.v1',
        inputs=hashlib.sha256(ib).hexdigest(), targets=hashlib.sha256(tb).hexdigest(),
        provenance=hashlib.sha256(pb).hexdigest()))).hexdigest()
    return MoleculeProfileView(direction, modality, tuple(axis), ib, tb, pb, binding)


def read_molecule_profile_views(manifest, *, root, expected_manifest_sha256,
                                modality, direction, limit):
    """Read at most limit rows from pinned, hash-verified corpus shards/sidecars.

    Explicit limit 1..256 is an inspection budget, never truncation of an axis.
    No evaluation eligibility is inferred from global corpus allocation.
    """
    import pyarrow.parquet as pq
    if type(limit) is not int or not 1 <= limit <= 256:
        raise ValueError('explicit bounded limit 1..256 required')
    if modality not in ('CellPainting', 'L1000') or direction not in ('molecule_to_profile', 'profile_to_molecule'):
        raise ValueError('unsupported modality/direction')
    root = Path(root)
    path = (root / manifest).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('manifest outside project')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_manifest_sha256:
        raise ValueError('manifest hash mismatch')
    m = json.loads(raw)
    source = 'Rosetta-CDRP-molecule-profile-' + modality
    views, seen = [], set()
    for shard in m['shards']:
        if shard.get('source_family') != source or shard.get('split') != 'train':
            continue
        rows = pq.read_table(io.BytesIO(_verified_bytes(root, shard))).to_pylist()
        sidecar = m['effective_group_lookup'][shard['path']]
        groups = pq.read_table(io.BytesIO(_verified_bytes(root, sidecar))).to_pylist()
        if len(rows) != shard['rows'] or len(groups) != sidecar['rows'] or len(rows) != len(groups):
            raise ValueError('shard/group row count mismatch')
        for row, group in zip(rows, groups):
            if row['task_id'] != group['task_id'] or row['split'] != shard['split']:
                raise ValueError('effective group identity/allocation mismatch')
            if row['task_name'] != direction:
                continue
            if row['task_id'] in seen:
                raise ValueError('duplicate task identity')
            seen.add(row['task_id'])
            view = molecule_profile_view(row, effective_groups=json.loads(group['effective_groups_json']))
            provenance = view.provenance()
            provenance.update(manifest_sha256=expected_manifest_sha256,
                shard_sha256=shard['sha256'], effective_group_sidecar_sha256=sidecar['sha256'])
            pb = _canonical(provenance)
            binding = hashlib.sha256(_canonical(dict(view_binding=view.binding_sha256,
                corpus_provenance=hashlib.sha256(pb).hexdigest()))).hexdigest()
            views.append(replace(view, provenance_bytes=pb, binding_sha256=binding))
            if len(views) == limit:
                return tuple(views)
    if not views:
        raise ValueError('no admitted matching source tasks')
    return tuple(views)
