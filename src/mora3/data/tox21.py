"""Source-native Tox21 call priors, with distinct inactive and unknown states.

No dose-response, image, clinical-risk or cross-source pairing is inferred.
The inverse view exposes desired observed-call patterns, not guaranteed activity.
"""
from dataclasses import dataclass, replace
import hashlib
import io
import json
from pathlib import Path

from .uniprot_annotations import _canonical, _verified_bytes

ASSAYS = ('NR-AhR', 'NR-AR', 'NR-AR-LBD', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
          'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53')
FORWARD = 'source_Tox21_molecule_to_observed_assay_calls'
INVERSE = 'source_Tox21_observed_call_pattern_to_molecule_prior'
SOURCE = 'Tox21-original-source-call-prior'


@dataclass(frozen=True)
class Tox21View:
    direction: str
    values: tuple
    observed_mask: tuple[bool, ...]
    input_bytes: bytes
    target_bytes: bytes
    provenance_bytes: bytes
    binding_sha256: str
    axis: tuple[str, ...] = ASSAYS

    def inputs(self):
        return json.loads(self.input_bytes)

    def targets(self):
        return json.loads(self.target_bytes)

    def provenance(self):
        return json.loads(self.provenance_bytes)


def _binding(inputs, targets, provenance):
    return hashlib.sha256(_canonical(dict(schema='mora3.Tox21View/v1',
        inputs=hashlib.sha256(inputs).hexdigest(),
        targets=hashlib.sha256(targets).hexdigest(),
        provenance=hashlib.sha256(provenance).hexdigest()))).hexdigest()


def tox21_view(row, *, effective_groups):
    """Validate corpus admission without changing it; only allow explicit inputs."""
    payload = row['native_payload']
    if hashlib.sha256(payload).hexdigest() != row['native_payload_sha256']:
        raise ValueError('native payload hash mismatch')
    task = json.loads(payload)
    direction, split = row['task_name'], row['split']
    if (direction not in (FORWARD, INVERSE) or task.get('task') != direction or
            task.get('schema') != 'mora3.CorpusTask/v1' or row['native_schema'] != task['schema'] or
            row['source_family'] != SOURCE or split not in ('train', 'validation', 'test')):
        raise ValueError('unsupported native Tox21 task/source/split')
    receipt = json.loads(row['admission_receipt_json'])
    if split == 'train':
        eligible = (receipt.get('license') == 'US-government-public-domainoriginalTox21data;officialNCATSproviderstatement'
            and receipt.get('all12assayaxes_retained') is True
            and receipt.get('physical_dose_time_missing') is True
            and receipt.get('measuredcallannotation_prioronly') is True)
    else:
        eligible = (receipt.get('source_license') == 'originalUSgovTox21publicdomain'
            and receipt.get('source_protocol_split') is True
            and receipt.get('known_chemical_train_overlap') is False
            and receipt.get('prior_model_and_benchmark_exposure_unknown') is True
            and receipt.get('not_final_unseen_biological_validation') is True)
    groups = tuple(effective_groups)
    original_groups = json.loads(row['group_ids_json'])
    if (row['admitted_for_task'] is not True or not eligible or not groups or
            any(not isinstance(g, str) or not g for g in groups) or len(set(groups)) != len(groups) or
            not set(original_groups) <= set(groups) or
            not any(g.startswith('molecule:') for g in original_groups) or
            not any(g.startswith('scaffold:') for g in original_groups)):
        raise ValueError('admission/rights/chemical group gate failed')
    conditions, targets, audit = task['conditions'], task['targets'], task['audit']
    forward = direction == FORWARD
    expected_inputs = ({'canonical_isomeric_smiles', 'known_assay_axis'} if forward else
        {'known_assay_axis', 'desired_source_call_values', 'desired_source_call_mask'})
    expected_targets = {'source_call_values', 'source_call_mask'} if forward else {'canonical_isomeric_smiles'}
    if set(conditions) != expected_inputs or set(targets) != expected_targets:
        raise ValueError('unexpected fields; possible target leakage')
    if tuple(conditions['known_assay_axis']) != ASSAYS or row['full_axis_values'] != 12:
        raise ValueError('exact full source assay order required')
    smi = (conditions if forward else targets)['canonical_isomeric_smiles']
    if not isinstance(smi, str) or not smi.strip():
        raise ValueError('source molecular representation required')
    calls = targets if forward else conditions
    values = calls['source_call_values' if forward else 'desired_source_call_values']
    mask = calls['source_call_mask' if forward else 'desired_source_call_mask']
    if (len(values) != 12 or len(mask) != 12 or any(type(m) is not bool for m in mask) or
            not any(mask) or any((type(v) is not int or v not in (0, 1)) if m else v is not None
                                 for v, m in zip(values, mask))):
        raise ValueError('observed binary calls and unknown None mask required')
    ref = audit['source_ref']
    role = {'train': 'source_training', 'validation': 'source_leaderboard_testing',
            'test': 'source_final_evaluation_structures'}[split]
    if (ref != receipt.get('source_ref') or ref.get('native_jsonl_sha256') != row['source_sha256'] or
            audit.get('source_protocol_role') != role or audit.get('source_missing_not_negative') is not True or
            audit.get('physical_dose') is not None or audit.get('exposure_time') is not None or
            'perassayphysicaldose_exposuretime_cellline' not in task['missing']):
        raise ValueError('source identity/protocol/context mismatch')
    provenance = dict(task_id=row['task_id'], record_id=row['record_id'],
        source_revision=row['source_revision'], source_sha256=row['source_sha256'],
        payload_sha256=row['native_payload_sha256'], split=split,
        effective_groups=groups, admission=receipt, audit=audit, missing=task['missing'],
        evaluation_scope='known_chemical_group_source_protocol_subset' if split != 'train' else 'train_only',
        prior_public_model_exposure_unknown=True, independent_biological_validation=False,
        image_available=False, mask_available=False, physical_dose_available=False,
        exposure_time_available=False, shared_assay_cell_context_available=False,
        molecular_generation_implemented=False, call_role='observed_target' if forward else 'desired_source_call_pattern',
        source_raw_readback_performed=False)
    ib, tb, pb = _canonical(conditions), _canonical(targets), _canonical(provenance)
    return Tox21View(direction, tuple(values), tuple(mask), ib, tb, pb, _binding(ib, tb, pb))


def read_tox21_views(manifest, *, root, expected_manifest_sha256, direction, split, limit):
    """Bounded hash-verified master-corpus reader; never select an evaluation candidate table."""
    import pyarrow.parquet as pq
    if type(limit) is not int or not 1 <= limit <= 256:
        raise ValueError('explicit inspection limit 1..256 required')
    if direction not in (FORWARD, INVERSE) or split not in ('train', 'validation', 'test'):
        raise ValueError('unsupported direction/split')
    root = Path(root).resolve()
    path = (root / manifest).resolve()
    if not path.is_relative_to(root):
        raise ValueError('manifest outside project')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_manifest_sha256:
        raise ValueError('manifest hash mismatch')
    m = json.loads(raw)
    views, seen = [], set()
    for shard in m['shards']:
        if shard.get('source_family') != SOURCE or shard.get('split') != split:
            continue
        rows = pq.read_table(io.BytesIO(_verified_bytes(root, shard))).to_pylist()
        sidecar = m['effective_group_lookup'][shard['path']]
        groups = pq.read_table(io.BytesIO(_verified_bytes(root, sidecar))).to_pylist()
        if len(rows) != shard['rows'] or len(groups) != sidecar['rows'] or len(rows) != len(groups):
            raise ValueError('shard/sidecar row count mismatch')
        for row, group in zip(rows, groups):
            if row['task_id'] != group['task_id'] or row['split'] != split:
                raise ValueError('task/group allocation mismatch')
            if row['task_name'] != direction:
                continue
            if row['task_id'] in seen:
                raise ValueError('duplicate task identity')
            seen.add(row['task_id'])
            v = tox21_view(row, effective_groups=json.loads(group['effective_groups_json']))
            provenance = v.provenance()
            provenance.update(manifest_sha256=expected_manifest_sha256, shard_sha256=shard['sha256'],
                effective_group_sidecar_sha256=sidecar['sha256'])
            pb = _canonical(provenance)
            views.append(replace(v, provenance_bytes=pb, binding_sha256=_binding(v.input_bytes, v.target_bytes, pb)))
            if len(views) == limit:
                return tuple(views)
    if not views:
        raise ValueError('no admitted matching Tox21 tasks')
    return tuple(views)


def tox21_call_tensors(views, *, role, device='cpu'):
    """Return numeric placeholders AND mandatory boolean masks; unknown never a negative label.

    Consumers must mask elementwise objectives before reduction. Desired conditions
    are distinct from targets; mixed direction batches are refused.
    """
    import torch
    expected = {'target': FORWARD, 'desired_condition': INVERSE}.get(role)
    if not views or expected is None or any(v.direction != expected or v.axis != ASSAYS for v in views):
        raise ValueError('nonempty role-compatible full-axis batch required')
    values = torch.tensor([[v if m else 0 for v, m in zip(x.values, x.observed_mask)]
                           for x in views], dtype=torch.float32, device=device)
    mask = torch.tensor([x.observed_mask for x in views], dtype=torch.bool, device=device)
    return values, mask
