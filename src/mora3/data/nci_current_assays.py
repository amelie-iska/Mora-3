"""Typed NCI task views over the exact current retained source coordinates."""

from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re

from .finite_union_native_rows import FiniteUnionNativeView
from .nci_aids import CLASS_AXIS, CONTEXT, EVIDENCE, FAMILIES, MISSING, PAIRED_AXIS, PANEL, POLICY, READOUT, REVISION, SOURCE_REF
from .uniprot_annotations import _canonical


CONCLUSION = 'NCI-AIDS-source-conclusion-prior'
ENDPOINT = 'NCI-AIDS-source-endpoint-prior'
PAIRED = 'NCI-AIDS-paired-assay-profile-prior'
SMILES = 'molecule_source_connectivity_smiles'
_DIRECTIONS = ('prediction', 'generation')
_SPLITS = ('train', 'validation', 'test')


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


@dataclass(frozen=True)
class NciAssayPartition:
    family: str
    readout: str | None
    unit: str | None

    def __post_init__(self):
        if self.family == CONCLUSION and self.readout is None and self.unit is None:
            return
        if self.family == ENDPOINT and self.readout in READOUT.values() and self.unit in ('M', 'u'):
            return
        if self.family == PAIRED and self.readout is None and self.unit in ('M', 'u'):
            return
        raise ValueError('Exact NCI conclusion, single readout or paired source unit partition required')

    @property
    def output_field(self):
        return {
            CONCLUSION: 'source_conclusion',
            ENDPOINT: 'source_log10_endpoint_mean',
            PAIRED: 'source_log10_endpoint_means',
        }[self.family]

    @property
    def coordinate_axis(self):
        return {CONCLUSION: CLASS_AXIS, ENDPOINT: (self.output_field,), PAIRED: PAIRED_AXIS}[self.family]

    def binding(self):
        return self.family, PANEL, self.readout, self.unit, self.coordinate_axis

    @property
    def identity(self):
        return _sha(_canonical(self.binding()))


def _all_partitions():
    partitions = [NciAssayPartition(CONCLUSION, None, None)]
    partitions.extend(NciAssayPartition(ENDPOINT, readout, unit) for readout in READOUT.values() for unit in ('M', 'u'))
    partitions.extend(NciAssayPartition(PAIRED, None, unit) for unit in ('M', 'u'))
    return tuple(sorted(partitions, key=lambda value: value.identity))


@dataclass(frozen=True)
class NciAssaySourceAxes:
    source_census_sha256: str
    native_view_policy_sha256: str
    partitions: tuple[NciAssayPartition, ...]
    max_target_utf8_bytes: int
    max_condition_utf8_bytes: int

    def __post_init__(self):
        if (
            not _digest(self.source_census_sha256) or not _digest(self.native_view_policy_sha256)
            or type(self.partitions) is not tuple or self.partitions != _all_partitions()
            or any(type(n) is not int or n < 1 for n in (self.max_target_utf8_bytes, self.max_condition_utf8_bytes))
        ):
            raise ValueError('Complete pinned current NCI source assay axes and byte bounds required')

    def binding(self):
        self.__post_init__()
        return dict(
            schema='Mora3.NciCurrentAssayAxes/v1', source_census_sha256=self.source_census_sha256,
            native_view_policy_sha256=self.native_view_policy_sha256,
            partitions=tuple(partition.binding() for partition in self.partitions),
            max_target_utf8_bytes=self.max_target_utf8_bytes, max_condition_utf8_bytes=self.max_condition_utf8_bytes,
        )


@dataclass(frozen=True)
class NciCurrentAssayTask:
    partition: NciAssayPartition
    direction: str
    original_split: str
    condition_bytes: bytes
    target: str | bytes | tuple[Decimal, ...]
    source_coordinate: tuple[str, int, str]

    def inputs(self):
        return json.loads(self.condition_bytes)


def _numbers(value, *, multiple):
    values = value if multiple else [value]
    if (
        type(values) is not list or len(values) != (len(PAIRED_AXIS) if multiple else 1)
        or any(type(x) not in (int, float, Decimal) or not math.isfinite(x) for x in values)
    ):
        raise ValueError('Complete finite observed native NCI log10 source values required')
    return values


def _smiles(value):
    if type(value) is not str or not value or not value.isascii():
        raise ValueError('Complete source native connectivity SMILES text required')
    return value.encode('utf-8')


def nci_known_condition(partition, direction, condition):
    """Canonical bytes of the exact source-known condition role."""
    if type(partition) is not NciAssayPartition or direction not in _DIRECTIONS or type(condition) is not dict:
        raise ValueError('Exact NCI source partition and condition direction required')
    partition.__post_init__()
    desired = 'desired_' + partition.output_field
    keys = {'assay_panel', 'source_stereochemistry_known', SMILES if direction == 'prediction' else desired}
    if partition.family != CONCLUSION:
        keys |= {'concentration_unit', 'source_log10_highest_tested_concentration'}
        keys.add('assay_readout' if partition.family == ENDPOINT else 'endpoint_axis')
    if set(condition) != keys or condition['assay_panel'] != PANEL or condition['source_stereochemistry_known'] is not False:
        raise ValueError('Exact native known NCI fields required, with source stereo unknown')
    if partition.family != CONCLUSION:
        if condition['concentration_unit'] != partition.unit:
            raise ValueError('NCI native units differ from the declared partition')
        _numbers(condition['source_log10_highest_tested_concentration'], multiple=False)
        if partition.family == ENDPOINT and condition['assay_readout'] != partition.readout:
            raise ValueError('Native NCI assay readout differs from the declared partition')
        if partition.family == PAIRED and (type(condition['endpoint_axis']) is not list
                                          or tuple(condition['endpoint_axis']) != PAIRED_AXIS):
            raise ValueError('Complete ordered separate-assay source axis required')
    if direction == 'prediction':
        _smiles(condition[SMILES])
    elif partition.family == CONCLUSION:
        if condition[desired] not in CLASS_AXIS:
            raise ValueError('Known desired NCI source conclusion outside its literal axis')
    else:
        _numbers(condition[desired], multiple=partition.family == PAIRED)
    return _canonical(condition)


def _source_policy(view):
    if type(view) is not FiniteUnionNativeView:
        raise ValueError('Exact finite original NCI native view required')
    return _sha(_canonical([view.manifest, getattr(view, 'effective_filter', None)]))


def _project(row, path, ordinal):
    family = row['source_family']
    directions = [d for d in _DIRECTIONS if family in FAMILIES and row['task_name'] == FAMILIES[family] + d + '_prior']
    raw = row['native_payload']
    digest = _sha(raw)
    if (
        len(directions) != 1 or digest != row['native_payload_sha256']
        or row['native_schema'] != 'mora3.CorpusTask/v1' or row['source_revision'] != REVISION
        or row['source_sha256'] != SOURCE_REF['sha256'] or row['admitted_for_task'] is not True
        or row['evidence_class'] != EVIDENCE or row['split_policy'] != POLICY
        or row['split'] not in _SPLITS or json.loads(row['condition_context_json']) != CONTEXT
    ):
        raise ValueError('Exact original current NCI source task metadata required')
    direction, = directions
    task = json.loads(raw)
    if (
        set(task) != {'schema', 'task', 'conditions', 'targets', 'audit', 'missing'}
        or task['schema'] != row['native_schema'] or task['task'] != row['task_name'] or task['missing'] != MISSING
        or _canonical(task) != raw or task['audit']['source_ref']['native_source'] != SOURCE_REF
        or task['audit']['future_generated_outputs_require_model_produced_provenance'] is not True
    ):
        raise ValueError('Exact immutable source NCI native task payload required')
    inputs, targets, audit = task['conditions'], task['targets'], task['audit']
    partition = NciAssayPartition(family, inputs.get('assay_readout'), inputs.get('concentration_unit'))
    condition_bytes = nci_known_condition(partition, direction, inputs)
    if family == ENDPOINT and (audit['source_flag'] != '=' or READOUT[audit['source_endpoint']] != partition.readout):
        raise ValueError('Single endpoint task requires a native uncensored source mean')
    if family == PAIRED and (
        audit['source_censor_flags'] != ['=', '=']
        or any(audit[field] is not False for field in (
            'same_cell_or_replicate_pair', 'source_lot_or_full_protocol_identity_verified',
            'selectivity_ratio_or_therapeutic_index_derived',
        ))
    ):
        raise ValueError('Paired task requires the exact original separate uncensored assay summaries')
    key = SMILES if direction == 'generation' else partition.output_field
    if type(targets) is not dict or set(targets) != {key}:
        raise ValueError('Complete source-native target in its separate role required')
    if direction == 'generation':
        target = _smiles(targets[key])
        dimensions = len(target)
    elif family == CONCLUSION:
        target, dimensions = targets[key], 1
        if type(target) is not str or target not in CLASS_AXIS:
            raise ValueError('Source NCI conclusion outside its literal class axis')
    else:
        literal = json.loads(raw, parse_float=Decimal)['targets'][key]
        target = tuple(Decimal(value) for value in _numbers(literal, multiple=family == PAIRED))
        dimensions = len(target)
    if row['full_axis_values'] != dimensions:
        raise ValueError('Complete native source NCI target dimensions differ')
    return NciCurrentAssayTask(partition, direction, row['split'], condition_bytes, target, (path, ordinal, digest))


class NciCurrentAssaySource:
    """Bind exact retained native shards and original roles to immutable assay axes."""

    def __init__(self, view, census_path, expected_sha256):
        view_sha = _source_policy(view)
        root = view.root
        path = (root / Path(census_path)).resolve()
        path.relative_to(root)
        raw = path.read_bytes()
        if not _digest(expected_sha256) or _sha(raw) != expected_sha256:
            raise ValueError('Pinned current NCI source census checksum required')
        proof = json.loads(raw)
        if (
            proof['schema'] != 'Mora3.CompleteCurrentNciAssayAxisCensus/v1'
            or proof['native_source_manifest'] != SOURCE_REF
            or proof['exact_prediction_generation_role_condition_target_pair_agreement'] is not True
            or proof['all_scalar_endpoints_explicit_uncensored_original_means'] is not True
            or proof['paired_distinct_assays_no_selectivity_or_same_cell_claim'] is not True
            or tuple(proof['class_axis']) != CLASS_AXIS or tuple(proof['paired_axis']) != PAIRED_AXIS
        ):
            raise ValueError('Complete native source NCI assay semantics and paired route evidence required')
        filter_ref = proof['effective_filter']
        filter_path = (root / filter_ref['path']).resolve()
        filter_path.relative_to(root)
        filter_raw = filter_path.read_bytes()
        if _sha(filter_raw) != filter_ref['sha256'] or json.loads(filter_raw) != view.effective_filter:
            raise ValueError('Pinned NCI census and current exact native filter differ')
        partitions = []
        total_rows = 0
        for entry in proof['native_partition_counts']:
            family, panel, readout, unit, axis = entry['partition']
            partition = NciAssayPartition(family, readout, unit)
            if panel != PANEL or tuple(axis) != partition.coordinate_axis or {
                r['direction'] for r in entry['rows'] if r['split'] == 'train' and r['rows'] > 0
            } != set(_DIRECTIONS):
                raise ValueError('Each exact native NCI partition requires complete declared source training observations')
            partitions.append(partition)
            total_rows += sum(row['rows'] for row in entry['rows'])
        if total_rows != proof['complete_original_taskviews'] or total_rows != 2 * proof['forward_inverse_record_pairs']:
            raise ValueError('Current source NCI partition counts do not close')
        self.axes = NciAssaySourceAxes(
            expected_sha256, view_sha, tuple(sorted(partitions, key=lambda value: value.identity)),
            proof['complete_native_connectivity_smiles_bytes_max'], proof['canonical_known_condition_bytes_max'],
        )
        self.source_paths = tuple(proof['original_source_shards'])
        if not self.source_paths or len(set(self.source_paths)) != len(self.source_paths):
            raise ValueError('Exact distinct source NCI shard paths required')
        self._view = view

    def tasks(self, batch, *, split, direction):
        if split not in _SPLITS or direction not in _DIRECTIONS or _source_policy(self._view) != self.axes.native_view_policy_sha256:
            raise ValueError('Exact bound current native NCI direction and role required')
        self._view.require_native_rows(batch, split=split)
        path = batch.source_shard['path']
        if path not in self.source_paths:
            raise ValueError('Original row path outside the current native NCI census')
        tasks = []
        try:
            for row, ordinal in zip(batch.native_rows.to_pylist(), batch.source_row_ordinals):
                task = _project(row, path, ordinal)
                if task.direction != direction:
                    continue
                if (
                    len(task.condition_bytes) > self.axes.max_condition_utf8_bytes
                    or task.direction == 'generation' and len(task.target) > self.axes.max_target_utf8_bytes
                ):
                    raise ValueError('Complete source NCI condition or target exceeds the bound census')
                tasks.append(task)
        except (KeyError, TypeError, OverflowError) as exc:
            raise ValueError('Malformed original native NCI source task') from exc
        if not tasks:
            raise ValueError('No selected native NCI direction in the bound source batch')
        return tuple(tasks)
