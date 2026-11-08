"""Exact current ExpansionRx source task views and literal assay partitions."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from .admet import (
    AdmetObservation, AssayContext, EndpointDefinition, EndpointRegistry, MOLECULE_CLASSES,
    NativeEntity, canonical,
)
from .finite_union_native_rows import FiniteUnionNativeView


FAMILY = 'ExpansionRx-source-native-ADMET-readouts'
DIRECTIONS = ('entity_to_measurement', 'measurement_to_entity')
_ENDPOINTS = frozenset('ExpansionRx:' + name for name in (
    'LogD', 'KSOL', 'MLM CLint', 'HLM CLint', 'RLM CLint', 'Caco-2 Permeability Papp A>B',
    'Caco-2 Permeability Efflux', 'MPPB', 'MBPB', 'MGMB',
))
_NOMINAL = frozenset(('ExpansionRx:LogD', 'ExpansionRx:Caco-2 Permeability Efflux'))


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


@dataclass(frozen=True)
class ExpansionRxPartition:
    definition: EndpointDefinition
    context: AssayContext

    def __post_init__(self):
        if (
            type(self.definition) is not EndpointDefinition or type(self.context) is not AssayContext
            or self.definition.key not in _ENDPOINTS or self.definition.objective_kind != 'regression'
            or self.definition.applicability_for('small_molecule') != 'applicable'
            or any(self.definition.applicability_for(k) != 'unknown' for k in MOLECULE_CLASSES if k != 'small_molecule')
        ):
            raise ValueError('Exact native ExpansionRx regression endpoint and source assay partition required')
        if self.nominal:
            if self.definition.unit is not None or self.definition.unit_basis != 'unresolved':
                raise ValueError('Native nominal ExpansionRx coordinates require literal unresolved units')
        elif self.definition.unit_basis != 'official_source_schema' or type(self.definition.unit) is not str:
            raise ValueError('Exact official native ExpansionRx source schema unit required')

    @property
    def nominal(self):
        return self.definition.key in _NOMINAL

    @property
    def identity(self):
        return _sha(canonical(self.binding()))

    def binding(self):
        return dict(endpoint=self.definition.native(), context=self.context.as_mapping(),
                    molecule_class='small_molecule', unit_policy='nominal_source_only' if self.nominal else 'official_schema')


@dataclass(frozen=True)
class ExpansionRxSourceAxes:
    source_census_sha256: str
    native_view_policy_sha256: str
    partitions: tuple[ExpansionRxPartition, ...]
    native_entity_template: bytes
    max_connectivity_bytes: int
    max_original_condition_bytes: int

    def __post_init__(self):
        if (
            not _digest(self.source_census_sha256) or not _digest(self.native_view_policy_sha256)
            or type(self.partitions) is not tuple or len(self.partitions) != len(_ENDPOINTS)
            or any(type(partition) is not ExpansionRxPartition for partition in self.partitions)
            or {partition.definition.key for partition in self.partitions} != _ENDPOINTS
            or self.partitions != tuple(sorted(self.partitions, key=lambda partition: partition.identity))
            or any(type(size) is not int or size < 1 for size in (self.max_connectivity_bytes, self.max_original_condition_bytes))
            or type(self.native_entity_template) is not bytes
        ):
            raise ValueError('Complete native ExpansionRx partition, source and byte axes required')
        template = json.loads(self.native_entity_template)
        if (
            type(template) is not dict or 'representation' in template
            or template.get('molecule_class') != 'small_molecule' or template.get('representation_type') != 'native_SMILES'
            or canonical(template) != self.native_entity_template
        ):
            raise ValueError('Exact complete constant ExpansionRx non-representation entity fields required')
        NativeEntity.from_mapping(dict(template, representation='C'))

    def binding(self):
        self.__post_init__()
        return dict(
            schema='Mora3.ExpansionRxSourceAxes/v1', source_census_sha256=self.source_census_sha256,
            native_view_policy_sha256=self.native_view_policy_sha256,
            partitions=tuple(partition.binding() for partition in self.partitions),
            native_entity_template=json.loads(self.native_entity_template),
            max_connectivity_bytes=self.max_connectivity_bytes,
            max_original_condition_bytes=self.max_original_condition_bytes,
        )

    def registry(self):
        return EndpointRegistry(tuple(partition.definition for partition in self.partitions))


@dataclass(frozen=True)
class ExpansionRxNativeTask:
    partition: ExpansionRxPartition
    direction: str
    split: str
    observation: AdmetObservation
    source_coordinate: tuple[str, int, str]

    def native_connectivity_bytes(self):
        return self.observation.entity.as_mapping()['representation'].encode('utf-8')


def _view_policy(view):
    if type(view) is not FiniteUnionNativeView:
        raise ValueError('Exact finite current original ExpansionRx source view required')
    return _sha(canonical([view.manifest, getattr(view, 'effective_filter', None)]))


def _read_exact(root, reference):
    path = (root / reference['path']).resolve()
    path.relative_to(root)
    raw = path.read_bytes()
    if not _digest(reference['sha256']) or _sha(raw) != reference['sha256']:
        raise ValueError('Exact pinned ExpansionRx source evidence checksum required')
    return raw


def _partition(native):
    if set(native) != {'endpoint', 'assay_context', 'molecule_class'} or native['molecule_class'] != 'small_molecule':
        raise ValueError('Exact current ExpansionRx assay/source class required')
    applicability = {key: 'applicable' if key == 'small_molecule' else 'unknown' for key in MOLECULE_CLASSES}
    definition = EndpointDefinition.from_native(native['endpoint'], objective_kind='regression',
                                                applicability=applicability)
    return ExpansionRxPartition(definition, AssayContext.from_mapping(native['assay_context']))


def _project(row, path, ordinal, axes):
    raw = row['native_payload']
    digest = _sha(raw)
    direction = row['direction']
    split = row['split']
    if (
        row['source_family'] != FAMILY or direction not in DIRECTIONS or split not in ('train', 'test')
        or row['admitted_for_task'] is not True or row['native_schema'] != 'mora3.CorpusTask/v1'
        or row['task_name'] != 'source_native_ADMET_' + direction or digest != row['native_payload_sha256']
    ):
        raise ValueError('Exact retained original ExpansionRx native task/role required')
    task = json.loads(raw)
    if task['schema'] != row['native_schema'] or task['task'] != row['task_name'] or canonical(task) != raw:
        raise ValueError('Exact native ExpansionRx task bytes required')
    audit = task['audit']
    if audit['directions_reuse_one_source_observation'] is not True or audit['evidence_is_not_clinical_or_causal_validation'] is not True:
        raise ValueError('Source-observed task roles and evidence disposition required')
    observation = AdmetObservation.from_native(canonical(audit['source_observation']), applicability='applicable')
    native = observation.native()
    entity = observation.entity.as_mapping()
    cell = observation.cell
    if (
        not observation.experimental_measurement_candidate or observation.evidence_role != 'source_curated_measurement'
        or native['source_split']['native_export'] != 'source_raw'
        or native['source_split']['parent_protocol'] != 'native_' + split
        or entity['molecule_class'] != 'small_molecule'
    ):
        raise ValueError('Exact current native source measurement or censor bound required')
    partition = _partition(dict(endpoint=native['endpoint'], assay_context=native['known_context'],
                                molecule_class=entity['molecule_class']))
    if partition not in axes.partitions:
        raise ValueError('Native ExpansionRx endpoint/context/unit outside the complete source axes')
    base = dict(endpoint=native['endpoint'], assay_context=native['known_context'], evidence_role=observation.evidence_role)
    constraint = cell.constraint_fields()
    if direction == 'entity_to_measurement':
        expected_inputs = dict(base, native_entity=entity)
        expected_targets = dict(native_measurement=constraint)
    else:
        expected_inputs = dict(base, molecule_class='small_molecule', desired_native_measurement=constraint)
        expected_targets = dict(native_entity=entity)
    if task['conditions'] != expected_inputs or task['targets'] != expected_targets:
        raise ValueError('Original ExpansionRx condition and target roles differ')
    template = dict(entity)
    representation = template.pop('representation')
    if (
        canonical(template) != axes.native_entity_template or not representation.isascii()
        or not 1 <= len(representation.encode()) <= axes.max_connectivity_bytes
        or len(canonical(expected_inputs)) > axes.max_original_condition_bytes
    ):
        raise ValueError('Complete native ExpansionRx identity or bound source input differs')
    return ExpansionRxNativeTask(partition, direction, split, observation, (path, ordinal, digest))


class ExpansionRxCurrentSource:
    """Pinned complete current source, with source-task roles kept separate from optimization."""

    def __init__(self, view, census_path, census_sha256):
        policy_sha = _view_policy(view)
        root = view.root
        proof = json.loads(_read_exact(root, dict(path=str(census_path), sha256=census_sha256)))
        if (
            proof['schema'] != 'Mora3.CompleteCurrentExpansionRxNativeAssayCensus/v1'
            or proof['both_directions_exact_same_source_observations'] is not True
            or proof['no_unit_or_context_inference'] is not True
            or proof['complete_original_taskviews'] != 2 * proof['original_forward_inverse_native_axis_cells']
            or json.loads(_read_exact(root, proof['effective_filter'])) != view.effective_filter
        ):
            raise ValueError('Complete unchanged current ExpansionRx native task/role evidence required')
        protocol = json.loads(_read_exact(root, proof['native_source_protocol']))
        independent = json.loads(_read_exact(root, proof['original_source_independent_proof']))
        if (
            protocol['task_views'] != proof['complete_original_taskviews']
            or protocol['source_test_not_globally_blind'] is not True
            or independent['all_original_source_CSV_fields_and_native_axis_values_checked'] is not True
            or independent['all_native_literal_censor_units_context_target_direction_ID_location_groups_checked'] is not True
        ):
            raise ValueError('Complete original ExpansionRx source measurement/role readback required')
        reviews = {
            row['source_endpoint']: row
            for row in protocol['complete_native_endpoint_registry']['endpoint_reviews']
            if row['source_family'] == 'ExpansionRx'
        }
        partitions = []
        for row in proof['native_partition_counts']:
            partition = _partition(row['partition'])
            native = partition.definition.native()
            review = reviews[native['source_endpoint']]
            if (
                review['source_native_coordinate_objective_supported'] is not True
                or review['native_coordinate_only'] is not partition.nominal
                or review['unit_from_official_schema'] != partition.definition.unit
                or {x['direction'] for x in row['rows'] if x['split'] == 'train' and x['rows'] > 0} != set(DIRECTIONS)
            ):
                raise ValueError('Exact source coordinate review and original train directions required')
            partitions.append(partition)
        template, = proof['native_entity_templates']
        if template['rows'] != proof['complete_original_taskviews']:
            raise ValueError('Complete constant source molecule entity fields required')
        self.axes = ExpansionRxSourceAxes(
            census_sha256, policy_sha, tuple(sorted(partitions, key=lambda partition: partition.identity)),
            canonical(template['fields']), proof['max_native_connectivity_bytes'], proof['max_original_condition_bytes'],
        )
        self.source_paths = tuple(proof['original_source_shards'])
        if not self.source_paths or len(set(self.source_paths)) != len(self.source_paths):
            raise ValueError('Exact distinct original source ExpansionRx shard paths required')
        self._view = view

    def tasks(self, batch, *, split, direction):
        if split not in ('train', 'test') or direction not in DIRECTIONS or _view_policy(self._view) != self.axes.native_view_policy_sha256:
            raise ValueError('Exact current source-native ExpansionRx role and direction required')
        self._view.require_native_rows(batch, split=split)
        path = batch.source_shard['path']
        if path not in self.source_paths:
            raise ValueError('Native row path outside the complete current ExpansionRx census')
        try:
            tasks = tuple(_project(row, path, ordinal, self.axes)
                          for row, ordinal in zip(batch.native_rows.to_pylist(), batch.source_row_ordinals)
                          if row['direction'] == direction)
        except (KeyError, TypeError, OverflowError) as exc:
            raise ValueError('Malformed original native ExpansionRx task') from exc
        if not tasks:
            raise ValueError('Exact ExpansionRx batch has no selected native direction')
        return tasks
