"""Exact current enzyme-product task views with unresolved source lineage held out."""

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re

from .admet import canonical
from .finite_union_native_rows import FiniteUnionNativeView


FAMILY = 'ReactZyme'
TASK = 'enzyme_annotated_reaction_product'
SCHEMA = 'mora3.CorpusTask/v1'
REVISION = 'Zenodo11494913+UniProt2026_03'
FIELDS = ('full_amino_acid_sequence_UTF8', 'full_native_substrate_UTF8', 'full_native_product_UTF8')
ROLES = ('train', 'validation', 'test')


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest(value):
    return type(value) is str and re.fullmatch('[a-f0-9]{64}', value) is not None


@dataclass(frozen=True)
class ReactZymeCondition:
    amino_acid_sequence: str
    substrate_smiles: str

    def __post_init__(self):
        if any(type(value) is not str or not value or not value.isascii()
               for value in (self.amino_acid_sequence, self.substrate_smiles)):
            raise ValueError('Complete original native enzyme sequence and substrate strings required')


@dataclass(frozen=True)
class ReactZymeSourceAxes:
    source_census_sha256: str
    native_view_policy_sha256: str
    max_source_bytes: tuple[int, int, int]
    native_alphabets: tuple[str, str, str]
    guard_clear_role_counts: tuple[int, int, int]

    def __post_init__(self):
        if (
            not _digest(self.source_census_sha256) or not _digest(self.native_view_policy_sha256)
            or type(self.max_source_bytes) is not tuple or len(self.max_source_bytes) != len(FIELDS)
            or any(type(size) is not int or size < 1 for size in self.max_source_bytes)
            or type(self.native_alphabets) is not tuple or len(self.native_alphabets) != len(FIELDS)
            or any(type(chars) is not str or not chars or not chars.isascii()
                   or chars != ''.join(sorted(set(chars))) for chars in self.native_alphabets)
            or type(self.guard_clear_role_counts) is not tuple or len(self.guard_clear_role_counts) != len(ROLES)
            or any(type(count) is not int or count < 0 for count in self.guard_clear_role_counts)
            or self.guard_clear_role_counts[0] < 1 or self.guard_clear_role_counts[1:] != (0, 0)
        ):
            raise ValueError('Complete source-bounded ReactZyme train-only native axes required')

    def binding(self):
        self.__post_init__()
        return dict(
            schema='Mora3.ReactZymeNativeSourceAxes/v1', source_census_sha256=self.source_census_sha256,
            native_view_policy_sha256=self.native_view_policy_sha256,
            max_source_bytes=dict(zip(FIELDS, self.max_source_bytes)),
            native_alphabets=dict(zip(FIELDS, self.native_alphabets)),
            guard_clear_role_counts=dict(zip(ROLES, self.guard_clear_role_counts)),
        )

    def require_condition(self, condition):
        if type(condition) is not ReactZymeCondition:
            raise ValueError('Exact target-free native ReactZyme condition required')
        condition.__post_init__()
        for value, bound, alphabet in zip(
            (condition.amino_acid_sequence, condition.substrate_smiles), self.max_source_bytes, self.native_alphabets,
        ):
            if len(value.encode('utf-8')) > bound or set(value) - set(alphabet):
                raise ValueError('Native ReactZyme condition outside complete source byte/alphabet bounds')


@dataclass(frozen=True)
class ReactZymeNativeTask:
    condition: ReactZymeCondition
    product_bytes: bytes
    source_coordinate: tuple[str, int, str]


def _view_policy(view):
    if type(view) is not FiniteUnionNativeView:
        raise ValueError('Exact current original ReactZyme native view required')
    return _sha(canonical([view.manifest, getattr(view, 'effective_filter', None)]))


def _read_exact(root, reference):
    path = (root / reference['path']).resolve()
    path.relative_to(root)
    raw = path.read_bytes()
    if not _digest(reference['sha256']) or _sha(raw) != reference['sha256']:
        raise ValueError('Exact pinned ReactZyme source evidence checksum required')
    return raw


def _project(row, path, ordinal, axes):
    raw = row['native_payload']
    digest = _sha(raw)
    if (
        (row['source_family'], row['task_name'], row['native_schema'], row['source_revision'], row['split'])
        != (FAMILY, TASK, SCHEMA, REVISION, 'train')
        or row['admitted_for_task'] is not True or digest != row['native_payload_sha256']
        or json.loads(row['condition_context_json']) != {}
    ):
        raise ValueError('Exact original ReactZyme train task row required')
    task = json.loads(raw)
    if (
        set(task) != {'schema', 'task', 'conditions', 'targets', 'audit', 'missing'}
        or task['schema'] != SCHEMA or task['task'] != TASK or canonical(task) != raw
        or set(task['conditions']) != {'amino_acid_sequence', 'substrate_smiles'}
        or set(task['targets']) != {'product_smiles'}
        or set(task['audit']) != {'annotation_score', 'ec_number', 'entry', 'rhea_id', 'source_date'}
    ):
        raise ValueError('Exact source native ReactZyme condition and product roles required')
    condition = ReactZymeCondition(**task['conditions'])
    axes.require_condition(condition)
    product = task['targets']['product_smiles']
    if (
        type(product) is not str or not product or not product.isascii()
        or len(product.encode('utf-8')) > axes.max_source_bytes[2] or set(product) - set(axes.native_alphabets[2])
    ):
        raise ValueError('Complete native ReactZyme product outside its source-bound alphabet/bytes')
    return ReactZymeNativeTask(condition, product.encode('utf-8'), (path, ordinal, digest))


class ReactZymeCurrentSource:
    """Current source view with an additional prospective model-only unresolved guard."""

    def __init__(self, view, census_path, expected_sha256):
        policy = _view_policy(view)
        root = view.root
        proof = json.loads(_read_exact(root, dict(path=str(census_path), sha256=expected_sha256)))
        if (
            proof['schema'] != 'Mora3.CurrentReactZymeExactGuardedNativeRouteCensus/v1'
            or proof['direct_v2_original_exclusions_verified'] != 22
            or proof['all_unresolved_still_preserved_and_blocked_only_in_prospective_model_route'] is not True
            or proof['same_direction_physical_or_experimental_identity_assumed'] is not False
            or json.loads(_read_exact(root, proof['effective_filter'])) != view.effective_filter
        ):
            raise ValueError('Exact current native ReactZyme source/exclusion/lineage census required')
        trace = json.loads(_read_exact(root, proof['direct_study_trace']))
        lineage = json.loads(_read_exact(root, proof['unresolved_original_lineage_manifest']))
        lineage_rows = json.loads(_read_exact(root, proof['unresolved_original_lineage_rows']))
        if (
            lineage['parent_task_study_trace']['sha256'] != proof['direct_study_trace']['sha256']
            or lineage['per_record_lineage']['sha256'] != proof['unresolved_original_lineage_rows']['sha256']
            or lineage['unproved_historical_association_remains_explicit'] is not True
            or trace['family_counts'][FAMILY]['unresolved_target_or_label_scope_taskviews'] != 16
            or any(row['same_direction_or_physical_experiment_identity_proven'] is not False for row in lineage_rows)
        ):
            raise ValueError('Exact unproved historical ReactZyme label provenance required')
        unresolved = {
            (row['source_task']['source_shard']['path'], row['source_task']['absolute_original_row_ordinal']):
            row['source_task']['native_payload_sha256']
            for row in lineage_rows
        }
        expected_unresolved = {(row['path'], row['ordinal']): row['native_payload_sha256']
                               for row in proof['unresolved_original_coordinates']}
        if len(unresolved) != 16 or unresolved != expected_unresolved:
            raise ValueError('Exact all sixteen unresolved original ReactZyme coordinates required')
        clear = tuple(proof['guard_clear_prospective_original_roles'][role] for role in ROLES)
        if any(proof['current_retained_roles'][role] != clear[i] + proof['unresolved_current_original_roles'][role]
               for i, role in enumerate(ROLES)):
            raise ValueError('Original retained and unresolved ReactZyme roles do not close')
        self.axes = ReactZymeSourceAxes(
            expected_sha256, policy,
            tuple(proof['guard_clear_complete_max_bytes'][name] for name in FIELDS),
            tuple(proof['exact_source_character_alphabets'][name] for name in FIELDS), clear,
        )
        self.source_paths = tuple(proof['original_source_paths'])
        if not self.source_paths or len(set(self.source_paths)) != len(self.source_paths):
            raise ValueError('Exact distinct original ReactZyme source shard paths required')
        self._unresolved = unresolved
        self._view = view

    def _require_view(self):
        if _view_policy(self._view) != self.axes.native_view_policy_sha256:
            raise ValueError('Bound current ReactZyme native source view changed')

    def iter_guard_clear_batches(self, *, batch_size):
        """Yield only guard-clear original train rows; retain all field values and ordinals."""
        import pyarrow as pa

        self._require_view()
        for batch in self._view.iter_native_task_batches(
            split='train', batch_size=batch_size, source_paths=self.source_paths,
        ):
            path = batch.source_shard['path']
            kept = [i for i, ordinal in enumerate(batch.source_row_ordinals)
                    if (path, ordinal) not in self._unresolved]
            if not kept:
                continue
            if len(kept) == len(batch.source_row_ordinals):
                yield batch
            else:
                yield replace(
                    batch, source_row_ordinals=tuple(batch.source_row_ordinals[i] for i in kept),
                    native_rows=batch.native_rows.take(pa.array(kept, type=pa.int64())),
                )

    def tasks(self, batch, *, split='train'):
        self._require_view()
        if split != 'train':
            raise ValueError('Original retained ReactZyme data has no validation/test roles')
        self._view.require_native_rows(batch, split=split)
        path = batch.source_shard['path']
        if path not in self.source_paths:
            raise ValueError('Original ReactZyme row path outside the pinned source census')
        if any((path, ordinal) in self._unresolved for ordinal in batch.source_row_ordinals):
            raise ValueError('Unresolved original ReactZyme source lineage is audit-only')
        try:
            return tuple(_project(row, path, ordinal, self.axes)
                         for row, ordinal in zip(batch.native_rows.to_pylist(), batch.source_row_ordinals))
        except (KeyError, TypeError, OverflowError) as exc:
            raise ValueError('Malformed current native ReactZyme source task') from exc
