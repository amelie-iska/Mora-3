"""Current finite Tox21 native roles and complete desired-call patterns."""

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import re

from .admet import canonical
from .finite_union_native_rows import FiniteUnionNativeView, read_exact
from .tox21 import ASSAYS, FORWARD, INVERSE, SOURCE, tox21_view


_SPLITS = ('train', 'validation', 'test')


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest(value):
    return type(value) is str and re.fullmatch('[a-f0-9]{64}', value) is not None


@dataclass(frozen=True)
class Tox21DesiredCallPattern:
    values: tuple[int | None, ...]
    observed: tuple[bool, ...]

    def __post_init__(self):
        if (
            type(self.values) is not tuple or type(self.observed) is not tuple
            or len(self.values) != len(ASSAYS) or len(self.observed) != len(ASSAYS)
            or any(type(mask) is not bool for mask in self.observed) or not any(self.observed)
            or any((type(value) is not int or value not in (0, 1)) if mask else value is not None
                   for value, mask in zip(self.values, self.observed))
        ):
            raise ValueError('Complete native 12-axis Tox21 pattern with observed binary and unknown None required')

    @classmethod
    def from_inputs(cls, value):
        if (
            type(value) is not dict or set(value) != {'known_assay_axis', 'desired_source_call_values', 'desired_source_call_mask'}
            or any(type(value[field]) not in (list, tuple) for field in value)
            or tuple(value['known_assay_axis']) != ASSAYS
        ):
            raise ValueError('Exact target-free Tox21 desired assay axis and mask required')
        return cls(tuple(value['desired_source_call_values']), tuple(value['desired_source_call_mask']))

    def inputs(self):
        self.__post_init__()
        return dict(known_assay_axis=list(ASSAYS), desired_source_call_values=list(self.values),
                    desired_source_call_mask=list(self.observed))


@dataclass(frozen=True)
class Tox21CurrentAxes:
    source_census_sha256: str
    native_view_policy_sha256: str
    original_role_counts: tuple[int, int, int]
    max_full_smiles_bytes: int

    def __post_init__(self):
        if (
            not _digest(self.source_census_sha256) or not _digest(self.native_view_policy_sha256)
            or type(self.original_role_counts) is not tuple or len(self.original_role_counts) != len(_SPLITS)
            or any(type(count) is not int or count < 1 for count in self.original_role_counts)
            or type(self.max_full_smiles_bytes) is not int or self.max_full_smiles_bytes < 1
        ):
            raise ValueError('Exact complete native Tox21 original roles, source and target bounds required')

    def binding(self):
        self.__post_init__()
        return dict(
            schema='Mora3.Tox21CurrentNativeAxes/v1', source_census_sha256=self.source_census_sha256,
            native_view_policy_sha256=self.native_view_policy_sha256, full_original_assays=ASSAYS,
            original_role_counts_per_direction=dict(zip(_SPLITS, self.original_role_counts)),
            max_full_smiles_bytes=self.max_full_smiles_bytes,
            observed_call_states='observed native active 1 or inactive 0, separate unknown None mask',
        )


def _view_policy(view):
    if type(view) is not FiniteUnionNativeView:
        raise ValueError('Exact current finite native Tox21 view required')
    return _sha(canonical([view.manifest, getattr(view, 'effective_filter', None)]))


def _read_pin(root, reference):
    path = (root / reference['path']).resolve()
    path.relative_to(root)
    raw = path.read_bytes()
    if not _digest(reference['sha256']) or _sha(raw) != reference['sha256']:
        raise ValueError('Pinned current Tox21 source evidence checksum differs')
    return raw


class Tox21CurrentSource:
    """Bind all current originals through the established Tox21 source reader."""

    def __init__(self, view, census_path, expected_sha256):
        import pyarrow.parquet as pq

        policy = _view_policy(view)
        root = view.root
        proof = json.loads(_read_pin(root, dict(path=str(census_path), sha256=expected_sha256)))
        if (
            proof['schema'] != 'Mora3.CurrentNativeTox21CallAndMoleculeCensus/v1'
            or tuple(proof['full_original_assay_axis']) != ASSAYS
            or proof['same_original_roles_calls_masks_and_full_smiles_across_directions'] is not True
            or proof['unknown_never_inactive'] is not True
            or proof['original_role_contexts_preserved'] is not True
            or json.loads(_read_pin(root, proof['effective_filter'])) != view.effective_filter
        ):
            raise ValueError('Exact complete current native Tox21 role/call census required')
        counts = {(row['split'], row['direction']): row['rows'] for row in proof['original_role_direction_counts']}
        if (
            set(counts) != {(split, direction) for split in _SPLITS for direction in (FORWARD, INVERSE)}
            or any(counts[(split, FORWARD)] != counts[(split, INVERSE)] for split in _SPLITS)
            or sum(counts.values()) != proof['all_original_taskviews']
            or 2 * proof['exact_mirrored_source_molecule_pairs'] != proof['all_original_taskviews']
        ):
            raise ValueError('Tox21 forward and inverse original role counts do not close')
        self.axes = Tox21CurrentAxes(
            expected_sha256, policy, tuple(counts[(split, FORWARD)] for split in _SPLITS),
            proof['max_full_native_smiles_UTF8_bytes'],
        )
        self.source_paths = tuple(proof['original_source_paths'])
        descriptors = {d['source_shard']['path']: d for d in view.descriptors if any(
            row['source_family'] == SOURCE and row['eligible_view_disposition'] in (0, 1, 2)
            for row in d['source_task_schema_counts'])}
        if not self.source_paths or len(set(self.source_paths)) != len(self.source_paths) or set(self.source_paths) != set(descriptors):
            raise ValueError('Exact complete current Tox21 source shard paths required')
        groups = {}
        for path, descriptor in descriptors.items():
            reference = descriptor['original_group_sidecar']
            table = pq.read_table(io.BytesIO(read_exact(reference, root=root)), use_threads=False)
            if len(table) != reference['rows'] or len(table) != descriptor['source_shard']['rows']:
                raise ValueError('Current Tox21 original source/group role allocation differs')
            groups[path] = tuple((row['task_id'], tuple(json.loads(row['effective_groups_json'])))
                                 for row in table.to_pylist())
        self._groups = groups
        self._view = view

    def views(self, batch, *, split, direction):
        if split not in _SPLITS or direction not in (FORWARD, INVERSE) or _view_policy(self._view) != self.axes.native_view_policy_sha256:
            raise ValueError('Exact original Tox21 role, source and direction required')
        self._view.require_native_rows(batch, split=split)
        path = batch.source_shard['path']
        if path not in self._groups:
            raise ValueError('Original Tox21 batch path outside the complete source census')
        result = []
        for ordinal, row in zip(batch.source_row_ordinals, batch.native_rows.to_pylist()):
            if row['task_name'] != direction:
                continue
            task_id, groups = self._groups[path][ordinal]
            if row['task_id'] != task_id:
                raise ValueError('Original Tox21 group/source coordinate changed')
            item = tox21_view(row, effective_groups=groups)
            molecule = item.inputs() if direction == FORWARD else item.targets()
            if len(molecule['canonical_isomeric_smiles'].encode('utf-8')) > self.axes.max_full_smiles_bytes:
                raise ValueError('Complete original native Tox21 SMILES exceeds bound source bytes')
            result.append(item)
        if not result:
            raise ValueError('Native batch has no selected original Tox21 direction')
        return tuple(result)
