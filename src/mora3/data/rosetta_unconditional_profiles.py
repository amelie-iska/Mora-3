"""Original target-only Rosetta profile arrays and exact finite assay axes."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from .array_tasks import ArrayTask
from .finite_union_native_rows import FiniteUnionNativeView


_TASK = 'unconditional_source_profile_modeling'
_SCHEMA = 'mora3.ArrayTask/v1'
_FAMILIES = {'Rosetta-CellPainting': 'CellPainting', 'Rosetta-L1000': 'L1000'}
_DIMENSIONS = {'CellPainting': 1783, 'L1000': 978}


def _is_sha256(value):
    return type(value) is str and re.fullmatch('[a-f0-9]{64}', value) is not None


def _json_sha256(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'),
                     ensure_ascii=False, allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class RosettaProfileAxis:
    modality: str
    axis_identity: str
    axis_ids: tuple[str, ...]
    units: str

    def __post_init__(self):
        if (
            self.modality not in _DIMENSIONS or type(self.axis_identity) is not str or not self.axis_identity
            or type(self.units) is not str or not self.units
            or type(self.axis_ids) is not tuple or not self.axis_ids
            or any(type(name) is not str or not name for name in self.axis_ids)
            or len(set(self.axis_ids)) != len(self.axis_ids)
        ):
            raise ValueError('Complete ordered native Rosetta profile axis and units required')

    def binding(self):
        return dict(modality=self.modality, axis_identity=self.axis_identity,
                    axis_ids=self.axis_ids, units=self.units)

    @property
    def sha256(self):
        return _json_sha256(self.binding())


@dataclass(frozen=True)
class RosettaProfileTarget:
    axis: RosettaProfileAxis
    values: tuple[float, ...]
    source_coordinate: tuple[str, int, str]


def rosetta_view_policy_sha256(view):
    if type(view) is not FiniteUnionNativeView:
        raise ValueError('Exact original native Rosetta source view required')
    return _json_sha256([view.manifest, getattr(view, 'effective_filter', None)])


def load_rosetta_profile_axes(root, census_path, census_sha256, view):
    """Read all four exact axes from a SHA-pinned complete retained-row census."""
    rosetta_view_policy_sha256(view)
    root = Path(root)
    raw = (root / census_path).read_bytes()
    if not _is_sha256(census_sha256) or hashlib.sha256(raw).hexdigest() != census_sha256:
        raise ValueError('Unconditional Rosetta source census checksum mismatch')
    census = json.loads(raw)
    if (
        census['schema'] != 'Mora3.UnconditionalRosettaArrayCensus/v1'
        or census['task_name'] != _TASK or census['source_target_only'] is not True
        or census['source_context_is_audit_only'] is not True
        or set(census['route_results']) != set(_FAMILIES)
    ):
        raise ValueError('Complete unconditional source-array census required')
    filter_ref = census['effective_filter']
    if type(filter_ref) is not dict or set(filter_ref) != {'path', 'sha256'}:
        raise ValueError('Exact Rosetta effective filter pin required')
    filter_raw = (root / filter_ref['path']).read_bytes()
    if (hashlib.sha256(filter_raw).hexdigest() != filter_ref['sha256']
            or _json_sha256(json.loads(filter_raw)) != _json_sha256(getattr(view, 'effective_filter', None))):
        raise ValueError('Rosetta census is bound to a different effective source filter')
    result = []
    for family, modality in _FAMILIES.items():
        route = census['route_results'][family]
        entries = route['axis_specs']
        if len(entries) != 2 or sum(row['rows'] for row in entries) != route['rows']:
            raise ValueError('Both complete source-study axes required per Rosetta modality')
        for row in entries:
            axis = RosettaProfileAxis(row['modality'], row['axis_identity'], tuple(row['axis_ids']), row['units'])
            if (axis.sha256 != row['sha256'] or axis.modality != modality
                    or len(axis.axis_ids) != _DIMENSIONS[modality]
                    or type(row['rows']) is not int or row['rows'] < 1):
                raise ValueError('Unconditional Rosetta finite axis identity or dimension differs')
            result.append(axis)
    if len({axis.sha256 for axis in result}) != len(result):
        raise ValueError('Duplicate unconditional Rosetta axis')
    return tuple(sorted(result, key=lambda axis: axis.sha256))


def _project(row, path, ordinal, axes):
    family = row['source_family']
    if family not in _FAMILIES or (row['task_name'], row['native_schema'], row['split']) != (_TASK, _SCHEMA, 'train'):
        raise ValueError('Exact retained original Rosetta profile train route required')
    raw = row['native_payload']
    digest = hashlib.sha256(raw).hexdigest()
    if digest != row['native_payload_sha256']:
        raise ValueError('Native Rosetta payload checksum mismatch')
    task = ArrayTask.from_bytes(raw)
    if task.roles != (('profile', 'target'),) or task.source_sha256 != row['source_sha256']:
        raise ValueError('Exact source-bound target-only Rosetta array required')
    array, = task.arrays
    if array.status != 'observed' or len(array.axis_ids) != row['full_axis_values']:
        raise ValueError('Complete observed original Rosetta values required')
    axis = RosettaProfileAxis(_FAMILIES[family], array.axis_identity, array.axis_ids, array.units)
    if axes.get(axis.sha256) != axis:
        raise ValueError('Rosetta target axis outside the bound ordered source axes')
    return RosettaProfileTarget(axes[axis.sha256], array.values, (path, ordinal, digest))


def rosetta_profile_targets(view, batch, axes, *, split):
    """Authenticate every original row. Keep intervention and audit data out of the model."""
    if type(view) is not FiniteUnionNativeView or split != 'train':
        raise ValueError('Exact original native Rosetta train role required')
    view.require_native_rows(batch, split=split)
    if not axes or any(type(axis) is not RosettaProfileAxis for axis in axes):
        raise ValueError('Exact finite Rosetta profile axes required')
    axes_by_digest = {axis.sha256: axis for axis in axes}
    if len(axes_by_digest) != len(axes):
        raise ValueError('Duplicate finite Rosetta profile axes')
    try:
        return tuple(_project(row, batch.source_shard['path'], ordinal, axes_by_digest)
                     for row, ordinal in zip(batch.native_rows.to_pylist(), batch.source_row_ordinals))
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError('Malformed original Rosetta profile array') from exc
