"""Typed deposited-coordinate packets authenticated by the current native view."""
from dataclasses import dataclass
import hashlib
import json
import math

from .finite_union_native_rows import FiniteUnionNativeView


FAMILY = 'PDB-nucleic-masked-coordinates'
TASK = 'source_structure_masked_coordinate_reconstruction'
SCHEMA = 'mora3.CorpusTask/v1'
ATOM_FIELDS = (
    'type_symbol', 'label_atom_id', 'label_comp_id', 'label_asym_id', 'label_seq_id', 'label_alt_id',
)


@dataclass(frozen=True)
class PDBCoordinateCondition:
    atom_labels: tuple[tuple[str, ...], ...]
    positions_angstrom: tuple[tuple[float, ...] | None, ...]


@dataclass(frozen=True)
class PDBCoordinatePacket:
    condition: PDBCoordinateCondition
    masked_targets_angstrom: tuple[tuple[float, ...] | None, ...]
    source_coordinate: tuple[str, int, str]


@dataclass(frozen=True)
class PDBCoordinateAxes:
    atom_label_axes: tuple[tuple[str, ...], ...]
    max_atoms: int
    source_coordinates: tuple[tuple[str, int, str], ...]
    native_view_policy_sha256: str

    def binding(self):
        if (
            type(self.atom_label_axes) is not tuple or len(self.atom_label_axes) != len(ATOM_FIELDS)
            or any(type(a) is not tuple or not a or any(type(x) is not str for x in a)
                   or a != tuple(sorted(set(a))) for a in self.atom_label_axes)
            or not _is_sha256(self.native_view_policy_sha256)
            or type(self.max_atoms) is not int or self.max_atoms < 1
            or not self.source_coordinates
            or len(set(self.source_coordinates)) != len(self.source_coordinates)
            or any(type(path) is not str or not path or type(row) is not int or row < 0
                   or not _is_sha256(sha)
                   for path, row, sha in self.source_coordinates)
        ):
            raise ValueError('Exact complete native PDB source axes required')
        return dict(
            schema='Mora3.PDBMaskedCoordinateAxes/v1',
            atom_fields=ATOM_FIELDS,
            atom_label_axes=self.atom_label_axes,
            max_atoms=self.max_atoms,
            source_coordinates=self.source_coordinates,
            native_view_policy_sha256=self.native_view_policy_sha256,
            coordinate_unit='angstrom',
            original_split='train',
        )


def _is_sha256(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def native_view_policy_sha256(view):
    if type(view) is not FiniteUnionNativeView:
        raise ValueError('Exact current native view required')
    raw = json.dumps([view.manifest, getattr(view, 'effective_filter', None)],
                     sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _position(value):
    if value is None:
        return None
    if (type(value) is not list or len(value) != 3
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in value)):
        raise ValueError('Native coordinates must be complete finite XYZ triples')
    return tuple(float(x) for x in value)


def _project(row, path, ordinal):
    if (row['source_family'], row['task_name'], row['native_schema'], row['split']) != (FAMILY, TASK, SCHEMA, 'train'):
        raise ValueError('Exact original native PDB train task required')
    raw = row['native_payload']
    digest = hashlib.sha256(raw).hexdigest()
    if digest != row['native_payload_sha256']:
        raise ValueError('Native PDB payload checksum mismatch')
    task = json.loads(raw)
    if task['schema'] != SCHEMA or task['task'] != TASK:
        raise ValueError('Native PDB task identity mismatch')
    condition = task['conditions']
    targets = task['targets']
    if (set(condition) != {'coordinate_unit', 'known_atom_axis', 'positions_angstrom'}
            or condition['coordinate_unit'] != 'angstrom'
            or set(targets) != {'masked_positions_angstrom'}):
        raise ValueError('Exact native PDB condition and masked-coordinate target required')
    atom_axis = condition['known_atom_axis']
    if (type(atom_axis) is not list or not atom_axis
            or any(type(atom) is not dict or set(atom) != set(ATOM_FIELDS)
                   or any(type(atom[field]) is not str for field in ATOM_FIELDS) for atom in atom_axis)):
        raise ValueError('Complete literal native atom label axis required')
    atoms = tuple(tuple(atom[field] for field in ATOM_FIELDS) for atom in atom_axis)
    observed = tuple(_position(v) for v in condition['positions_angstrom'])
    withheld = tuple(_position(v) for v in targets['masked_positions_angstrom'])
    if len(atoms) != len(observed) or len(atoms) != len(withheld):
        raise ValueError('Native atom and coordinate axes must align exactly')
    if any((x is None) == (y is None) for x, y in zip(observed, withheld)):
        raise ValueError('Each coordinate is either observed or explicitly withheld')
    masked = tuple(i for i, point in enumerate(observed) if point is None)
    source_mask = task['audit']['masked_target_atom_indices']
    if (not masked or len(masked) == len(atoms) or any(type(i) is not int for i in source_mask)
            or tuple(sorted(source_mask)) != masked):
        raise ValueError('Native PDB withheld mask differs from the source allocation')
    return PDBCoordinatePacket(PDBCoordinateCondition(atoms, observed), withheld, (path, ordinal, digest))


def pdb_coordinate_packets(view, batch, *, split):
    """Use exact original retained rows and their source-native train role."""
    if type(view) is not FiniteUnionNativeView or split != 'train':
        raise ValueError('Exact current native view and original PDB train role required')
    view.require_native_rows(batch, split=split)
    try:
        return tuple(_project(row, batch.source_shard['path'], ordinal)
                     for row, ordinal in zip(batch.native_rows.to_pylist(), batch.source_row_ordinals))
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError('Malformed original native PDB coordinate packet') from exc


def complete_pdb_coordinate_source(view):
    """Read the complete finite PDB route once to fix its input axes and rank."""
    if type(view) is not FiniteUnionNativeView:
        raise ValueError('Exact current native view required')
    paths = []
    expected_rows = 0
    for descriptor in view.descriptors:
        entries = [e for e in descriptor['source_task_schema_counts']
                   if e['source_family'] == FAMILY and e['eligible_view_disposition'] == 0]
        if not entries:
            continue
        if any((e['task_name'], e['native_schema'], e['split']) != (TASK, SCHEMA, 'train') for e in entries):
            raise ValueError('Unexpected retained PDB route or split')
        paths.append(descriptor['source_shard']['path'])
        expected_rows += sum(e['rows'] for e in entries)
    if not paths:
        raise ValueError('Current native view has no retained PDB packets')
    packets = tuple(packet for batch in view.iter_native_task_batches(
        split='train', batch_size=4, source_paths=tuple(paths),
    ) for packet in pdb_coordinate_packets(view, batch, split='train'))
    if len(packets) != expected_rows:
        raise ValueError('Complete retained PDB source axis could not be closed')
    labels = tuple(tuple(sorted({row[i] for packet in packets for row in packet.condition.atom_labels}))
                   for i in range(len(ATOM_FIELDS)))
    axes = PDBCoordinateAxes(labels, max(len(p.condition.atom_labels) for p in packets),
                             tuple(sorted(p.source_coordinate for p in packets)),
                             native_view_policy_sha256(view))
    axes.binding()
    return axes, packets
