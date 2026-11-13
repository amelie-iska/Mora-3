"""Verified, inactive BETSE INIT and trajectory source for native owner assembly."""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np


MERGED = Path('outputs/betse-merged-native-replay-v2')
INIT = Path('outputs/betse-init-membrane-capture-v1')
MERGED_SHA = '31efa1da48b54d5aa36cb168acad0a4f8f680a5d60811f24da0c7c9f12c7c4eb'
INIT_SHA = '35784ba43444dff6fd062331b5d7f19fd57e51563bc095a5e5bff4021b6d32b2'
MUP = Path('outputs/betse-mup-width-family-v1/receipt.json')
MUP_SHA = '23a1884da2d806870c3ac79cb697ee5ac0eed1bafb50d6d0054ab4071b75df91'
PREPARED = Path('outputs/betse-membrane-prepared-targets-v2/receipt.json')
PREPARED_SHA = '9656464856da4b3e3724ab899edfb1cc04a862fec2d6c45a5ae0261d2299ba1d'
ROLE = 'inactive-betse-train-candidate'
AXES = (
    ('sim_cell_mean_membrane_voltage_V', 'V', 'cell', 0.1, 11571),
    ('sampled_potassium_membrane_diffusion_m2_s', 'm^2/s', 'membrane', 1e-18, 63726),
    ('sampled_gap_junction_open_fraction', 'fraction', 'membrane', 1.0, 63726),
)
KNOWN = (
    'known__init_sampled_time_s', 'known__init_cell_mean_membrane_voltage_V',
    'known__sim_sampled_time_s', 'index__case_cell_offsets', 'index__case_membrane_offsets',
)
INIT_FIELDS = (
    'init_membrane_voltage_V', 'init_membrane_area_m2', 'init_membrane_cell_ordinals',
    'init_K_target', 'init_gap_target',
)


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _sha(raw):
    return sha256(raw).hexdigest()


def _read(root, relative, expected, cap):
    path = root / relative
    _need(path.resolve() == path and path.is_file() and path.stat().st_size <= cap,
          'Exact bounded BETSE source file required')
    raw = path.read_bytes()
    _need(len(raw) <= cap and _sha(raw) == expected, 'BETSE source digest differs')
    return raw


def _array_sha(value):
    return _sha(value.tobytes('C'))


@dataclass(frozen=True)
class BETSECondition:
    init_time: np.ndarray
    cell_voltage: np.ndarray
    query_time: np.ndarray
    cell_offsets: np.ndarray
    membrane_offsets: np.ndarray
    membrane_cell: np.ndarray
    membrane_voltage: np.ndarray
    membrane_area: np.ndarray
    k_assignment: np.ndarray
    gap_assignment: np.ndarray
    protocol: tuple


@dataclass(frozen=True)
class BETSESource:
    """Filesystem paths remain outside checkpoint identity. Sealed bytes bind the scope."""

    root: Path
    joint_snapshot_path: str | None = None
    joint_snapshot_sha256: str | None = None
    homology_completion_sha256: str | None = None

    def __post_init__(self):
        _need(type(self.root) is type(Path()) and self.root.is_absolute() and self.root.resolve() == self.root,
              'Exact absolute BETSE source root required')
        self.binding()

    def _joint_snapshot(self):
        if self.joint_snapshot_path is self.joint_snapshot_sha256 is self.homology_completion_sha256 is None:
            return None
        _need(type(self.joint_snapshot_path) is str and type(self.joint_snapshot_sha256) is str
              and type(self.homology_completion_sha256) is str,
              'Exact complete joint BETSE snapshot reference required')
        relative = Path(self.joint_snapshot_path)
        _need(not relative.is_absolute() and '..' not in relative.parts
              and str(relative) == self.joint_snapshot_path,
              'Exact normalized joint BETSE snapshot path required')
        raw = _read(self.root, relative, self.joint_snapshot_sha256, 16 * 1024)
        from .betse_v6_authority import build_joint_document, wire
        _need(raw == wire(build_joint_document(self.root, self.homology_completion_sha256)),
              'Joint BETSE snapshot does not bind the completed v6 chemical-quarantine replay')
        return self.joint_snapshot_sha256

    def _authorities(self):
        prepared = json.loads(_read(self.root, PREPARED, PREPARED_SHA, 16 * 1024))
        mup = json.loads(_read(self.root, MUP, MUP_SHA, 32 * 1024))
        merged = json.loads(_read(self.root, MERGED / 'receipt.json', MERGED_SHA, 8192))
        contract_pin = merged['staged_files']['native-merged-contract.json']['sha256']
        contract = json.loads(_read(self.root, MERGED / 'native-merged-contract.json', contract_pin, 128 * 1024))
        init = json.loads(_read(self.root, INIT / 'receipt.json', INIT_SHA, 48 * 1024))
        route = prepared['staged_binding']
        _need(prepared['active_roles'] == prepared['data_admissions'] == 0
              and route['role'] == ROLE and route['training_authorized'] is False
              and _sha(_wire(route)) == prepared['staged_route_sha256']
              and (route['merged_cohort_sha256'], route['init_capture_sha256'], route['mup_receipt_sha256'])
              == (MERGED_SHA, INIT_SHA, MUP_SHA)
              and route['finite_axis_sha256'] == mup['finite_axis_sha256']
              and route['source_finite_scope_sha256'] == mup['stage_binding_sha256']
              and route['primary_snapshot_sha256'] == mup['parent_native_scope']['sha256']
              and route['coordinates'] == [list(axis) for axis in AXES]
              and contract['active_role_assignment'] is None
              and init['parent_receipt']['sha256'] == MERGED_SHA
              and _sha(_wire(contract['role_batches']['train'])) == route['packed_record_sha256'],
              'BETSE scope, source role or ordered native axes differ')
        return prepared, contract, init

    def binding(self):
        prepared, _, _ = self._authorities()
        binding = {'schema': 'Mora3.BETSESource/v1', 'prepared_receipt_sha256': PREPARED_SHA,
                   'route': prepared['staged_binding'], 'route_sha256': prepared['staged_route_sha256'],
                   'training_admission': False}
        joint = self._joint_snapshot()
        if joint is not None:
            binding['joint_snapshot_sha256'] = joint
        return binding

    @property
    def owner_snapshot_sha256(self):
        binding = self.binding()
        return binding.get('joint_snapshot_sha256', _sha(_wire(binding)))

    @property
    def candidate_cursors(self):
        prepared, _, _ = self._authorities()
        return prepared['initial_cursor'].encode(), prepared['exhausted_cursor'].encode()

    def _members(self, record, keys):
        raw = _read(self.root, MERGED / record['file'], record['sha256'], 2 * 1024**2)
        with np.load(BytesIO(raw), allow_pickle=False) as archive:
            _need(len(archive.files) == len(record['numeric_fields'])
                  and set(archive.files) == set(record['numeric_fields']), 'Exact BETSE numeric members required')
            members = tuple(archive[key] for key in keys)
        for key, value in zip(keys, members, strict=True):
            ref = record['numeric_fields'][key]
            _need(value.dtype.str == ref['dtype'] and list(value.shape) == ref['shape']
                  and _array_sha(value) == ref['sha256_C_order'], 'BETSE numeric coordinate or value differs')
        return members

    def conditions(self, role):
        _need(type(role) is str and role == 'train', 'Only the inactive incumbent train diagnostic partition is supported')
        _, contract, init = self._authorities()
        record = contract['role_batches'][role]
        init_time, cell_voltage, query_time, cell_offsets, membrane_offsets = self._members(record, KNOWN)
        values = []
        for i, name in enumerate(record['case_ids']):
            ref = init['source_coverage'][name]
            raw = _read(self.root, INIT / name / 'init-membrane.npz', ref['array_ref']['sha256'], 128 * 1024)
            with np.load(BytesIO(raw), allow_pickle=False) as archive:
                voltage, area, parent, k, gap = (archive[key] for key in INIT_FIELDS)
            cells, membranes = np.diff(cell_offsets)[i], np.diff(membrane_offsets)[i]
            _need(voltage.shape == (7, membranes) and voltage.dtype.str == area.dtype.str == '<f8'
                  and area.shape == parent.shape == k.shape == gap.shape == (membranes,)
                  and parent.dtype.str == '<i8' and k.dtype == gap.dtype == np.bool_
                  and np.array_equal(np.unique(parent), np.arange(cells))
                  and cells == ref['cells'] and membranes == ref['membranes'],
                  'Exact pre-intervention membrane and ragged coordinate order required')
            values.append((voltage, area, parent + cell_offsets[i], k, gap))
        voltage, area, parent, k, gap = zip(*values, strict=True)
        return BETSECondition(init_time, cell_voltage, query_time, cell_offsets, membrane_offsets,
                              np.concatenate(parent), np.concatenate(voltage, axis=1), np.concatenate(area),
                              np.concatenate(k), np.concatenate(gap), tuple(record['per_case_known_event_schedules']))

    def _candidate_targets(self, condition):
        _, contract, _ = self._authorities()
        record = contract['role_batches']['train']
        for key, array in zip(KNOWN, (condition.init_time, condition.cell_voltage, condition.query_time,
                                    condition.cell_offsets, condition.membrane_offsets), strict=True):
            _need(_array_sha(array) == record['numeric_fields'][key]['sha256_C_order'],
                  'Candidate conditions differ from exact packed target coordinates')
        keys = tuple('target__' + axis[0] for axis in AXES)
        values = self._members(record, keys)
        _need(all(value.dtype.str == '<f8' and value.size == axis[-1] and np.isfinite(value).all()
                  for value, axis in zip(values, AXES, strict=True)), 'Exact finite dense BETSE targets required')
        return tuple((value, np.ones(value.shape, dtype=np.bool_)) for value in values)
