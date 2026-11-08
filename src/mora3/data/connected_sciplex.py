"""Pinned sci-Plex inspection tasks with complete scalar/array context.

This source adapter is not a general cross-study join or a training eligibility
policy. Its physical dose mapping remains unresolved, so optimization is refused.
Categorical IDs below are source labels, not chemical structure representations.
"""
from dataclasses import dataclass
import json
import re
import torch
from .array_tasks import ArrayTask, OrderedArray
from .gperturb_sciplex import array_task


INTERVENTIONS = ('Dex', 'Nutlin', 'BMS', 'SAHA', 'Vehicle')


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def _load(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique)


@dataclass(frozen=True)
class SciPlexTask:
    """Immutable snapshot; canonical metadata includes hidden labels for resume only."""
    sidecar_bytes: bytes
    source_sha256: str

    def __post_init__(self):
        if not isinstance(self.source_sha256, str) or not re.fullmatch('[0-9a-f]{64}', self.source_sha256):
            raise ValueError('source SHA256 required')
        if not isinstance(self.sidecar_bytes, bytes):
            raise ValueError('immutable bytes required')
        try:
            s = _load(self.sidecar_bytes)
            direction = s['task']
            a = s['targets' if direction == 'response' else 'conditions']['expression']
            i = s['conditions' if direction == 'response' else 'targets']['intervention']
            e = s['evaluation']
            if (type(e['source_index']) is not int or e['source_index'] < 0 or
                not isinstance(e['cell_id'], str) or not e['cell_id'].strip() or
                type(e['n_genes']) is not int or e['n_genes'] < 0 or
                type(e['n_counts']) not in (int,float) or not 0 <= e['n_counts'] < float('inf')):
                raise ValueError('invalid evaluation metadata')
            row = dict(e, counts=a['values'], intervention=i['identity'],
                       dose_coordinate=i['source_coordinate'], dose_units=i['units'])
            expected = array_task(row, a['axis_ids'], direction=direction)
            # Exact source-specific context, role and missingness semantics. No
            # caller-supplied readiness, extra fields or guessed physical units.
            if _canonical(s) != _canonical(expected):
                raise ValueError('incompatible source task/context')
            OrderedArray('expression', a['axis_ids'], a['values'],
                         a['axis_identity'], a['units'], a['status'])
            object.__setattr__(self, 'sidecar_bytes', _canonical(s))
        except (KeyError, TypeError, UnicodeError, OverflowError) as exc:
            raise ValueError('malformed connected task') from exc

    @classmethod
    def from_sidecar(cls, sidecar, *, source_sha256):
        return cls(_canonical(sidecar), source_sha256)

    def _array_task(self):
        s = _load(self.sidecar_bytes)
        role = 'target' if s['task'] == 'response' else 'condition'
        a = s['targets' if role == 'target' else 'conditions']['expression']
        return ArrayTask((OrderedArray('expression', a['axis_ids'], a['values'],
            a['axis_identity'], a['units'], a['status']),),
            (('expression', role),), self.source_sha256)

    def input_metadata(self):
        s = _load(self.sidecar_bytes)
        result = dict(version=1, task=s['task'], context=s['conditions']['context'],
                      arrays=self._array_task().input_metadata(),
                      categorical_interventions=list(INTERVENTIONS),
                      dose_units='source_scaled_coordinate',
                      source_adapter='gperturb-sciplex2-inspection-v1')
        return result

    def project(self, *, device='cpu', dtype=torch.float64, for_training=False):
        if for_training:
            raise ValueError('Physical dose conversion unresolved; quarantined tasks cannot train')
        s = _load(self.sidecar_bytes)
        result = self._array_task().project(device=device, dtype=dtype)
        result['inputs']['exposure_hours'] = torch.tensor(
            s['conditions']['context']['exposure_hours'], device=device, dtype=dtype)
        where = 'conditions' if s['task'] == 'response' else 'targets'
        i = s[where]['intervention']
        values = dict(intervention_index=torch.tensor(INTERVENTIONS.index(i['identity']),
                          dtype=torch.int64, device=device),
                      dose_coordinate=torch.tensor(i['source_coordinate'], dtype=dtype, device=device))
        group = 'inputs' if where == 'conditions' else 'targets'
        result[group].update(values)
        if group == 'targets':
            result['target_masks'].update({key: torch.tensor(True, device=device) for key in values})
        return result

    def canonical_bytes(self):
        return _canonical(dict(version=1, source_sha256=self.source_sha256,
                               sidecar=_load(self.sidecar_bytes)))

    @classmethod
    def from_bytes(cls, raw):
        try:
            d = _load(raw)
            if set(d) != {'version', 'source_sha256', 'sidecar'} or type(d['version']) is not int or d['version'] != 1:
                raise ValueError('invalid connected metadata version/fields')
            return cls.from_sidecar(d['sidecar'], source_sha256=d['source_sha256'])
        except (TypeError, KeyError, UnicodeError) as exc:
            raise ValueError('malformed connected metadata') from exc

    def assert_resume_matches(self, other):
        if not isinstance(other, SciPlexTask) or self.canonical_bytes() != other.canonical_bytes():
            raise ValueError('resume task/source/context mismatch')
