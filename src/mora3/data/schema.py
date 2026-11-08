"""I1 v1 graph contract: ordered endpoints, explicit absence, UTF-8 serialization.

Canonical bytes cover the entire unpadded record, including provenance. This is
not permutation-canonical graph isomorphism or a conditional-loss denominator.
Numeric measurements require units (use 'dimensionless' when appropriate).
Arrays are deliberately deferred until a source adapter specifies their schema.
"""
from dataclasses import asdict, dataclass
import json
import math
from .provenance import LeakageKey, Provenance, require_text


def typed_tuple(value, cls, name):
    result = tuple(value)
    if any(not isinstance(item, cls) for item in result):
        raise ValueError(f'{name} contains invalid types')
    return result


@dataclass(frozen=True)
class Measurement:
    name: str
    value: str | int | float | bool | None
    units: str | None
    status: str
    available: bool

    def __post_init__(self):
        require_text(self.name, 'measurement name')
        if type(self.available) is not bool or self.status not in ('observed', 'predicted', 'missing'):
            raise ValueError('invalid availability or evidence status')
        if self.available != (self.value is not None) or (self.status == 'missing') != (not self.available):
            raise ValueError('value, status and availability disagree')
        if self.units is not None:
            require_text(self.units, 'units')
        if self.value is not None and type(self.value) not in (str, int, float, bool):
            raise ValueError('measurement must be a JSON scalar')
        if isinstance(self.value, str):
            try:
                self.value.encode('utf-8')
            except UnicodeError as exc:
                raise ValueError('invalid UTF-8 measurement') from exc
        if type(self.value) in (int, float):
            if self.units is None or (type(self.value) is float and not math.isfinite(self.value)):
                raise ValueError('numeric measurements need units and finite values')


def measurements(value):
    result = typed_tuple(value, Measurement, 'measurements')
    if len({m.name for m in result}) != len(result):
        raise ValueError('duplicate measurement names')
    return result


@dataclass(frozen=True)
class Node:
    kind: str
    measurements: tuple[Measurement, ...] = ()

    def __post_init__(self):
        require_text(self.kind, 'node kind')
        object.__setattr__(self, 'measurements', measurements(self.measurements))


@dataclass(frozen=True)
class Edge:
    source: int
    target: int
    kind: str
    directed: bool
    measurements: tuple[Measurement, ...] = ()

    def __post_init__(self):
        require_text(self.kind, 'edge kind')
        if type(self.directed) is not bool:
            raise ValueError('directed must be boolean')
        if any(type(i) is not int or i < 0 for i in (self.source, self.target)):
            raise ValueError('endpoints must be nonnegative integers')
        object.__setattr__(self, 'measurements', measurements(self.measurements))


@dataclass(frozen=True)
class GraphRecord:
    record_id: str
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    provenance: tuple[Provenance, ...]
    leakage_keys: tuple[LeakageKey, ...] = ()
    version: int = 1

    def __post_init__(self):
        require_text(self.record_id, 'record_id')
        if type(self.version) is not int or self.version != 1:
            raise ValueError('unsupported schema version')
        for name, cls in (('nodes', Node), ('edges', Edge), ('provenance', Provenance), ('leakage_keys', LeakageKey)):
            object.__setattr__(self, name, typed_tuple(getattr(self, name), cls, name))
        if not self.nodes or not self.provenance:
            raise ValueError('nodes and provenance are required')
        if any(max(e.source, e.target) >= len(self.nodes) for e in self.edges):
            raise ValueError('edge endpoint out of range')

    def canonical_bytes(self):
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True,
                          separators=(',', ':'), allow_nan=False).encode('utf-8')

    @property
    def canonical_byte_count(self):
        return len(self.canonical_bytes())

    @classmethod
    def from_bytes(cls, data):
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError('duplicate JSON key')
                result[key] = value
            return result
        try:
            d = json.loads(data, object_pairs_hook=unique)
            if set(d) != {'record_id', 'nodes', 'edges', 'provenance', 'leakage_keys', 'version'}:
                raise ValueError('invalid record fields')
            for name, constructor in (('nodes', Node), ('edges', Edge)):
                d[name] = tuple(constructor(**{**item, 'measurements': tuple(Measurement(**m) for m in item['measurements'])}) for item in d[name])
            d['provenance'] = tuple(Provenance(**p) for p in d['provenance'])
            d['leakage_keys'] = tuple(LeakageKey(**k) for k in d['leakage_keys'])
            return cls(**d)
        except (TypeError, KeyError, UnicodeError) as exc:
            raise ValueError('malformed graph record') from exc
