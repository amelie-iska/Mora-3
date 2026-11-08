"""Versioned conditional scalar-graph tasks, independent of source provenance.

Context bindings point to actual graph measurements, so they enter the same
encoder as other conditions. Matching is conservative exact equality, NOT a
cross-source join authorization, identity/similarity audit, or unit conversion.
Known topology/kinds are an explicit limitation; unknown-topology generation
must use a separate future contract. Callers must audit hidden semantic proxies.
"""
from dataclasses import asdict, dataclass
import json
from .schema import GraphRecord
from .provenance import require_text


def field_ref(value):
    value = tuple(value)
    if (len(value) != 3 or value[0] not in ('node', 'edge') or
            type(value[1]) is not int or value[1] < 0):
        raise ValueError('invalid field reference')
    require_text(value[2], 'field name')
    return value


@dataclass(frozen=True)
class ConditionalTask:
    roles: tuple
    context: tuple = ()
    topology_known: bool = False
    version: int = 1

    def __post_init__(self):
        if type(self.version) is not int or self.version != 1:
            raise ValueError('unsupported task version')
        if type(self.topology_known) is not bool:
            raise ValueError('topology_known must be boolean')
        roles = tuple((field_ref(ref), role) for ref, role in self.roles)
        if len(dict(roles)) != len(roles):
            raise ValueError('field has overlapping roles')
        if any(role not in ('condition', 'desired', 'target', 'evaluation', 'unused') for _, role in roles):
            raise ValueError('unknown role')
        if not any(role == 'target' for _, role in roles):
            raise ValueError('requested targets required')
        context = tuple((name, field_ref(ref)) for name, ref in self.context)
        for name, _ in context:
            require_text(name, 'context name')
        if len(dict(context)) != len(context) or len({r for _,r in context}) != len(context):
            raise ValueError('duplicate context binding')
        if any(dict(roles).get(ref) != 'condition' for _,ref in context):
            raise ValueError('experimental context must be a condition, not desired/hidden')
        object.__setattr__(self, 'roles', tuple(sorted(roles)))
        object.__setattr__(self, 'context', tuple(sorted(context)))

    def _fields(self, record):
        if not isinstance(record, GraphRecord):
            raise ValueError('validated GraphRecord required')
        if not self.topology_known:
            raise ValueError('unknown topology encoding unsupported')
        fields = {(role,i,m.name): m for role,entities in
                  (('node',record.nodes),('edge',record.edges))
                  for i,e in enumerate(entities) for m in e.measurements}
        if set(fields) != set(dict(self.roles)):
            raise ValueError('every field must have exactly one explicit role')
        return fields

    def input_fields(self, record):
        fields = self._fields(record)
        selected = frozenset(ref for ref,role in self.roles if role in ('condition','desired'))
        if any(not fields[ref].available for ref in selected):
            raise ValueError('unavailable condition or desired constraint')
        return selected

    def available_targets(self, record):
        fields = self._fields(record)
        targets = tuple(ref for ref,role in self.roles if role == 'target' and fields[ref].available)
        if any(fields[ref].status != 'observed' for ref in targets):
            raise ValueError('source predictions are not observed supervision')
        return targets

    def require_same_context(self, left, right):
        if not self.context:
            raise ValueError('empty context cannot establish a match')
        a,b = self._fields(left),self._fields(right)
        for _,ref in self.context:
            if not a[ref].available or not b[ref].available or a[ref] != b[ref]:
                raise ValueError('missing or incompatible experimental context')

    def canonical_bytes(self):
        return json.dumps(asdict(self),sort_keys=True,separators=(',',':'),
                          ensure_ascii=False,allow_nan=False).encode('utf-8')

    @classmethod
    def from_bytes(cls, data):
        def unique(pairs):
            result = {}
            for key,value in pairs:
                if key in result:
                    raise ValueError('duplicate task JSON key')
                result[key] = value
            return result
        try:
            d=json.loads(data,object_pairs_hook=unique)
            if set(d) != {'roles','context','topology_known','version'}:
                raise ValueError('invalid task fields')
            return cls(**d)
        except (TypeError,KeyError,UnicodeError) as exc:
            raise ValueError('malformed task') from exc
