"""Caller-declared desired color context, never an observed biological assertion.

This request contract grants no training rights and supplies no model capability.
It deliberately carries no pixels, masks, geometry, profiles or caller graph.
"""
from dataclasses import dataclass, asdict
import json

_FIELDS = frozenset(('organism', 'tissue', 'cell_type', 'stain'))


@dataclass(frozen=True)
class DesiredColorContext:
    image_sha256: str
    request_id: str
    origin: str
    values: tuple[tuple[str, str], ...]
    evidence_role: str = 'desired'
    training_admission: bool = False

    def __post_init__(self):
        import re
        if (not isinstance(self.image_sha256, str)
            or re.fullmatch('[0-9a-f]{64}', self.image_sha256) is None
            or self.evidence_role != 'desired' or self.training_admission is not False):
            raise ValueError('desired request binding/role required')
        for value in (self.request_id, self.origin):
            if not isinstance(value, str) or not value.strip():
                raise ValueError('explicit request identity and origin required')
        if not isinstance(self.values, (tuple, list)) or not self.values:
            raise ValueError('nonempty desired context pairs required')
        pairs = []
        for pair in self.values:
            if (not isinstance(pair, (tuple, list)) or len(pair) != 2
                or not isinstance(pair[0], str) or pair[0] not in _FIELDS
                or not isinstance(pair[1], str) or not pair[1].strip()):
                raise ValueError('supported desired text context required')
            pairs.append(tuple(pair))
        if len({key for key, _ in pairs}) != len(pairs):
            raise ValueError('duplicate desired context')
        object.__setattr__(self, 'values', tuple(sorted(pairs)))
        # Enforce valid Unicode and deterministic JSON at construction.
        try:
            self.canonical_bytes()
        except UnicodeError as exc:
            raise ValueError('invalid request text encoding') from exc

    def canonical_bytes(self):
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False,
                          separators=(',', ':'), allow_nan=False).encode('utf-8')
