"""Immutable source identities; license labels are evidence, not access grants."""
from dataclasses import dataclass
import re


def require_text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field} must be nonempty text')
    try:
        value.encode('utf-8')
    except UnicodeError as exc:
        raise ValueError(f'{field} must be valid UTF-8') from exc


@dataclass(frozen=True)
class Provenance:
    source: str
    revision: str
    source_id: str
    sha256: str
    license: str

    def __post_init__(self):
        for name in ('source', 'revision', 'source_id', 'license'):
            require_text(getattr(self, name), name)
        if not isinstance(self.sha256, str) or not re.fullmatch('[0-9a-f]{64}', self.sha256):
            raise ValueError('sha256 must be 64 lowercase hexadecimal characters')


@dataclass(frozen=True, order=True)
class LeakageKey:
    """Namespace must include the clustering policy/version or source scope."""
    namespace: str
    value: str

    def __post_init__(self):
        require_text(self.namespace, 'namespace')
        require_text(self.value, 'value')
