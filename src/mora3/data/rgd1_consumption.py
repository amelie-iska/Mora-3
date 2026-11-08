"""Conservative RGD1 corpus-use guard; audit serialization remains separate.

No RGD1 release has a qualified energy-use binding yet. This guard deliberately
refuses the complete identified RGD1 task, including zero-loss conditioning and
evaluation. It does not authenticate source identity or admit other sources.
"""
import hashlib
import json
from pathlib import Path

POLICY_PATH = Path(__file__).resolve().parents[3] / 'planning/rgd1-v6-energy-objective-quarantine-20261101T195010Z-v1.json'
POLICY_SHA256 = '0f77c5310a55161c5d1723c0688b04addc56b5ede7241d6afcb522fd8414133b'

# Identity fields used by corpus envelopes and their nested provenance. Gene
# names, annotations and free text are biological content, not source identity.
# This is conservative detection, not authentication of arbitrary input rows.
IDENTITY_FIELDS = frozenset({
    'source_family', 'source_release', 'source_revision', 'source_record_id',
    'record_id', 'task', 'task_name', 'source_reaction_family',
})


def _identified(value):
    if isinstance(value, dict):
        return any(
            (key in IDENTITY_FIELDS and isinstance(v, str)
             and v.strip().casefold().startswith('rgd1'))
            or _identified(v)
            for key, v in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_identified(v) for v in value)
    return False


def require_corpus_consumption(row, *, purpose='objective'):
    """Refuse identified RGD1 data before model input, loss or quality use.

    Nested native JSON identity fields are inspected as well as the outer row;
    unrelated biological annotation strings do not identify the dataset. Opaque
    non-JSON array payloads are left to their native readers. No audit bypass or
    caller-supplied release/admission assertion can lift this restriction.
    """
    if purpose not in ('conditioning', 'objective', 'evaluation'):
        raise ValueError('explicit model consumption purpose required')
    if not isinstance(row, dict):
        raise ValueError('corpus row dictionary required')
    identified = _identified(row)
    payload = row.get('native_payload')
    if isinstance(payload, (bytes, str)):
        try:
            identified = identified or _identified(json.loads(payload))
        except (ValueError, UnicodeError):
            pass
    if not identified:
        return
    try:
        digest = hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError('RGD1 quarantine policy integrity unavailable') from exc
    if digest != POLICY_SHA256:
        raise ValueError('RGD1 quarantine policy integrity mismatch')
    raise ValueError('RGD1 energy quarantine: corpus model consumption requires separately qualified source binding')
