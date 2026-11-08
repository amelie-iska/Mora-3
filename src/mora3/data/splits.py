"""Transitive leakage groups and stable hash assignments; no row-random splits.

Grouping is only as complete as upstream keys. No similarity search is performed.
Assignments are approximate fractions, with no guarantee of nonempty splits.
Freeze the input manifest: adding bridging records can merge existing groups.
"""
import hashlib
import json
import math
from .schema import GraphRecord


def leakage_groups(records):
    records = tuple(records)
    if any(not isinstance(r, GraphRecord) for r in records):
        raise ValueError('expected GraphRecord')
    ids = [r.record_id for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate record IDs')
    parent = {i: i for i in ids}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owners = {}
    for r in records:
        keys = [('key', k.namespace, k.value) for k in r.leakage_keys]
        # Conservatively group repeated biological source IDs across revisions.
        keys += [('source', p.source, p.source_id) for p in r.provenance]
        for key in keys:
            other = owners.setdefault(key, r.record_id)
            a, b = find(other), find(r.record_id)
            parent[max(a, b)] = min(a, b)
    return {i: find(i) for i in sorted(ids)}


def assign_splits(records, seed=0, fractions=(0.8, 0.1, 0.1)):
    if type(seed) is not int:
        raise ValueError('seed must be integer')
    if (len(fractions) != 3 or any(type(f) not in (int, float) or not math.isfinite(f) or f < 0 for f in fractions)
            or not math.isclose(sum(fractions), 1.0, rel_tol=0, abs_tol=1e-12)):
        raise ValueError('three nonnegative finite fractions must sum to one')
    groups = leakage_groups(records)
    assigned = {}
    for group in set(groups.values()):
        digest = hashlib.sha256(json.dumps([seed, group], ensure_ascii=False, separators=(',', ':')).encode()).digest()
        u = int.from_bytes(digest, 'big') / 2**256
        assigned[group] = 'train' if u < fractions[0] else 'validation' if u < sum(fractions[:2]) else 'test'
    return {i: assigned[g] for i, g in groups.items()}


def audit_splits(records, assignments):
    groups = leakage_groups(records)
    if set(assignments) != set(groups) or any(s not in ('train', 'validation', 'test') for s in assignments.values()):
        raise ValueError('assignments must cover exactly all records with valid splits')
    seen = {}
    for i, group in groups.items():
        seen.setdefault(group, set()).add(assignments[i])
    return [{'group': g, 'splits': sorted(s)} for g, s in sorted(seen.items()) if len(s) > 1]
