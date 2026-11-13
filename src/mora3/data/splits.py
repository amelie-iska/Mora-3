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


_SPLIT_ROLES = ('train', 'validation', 'test')


def _stratification_policy(seed, fractions):
    if type(seed) is not int or type(fractions) not in (tuple, list) or len(fractions) != 3:
        raise ValueError('Explicit integer seed and three split fractions required')
    if (any(type(f) not in (int, float) or not math.isfinite(f) or f < 0 for f in fractions)
            or not math.isclose(sum(fractions), 1., rel_tol=0, abs_tol=1e-12)):
        raise ValueError('Three nonnegative finite fractions must sum to one')
    return tuple(float(f) for f in fractions)


def _split_text(value):
    if type(value) is not str or not value.strip() or len(value) > 1024:
        raise ValueError('Bounded explicit component or stratum identity required')
    try:
        value.encode('utf-8')
    except UnicodeError as exc:
        raise ValueError('Valid UTF-8 identity required') from exc
    return value


def allocate_stratified_groups(groups, *, seed, fractions=(.8, .1, .1),
                               max_groups=200_000, max_strata=256):
    """Balance already established disjoint components without splitting them.

    Each group is {'rows': positive int, 'strata': {metadata_tag: positive int},
    'fixed_split': role or None}. Counts can describe millions of rows without
    loading their payloads. The caller must prove component membership, rights,
    original-role authority and source-appropriate metadata strata separately.
    No target values are inspected and no existing corpus partition is changed.
    This deterministic greedy objective is not an optimality or leakage theorem.
    """
    from collections import Counter
    fractions = _stratification_policy(seed, fractions)
    if (type(groups) is not dict or not groups
            or any(type(x) is not int or x < 1 for x in (max_groups, max_strata))
            or len(groups) > max_groups):
        raise ValueError('Complete nonempty bounded component inventory required')
    for identity in groups:
        _split_text(identity)
    normalized = {}
    totals = Counter()
    total_rows = 0
    for identity in sorted(groups):
        _split_text(identity)
        row = groups[identity]
        if (type(row) is not dict or set(row) != {'rows', 'strata', 'fixed_split'}
                or type(row['rows']) is not int or not 1 <= row['rows'] <= 2**53
                or type(row['strata']) is not dict or not row['strata']
                or row['fixed_split'] not in (*_SPLIT_ROLES, None)):
            raise ValueError('Exact positive component counts, strata and fixed split required')
        counts = {}
        for tag in row['strata']:
            _split_text(tag)
        for tag in sorted(row['strata']):
            _split_text(tag)
            count = row['strata'][tag]
            if type(count) is not int or not 1 <= count <= row['rows']:
                raise ValueError('Metadata-stratum counts must be positive and at most component rows')
            counts[tag] = count
            totals[tag] += count
        if len(totals) > max_strata:
            raise ValueError('Metadata stratum axis exceeds declared capacity')
        fixed = row['fixed_split']
        if fixed is not None and fractions[_SPLIT_ROLES.index(fixed)] == 0:
            raise ValueError('A fixed official role has zero requested capacity')
        normalized[identity] = dict(rows=row['rows'], strata=counts, fixed_split=fixed)
        total_rows += row['rows']
        if total_rows > 2**53:
            raise ValueError('Exact-count allocation exceeds supported integer range')
    tags = tuple(sorted(totals))
    by_split = {role: 0 for role in _SPLIT_ROLES}
    by_tag = {role: {tag: 0 for tag in tags} for role in _SPLIT_ROLES}
    assigned = {}

    def place(identity, role):
        row = normalized[identity]
        assigned[identity] = role
        by_split[role] += row['rows']
        for tag, count in row['strata'].items():
            by_tag[role][tag] += count

    for identity, row in normalized.items():
        if row['fixed_split'] is not None:
            place(identity, row['fixed_split'])

    def tie(*items):
        return hashlib.sha256(json.dumps([seed, *items], ensure_ascii=False,
            separators=(',', ':')).encode()).digest()

    def priority(identity):
        row = normalized[identity]
        concentration = math.fsum((n / totals[tag])**2 for tag, n in row['strata'].items())
        return (-concentration, -row['rows'], tie('component', identity), identity)

    for identity in sorted((i for i in normalized if i not in assigned), key=priority):
        row = normalized[identity]
        candidates = []
        for index, role in enumerate(_SPLIT_ROLES):
            fraction = fractions[index]
            if fraction == 0:
                continue
            # Incremental normalized squared error. Terms independent of the
            # selected role cancel; only strata present in this group change.
            old = by_split[role] / total_rows - fraction
            added = row['rows'] / total_rows
            delta = 2 * old * added + added * added
            delta += math.fsum(
                2 * (by_tag[role][tag] / totals[tag] - fraction) * (n / totals[tag])
                + (n / totals[tag])**2 for tag, n in row['strata'].items()
            ) / len(tags)
            candidates.append((delta, tie('role', identity, role), role))
        place(identity, min(candidates)[2])
    targets = {role: fractions[i] * total_rows for i, role in enumerate(_SPLIT_ROLES)}
    objective = math.fsum((by_split[role] / total_rows - fractions[i])**2
        for i, role in enumerate(_SPLIT_ROLES))
    objective += math.fsum((by_tag[role][tag] / totals[tag] - fractions[i])**2
        for i, role in enumerate(_SPLIT_ROLES) for tag in tags) / len(tags)
    binding = dict(schema='Mora3.ComponentStratificationInput/v1', seed=seed,
        fractions=fractions, groups=normalized,
        algorithm='fixed-first-concentration-ordered-incremental-squared-error-v1')
    return dict(schema='Mora3.ComponentStratification/v1',
        input_sha256=hashlib.sha256(json.dumps(binding, ensure_ascii=False, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
        seed=seed, fractions=fractions, group_assignments=dict(sorted(assigned.items())),
        total_rows=total_rows, total_groups=len(normalized),
        rows_by_split=by_split, groups_by_split={s: sum(v == s for v in assigned.values()) for s in _SPLIT_ROLES},
        target_rows_by_split=targets, stratum_totals=dict(sorted(totals.items())),
        strata_by_split=by_tag, normalized_squared_error=objective,
        fixed_groups=sum(row['fixed_split'] is not None for row in normalized.values()),
        empty_requested_splits=[s for i, s in enumerate(_SPLIT_ROLES) if fractions[i] > 0 and by_split[s] == 0],
        ratios_matched_exactly=all(by_split[s] == targets[s] for s in _SPLIT_ROLES),
        optimality_proven=False, unknown_leakage_excluded=False,
        scientific_source_admission=False, training_authorized=False)


def _native_stratification_inputs(records, strata, fixed_assignments, max_records):
    from itertools import islice
    if type(max_records) is not int or max_records < 1:
        raise ValueError('Positive native metadata row budget required')
    records = tuple(islice(iter(records), max_records + 1))
    if not records or len(records) > max_records:
        raise ValueError('Native grouping requires a complete bounded record population')
    groups = leakage_groups(records)
    fixed = {} if fixed_assignments is None else fixed_assignments
    if (type(strata) is not dict or set(strata) != set(groups) or type(fixed) is not dict
            or not set(fixed) <= set(groups) or any(s not in _SPLIT_ROLES for s in fixed.values())):
        raise ValueError('Exact stratum coverage and known fixed record roles required')
    for identity, tags in strata.items():
        if (type(tags) is not tuple or not tags or len(tags) > 32
                or any(type(t) is not str for t in tags) or len(set(tags)) != len(tags)):
            raise ValueError('Unique bounded immutable metadata stratum tags required')
        for tag in tags:
            _split_text(tag)
    return records, groups, fixed


def assign_stratified_splits(records, *, strata, seed, fixed_assignments=None,
                              fractions=(.8, .1, .1), max_records=200_000,
                              max_groups=200_000, max_strata=256):
    """Opt-in native grouping and stratification; legacy hash assignments unchanged."""
    from collections import Counter
    records, groups, fixed = _native_stratification_inputs(records, strata, fixed_assignments, max_records)
    metadata = {}
    for identity, group in groups.items():
        item = metadata.setdefault(group, dict(rows=0, strata=Counter(), fixed_split=None))
        item['rows'] += 1
        item['strata'].update(strata[identity])
        role = fixed.get(identity)
        if role is not None:
            if item['fixed_split'] is not None and item['fixed_split'] != role:
                raise ValueError('Conflicting fixed official splits in one transitive leakage component: ' + group)
            item['fixed_split'] = role
    metadata = {g: dict(row, strata=dict(row['strata'])) for g, row in metadata.items()}
    result = allocate_stratified_groups(metadata, seed=seed, fractions=fractions,
        max_groups=max_groups, max_strata=max_strata)
    result['assignments'] = {i: result['group_assignments'][g] for i, g in groups.items()}
    result['component_by_record'] = groups
    result['audit'] = audit_stratified_splits(records, result['assignments'], strata=strata,
        fixed_assignments=fixed, max_records=max_records)
    if not result['audit']['known_constraints_satisfied']:
        raise ValueError('Independent declared-group/fixed-role audit failed')
    return result


def audit_stratified_splits(records, assignments, *, strata, fixed_assignments=None,
                             max_records=200_000):
    """Recompute existing group separation and official roles, not unknown biology."""
    from collections import Counter
    records, groups, fixed = _native_stratification_inputs(records, strata, fixed_assignments, max_records)
    conflicts = audit_splits(records, assignments)
    role_conflicts = [dict(record_id=i, required=s, assigned=assignments[i])
        for i, s in sorted(fixed.items()) if assignments[i] != s]
    totals = {role: 0 for role in _SPLIT_ROLES}
    counts = {role: Counter() for role in _SPLIT_ROLES}
    for identity, role in assignments.items():
        totals[role] += 1
        counts[role].update(strata[identity])
    return dict(schema='Mora3.DeclaredGroupStratificationAudit/v1', rows=len(records),
        groups=len(set(groups.values())), rows_by_split=totals,
        strata_by_split={s: dict(sorted(c.items())) for s, c in counts.items()},
        declared_group_conflicts=conflicts, fixed_role_conflicts=role_conflicts,
        known_constraints_satisfied=not conflicts and not role_conflicts,
        unknown_biological_dependency_excluded=False, heldout_target_values_inspected=False,
        scientific_source_admission=False, training_authorized=False)
