"""Synthetic contract fixtures, never training-data coverage."""
import dataclasses
import importlib
import itertools
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))


def api():
    assert (Path(__file__).resolve().parents[2] / 'src/mora3/data/schema.py').exists(), 'I1 contracts missing'
    return importlib.import_module('mora3.data.schema')


def record(name='a', keys=()):
    s = api()
    p = s.Provenance('fixture', 'v1', name, '0' * 64, 'test-only')
    return s.GraphRecord(name, (s.Node('protein', (s.Measurement('sequence', 'שָׁלוֹם', None, 'observed', True),)),
                                s.Node('structure', (s.Measurement('coordinates', None, 'angstrom', 'missing', False),))),
                         (s.Edge(0, 1, 'has_structure', True), s.Edge(0, 0, 'self', True)), (p,), keys)


def test_roundtrip_direction_missingness_and_immutable_provenance():
    s = api()
    r = record()
    restored = s.GraphRecord.from_bytes(r.canonical_bytes())
    assert restored == r
    assert restored.edges[0].source == 0 and restored.edges[0].target == 1
    assert restored.edges[1].source == restored.edges[1].target == 0
    assert restored.nodes[1].measurements[0].available is False
    assert restored.nodes[0].measurements[0].value == 'שָׁלוֹם'
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.provenance[0].source_id = 'changed'
    assert r.canonical_bytes() == json.dumps(json.loads(r.canonical_bytes()), ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    assert r.canonical_byte_count == len(r.canonical_bytes())
    with pytest.raises(ValueError):
        s.GraphRecord.from_bytes(r.canonical_bytes().replace(b'"version":1', b'"version":2'))


@pytest.mark.parametrize('value,units,status,available', [
    (None, None, 'observed', True), (2.0, 'nM', 'missing', False),
    (float('nan'), 'nM', 'observed', True), (float('inf'), 'nM', 'predicted', True),
    (2.0, None, 'observed', True), ('x', None, 'invented', True),
    ([], None, 'observed', True), ('x', None, 'observed', 1),
])
def test_invalid_measurements_fail_closed(value, units, status, available):
    with pytest.raises(ValueError):
        api().Measurement('affinity', value, units, status, available)


@pytest.mark.parametrize('source,target', [(-1, 0), (0, 2), (True, 0), (0.0, 1)])
def test_invalid_endpoints_rejected(source, target):
    s = api()
    with pytest.raises(ValueError):
        dataclasses.replace(record(), edges=(s.Edge(source, target, 'binds', True),))


def test_valid_zero_predicted_values_and_sequence_freezing():
    s = api()
    m = s.Measurement('affinity', 0.0, 'nM', 'predicted', True)
    items = [m]
    node = s.Node('protein', items)
    items.clear()
    assert node.measurements == (m,)
    with pytest.raises(ValueError):
        dataclasses.replace(record(), provenance=())
    with pytest.raises(ValueError):
        dataclasses.replace(record().provenance[0], sha256='bad')
    with pytest.raises(ValueError):
        s.Node('protein', (m, m))


def test_transitive_groups_match_independent_reachability_and_input_order():
    s = api()
    split = importlib.import_module('mora3.data.splits')
    records = [record('a', (s.LeakageKey('sequence', 'x'),)),
               record('b', (s.LeakageKey('sequence', 'x'), s.LeakageKey('donor', 'y'))),
               record('c', (s.LeakageKey('donor', 'y'),)),
               record('d', (s.LeakageKey('study', 'x'),))]
    groups = split.leakage_groups(records)
    assert groups['a'] == groups['b'] == groups['c']
    assert groups['d'] != groups['a']
    assignments = split.assign_splits(records, seed=19)
    for perm in itertools.permutations(records):
        assert split.leakage_groups(perm) == groups
        assert split.assign_splits(perm, seed=19) == assignments
    assert split.audit_splits(records, assignments) == []
    assert split.audit_splits(records, {'a': 'train', 'b': 'train', 'c': 'test', 'd': 'test'})
    # Independent transitive closure oracle over a larger bridged graph.
    records = [record(str(i), tuple(s.LeakageKey('cluster', str(k)) for k in keys))
               for i, keys in enumerate([(1,), (2,), (1, 3), (3, 2), (), (7,), (7,)])]
    groups = split.leakage_groups(records)
    assert len(set(groups.values())) == 3
    assert groups['0'] == groups['1'] == groups['2'] == groups['3']
    assert groups['5'] == groups['6'] != groups['4']


def test_provenance_identity_also_prevents_leakage_and_bad_assignments_fail():
    s = api()
    split = importlib.import_module('mora3.data.splits')
    a = record('a')
    b = dataclasses.replace(a, record_id='b')
    assert len(set(split.leakage_groups([a, b]).values())) == 1
    with pytest.raises(ValueError):
        split.leakage_groups([a, a])
    with pytest.raises(ValueError):
        split.audit_splits([a, b], {'a': 'train'})
    with pytest.raises(ValueError):
        split.assign_splits([a], fractions=(0.8, 0.3, -0.1))
    assert split.assign_splits([a, b], fractions=(1, 0, 0)) == {'a': 'train', 'b': 'train'}
    assert split.assign_splits([a, b], fractions=(0, 0, 1)) == {'a': 'test', 'b': 'test'}
