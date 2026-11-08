"""Integrated synthetic policy regressions; no scientific source exposure."""
import hashlib
import json
import pytest
from test_corpus_consumption_entrypoints_owner import module, task_row, manifest
from mora3.data import rgd1_consumption as gate


@pytest.mark.parametrize('split', ['train', 'validation', 'test'])
@pytest.mark.parametrize('field', ['source_record_id', 'source_reaction_family', 'task_name'])
@pytest.mark.parametrize('location', ['outer', 'native'])
def test_alternate_nested_identity_refuses_before_corpus_yield(tmp_path, split, field, location):
    row = task_row(split)
    identity = {'sources': [{field: ' RgD1:unqualified'}]}
    if location == 'outer':
        row['provenance_json_structure'] = identity
    else:
        raw = json.loads(row['native_payload'])
        raw['audit']['nested_provenance'] = identity
        row['native_payload'] = json.dumps(raw).encode()
        row['native_payload_sha256'] = hashlib.sha256(row['native_payload']).hexdigest()
    path = manifest(tmp_path, row)
    with pytest.raises(ValueError, match='RGD1.*quarantine'):
        next(module().read_corpus_tasks(path, root=tmp_path, split=split))


@pytest.mark.parametrize('split', ['train', 'validation', 'test'])
def test_cached_corpus_loader_rechecks_policy_each_consumption(tmp_path, monkeypatch, split):
    m = module()
    policy = tmp_path/'policy.json'
    policy.write_bytes(gate.POLICY_PATH.read_bytes())
    monkeypatch.setattr(gate, 'POLICY_PATH', policy)
    path = manifest(tmp_path, task_row(split, rgd=True))
    with pytest.raises(ValueError, match='RGD1.*quarantine'):
        next(m.read_corpus_tasks(path, root=tmp_path, split=split))
    assert m._corpus_consumption_guard is gate
    policy.write_bytes(b'{}')
    with pytest.raises(ValueError, match='policy integrity'):
        next(m.read_corpus_tasks(path, root=tmp_path, split=split))


def test_cached_conditioning_loader_rechecks_policy(tmp_path, monkeypatch):
    m = module()
    policy = tmp_path/'policy.json'
    policy.write_bytes(gate.POLICY_PATH.read_bytes())
    monkeypatch.setattr(gate, 'POLICY_PATH', policy)
    task = m.CorpusTask('synthetic', {}, {'energy': 1},
                        {'nested': [{'source_reaction_family': 'RGD1'}]}, [])
    original = task.canonical_bytes()
    with pytest.raises(ValueError, match='RGD1.*quarantine'):
        task.input_bytes()
    policy.write_bytes(b'{}')
    with pytest.raises(ValueError, match='policy integrity'):
        task.input_bytes()
    assert task.canonical_bytes() == original
