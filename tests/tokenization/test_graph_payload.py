"""Synthetic codec tests: type erasure, boundary merging and incidence loss regressions."""
import copy
import importlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from mora3.data.schema import GraphRecord, Node, Edge, Measurement, Provenance
from mora3.tokenization.shortest_path import ByteTokenizer


def codec(tokenizer=None):
    path = Path(__file__).resolve().parents[2] / 'src/mora3/tokenization/graph_payload.py'
    assert path.exists(), 'graph payload codec is not implemented'
    return importlib.import_module('mora3.tokenization.graph_payload').GraphPayloadCodec(
        tokenizer or ByteTokenizer((b'AC', b'CG'), controls=('pad', 'node', 'edge')))


def record():
    values = ['שָׁלוֹם[C][O]node', '', 1, 1.0, True, None, -0.0, 'AC', 'G']
    ms = tuple(Measurement(str(i), v, 'dimensionless' if type(v) in (int, float) else None,
                           'missing' if v is None else 'observed', v is not None)
               for i, v in enumerate(values))
    return GraphRecord('fixture', (Node('rna', ms), Node('empty')), (
        Edge(1, 0, 'directed', True), Edge(0, 0, 'self', True),
        Edge(0, 1, 'undirected', False)),
        (Provenance('fixture', 'v1', 'a', '0' * 64, 'test-only'),))


def test_exact_typed_roundtrip_through_json():
    c = codec()
    r = record()
    packed = json.loads(json.dumps(c.encode(r)))
    restored = c.decode(packed)
    assert restored.canonical_bytes() == r.canonical_bytes()
    assert [type(m.value) for m in restored.nodes[0].measurements] == [str, str, int, float, bool, type(None), float, str, str]


def test_entity_roles_endpoints_and_empty_entities_preserved():
    p = codec().encode(record())
    assert len(p['entities']) == 5  # n+m, even empty node and self-loop edge
    assert [(e['role'], e['endpoints']) for e in p['entities']] == [
        ('node', [0, 0]), ('node', [1, 1]), ('edge', [1, 0]), ('edge', [0, 0]), ('edge', [0, 1])]
    assert p['entities'][0]['role_id'] != p['entities'][3]['role_id']
    assert [e['directed'] for e in p['entities'][2:]] == [True, True, False]


def test_measurement_boundaries_are_not_joined_or_interpreted_as_controls():
    c = codec()
    p = c.encode(record())
    fields = p['entities'][0]['measurements']
    assert fields[-2]['token_ids'] == [256]  # AC
    assert fields[-1]['token_ids'] == [71]  # G; cannot merge C+G across fields
    assert fields[1]['token_ids'] == []  # empty string
    assert fields[5]['value_type'] == 'null' and fields[5]['token_ids'] == []
    assert all(i < 258 for f in fields for i in f['token_ids'])
    assert [f['index'] for f in fields] == list(range(9))
    assert p['metadata']['provenance'][0]['source_id'] == 'a'
    assert 'provenance' not in p['entities'][0]


@pytest.mark.parametrize('mutation', ['role', 'endpoint', 'type', 'count', 'id', 'index', 'metadata', 'extra', 'version'])
def test_corrupt_or_ambiguous_artifacts_rejected(mutation):
    c = codec()
    p = copy.deepcopy(c.encode(record()))
    if mutation == 'role': p['entities'][3]['role'] = 'node'
    if mutation == 'endpoint': p['entities'][2]['endpoints'] = [0, 1]
    if mutation == 'type': p['entities'][0]['measurements'][2]['value_type'] = 'bool'
    if mutation == 'count': p['entities'][0]['measurements'][0]['byte_count'] += 1
    if mutation == 'id': p['entities'][0]['measurements'][0]['token_ids'] = [258]
    if mutation == 'index': p['entities'][0]['index'] = 7
    if mutation == 'metadata': p['metadata']['record_id'] = 'different'
    if mutation == 'extra': p['unrecognized'] = True
    if mutation == 'version': p['version'] = 2
    with pytest.raises(ValueError): c.decode(p)


def test_vocabulary_order_is_bound_to_artifact():
    p = codec().encode(record())
    different = codec(ByteTokenizer((b'CG', b'AC'), controls=('pad', 'node', 'edge')))
    with pytest.raises(ValueError, match='vocabulary'): different.decode(p)


def test_predicted_and_missing_units_survive():
    r = GraphRecord('x', (Node('structure', (
        Measurement('score', .75, 'dimensionless', 'predicted', True),
        Measurement('coords', None, 'angstrom', 'missing', False))),), (), record().provenance)
    c = codec()
    assert c.decode(c.encode(r)).canonical_bytes() == r.canonical_bytes()

@pytest.mark.parametrize('case', ['valid', 'bad_checksum', 'existing_output', 'wrong_vocab'])
def test_graph_encoding_cli_preserves_inputs_and_rejects_mismatch(tmp_path, case):
    import hashlib
    import subprocess
    root = Path(__file__).resolve().parents[2]
    script = root / 'scripts/encode_graphs.py'
    assert script.exists(), 'graph encoding CLI is not implemented'
    raw = record().canonical_bytes() + b'\n'
    (tmp_path / 'records.jsonl').write_bytes(raw)
    manifest = {'checksum_files': {'records.jsonl': hashlib.sha256(raw).hexdigest()}}
    if case == 'bad_checksum': manifest['checksum_files']['records.jsonl'] = '0' * 64
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    vocab = {'version': 1, 'learned_hex': ['4143'], 'controls': ['pad', 'node', 'edge'],
             'byte_fallback': 256, 'size': 260}
    if case == 'wrong_vocab': vocab['size'] = 261
    (tmp_path / 'vocabulary.json').write_text(json.dumps(vocab))
    output = tmp_path / 'result'
    if case == 'existing_output': output.mkdir()
    before = {p: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    run = subprocess.run([sys.executable, str(script), '--manifest', str(tmp_path / 'manifest.json'),
        '--vocabulary', str(tmp_path / 'vocabulary.json'), '--output', str(output)], capture_output=True, text=True)
    assert {p: p.read_bytes() for p in before} == before
    if case != 'valid':
        assert run.returncode != 0
        assert not (output / 'encoded.jsonl').exists()
    else:
        assert run.returncode == 0, run.stderr
        report = json.loads((output / 'measurement.json').read_text())
        assert report['records'] == 1 and report['entities'] == 5
        assert report['all_roundtrips_exact'] and report['source_unchanged']
        p = json.loads((output / 'encoded.jsonl').read_text())
        assert codec(ByteTokenizer((b'AC',), controls=('pad', 'node', 'edge'))).decode(p) == record()
