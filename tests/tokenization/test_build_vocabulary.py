"""CLI tests guard split leakage, immutable outputs and checksum provenance."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from mora3.data.schema import GraphRecord, Node, Measurement, Provenance


def run(tmp_path, split='train', corrupt=False):
    script = Path('scripts/build_vocabulary.py')
    assert script.exists(), 'vocabulary CLI missing'
    record = GraphRecord('r', (Node('rna', (Measurement('sequence', 'ACAC', None, 'observed', True),)),),
                         (), (Provenance('fixture', 'v1', 'r', '0' * 64, 'test-only'),))
    raw = record.canonical_bytes() + b'\n'
    (tmp_path / 'records.jsonl').write_bytes(raw)
    manifest = {'split_assignments': {'r': split}, 'checksum_files': {
        'records.jsonl': '0' * 64 if corrupt else hashlib.sha256(raw).hexdigest()}}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    args = [sys.executable, str(script), '--manifest', str(tmp_path / 'manifest.json'),
            '--output', str(tmp_path / 'fit'), '--field', 'sequence', '--max-length', '2',
            '--vocab-size', '260']
    result = subprocess.run(args, capture_output=True, text=True)
    return result, args


def test_fit_cli_roundtrip_and_refuse_overwrite(tmp_path):
    result, args = run(tmp_path)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / 'fit/measurement.json').read_text())
    assert report['payload_bytes'] == 4
    assert report['rounded_objective'] == 2
    assert report['source_unchanged'] and report['all_roundtrips_exact']
    saved = (tmp_path / 'fit/vocabulary.json').read_bytes()
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode != 0
    assert (tmp_path / 'fit/vocabulary.json').read_bytes() == saved


def test_fit_rejects_held_out_corpus(tmp_path):
    result, _ = run(tmp_path, split='validation')
    assert result.returncode != 0
    assert 'train' in result.stderr
    assert not (tmp_path / 'fit/vocabulary.json').exists()


def test_fit_rejects_checksum_mismatch(tmp_path):
    result, _ = run(tmp_path, corrupt=True)
    assert result.returncode != 0
    assert 'checksum' in result.stderr
    assert not (tmp_path / 'fit/vocabulary.json').exists()
