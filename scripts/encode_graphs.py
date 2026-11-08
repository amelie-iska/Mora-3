#!/usr/bin/env python
"""Encode a bounded immutable graph slice using an already fitted vocabulary."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mora3.data.schema import GraphRecord
from mora3.tokenization.graph_payload import GraphPayloadCodec
from mora3.tokenization.shortest_path import ByteTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--vocabulary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output already exists; use a new directory')
    ancestor = args.output.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    budget = 10 * 2**20
    if shutil.disk_usage(ancestor).free < 200 * 2**30 + budget:
        raise ValueError('Must retain 200 GiB free disk')
    source = args.manifest.parent / 'records.jsonl'
    paths = (source, args.manifest, args.vocabulary)
    limits = (10 * 2**20, 2**20, 2**20)
    if any(p.stat().st_size > limit for p, limit in zip(paths, limits)):
        raise ValueError('Input exceeds bounded encoding limits')
    snapshots = {p: p.read_bytes() for p in paths}
    digest = lambda raw: hashlib.sha256(raw).hexdigest()
    manifest = json.loads(snapshots[args.manifest])
    if digest(snapshots[source]) != manifest['checksum_files']['records.jsonl']:
        raise ValueError('Source checksum mismatch')
    vocab = json.loads(snapshots[args.vocabulary])
    if vocab['version'] != 1 or vocab['byte_fallback'] != 256:
        raise ValueError('Unsupported vocabulary')
    tokenizer = ByteTokenizer(tuple(bytes.fromhex(t) for t in vocab['learned_hex']),
                              controls=vocab['controls'])
    if tokenizer.size != vocab['size']:
        raise ValueError('Vocabulary size mismatch')
    codec = GraphPayloadCodec(tokenizer)
    started = time.perf_counter()
    packed = []
    record_ids = set()
    for line in snapshots[source].splitlines():
        if not line.strip():
            continue
        record = GraphRecord.from_bytes(line)
        if record.record_id in record_ids:
            raise ValueError('Duplicate record ID')
        record_ids.add(record.record_id)
        encoded = codec.encode(record)
        restored = codec.decode(json.loads(json.dumps(encoded)))
        if restored.canonical_bytes() != record.canonical_bytes():
            raise RuntimeError('Graph round-trip failed')
        packed.append(encoded)
    if not packed:
        raise ValueError('No graph records')
    raw = ''.join(json.dumps(p, ensure_ascii=False, separators=(',', ':')) + '\n' for p in packed).encode('utf-8')
    if len(raw) > budget - 65536:
        raise ValueError('Output budget exceeded')
    unchanged = all(p.read_bytes() == before for p, before in snapshots.items())
    if not unchanged:
        raise RuntimeError('Input changed during encoding')
    entities = [e for p in packed for e in p['entities']]
    fields = [f for e in entities for f in e['measurements']]
    report = {'records': len(packed), 'entities': len(entities),
        'nodes': sum(e['role'] == 'node' for e in entities),
        'edges': sum(e['role'] == 'edge' for e in entities),
        'fields': len(fields), 'payload_bytes': sum(f['byte_count'] for f in fields),
        'payload_tokens': sum(len(f['token_ids']) for f in fields),
        'canonical_record_bytes': sum(p['canonical_record_bytes'] for p in packed),
        'encoded_artifact_bytes': len(raw), 'all_roundtrips_exact': True,
        'source_unchanged': unchanged, 'elapsed_seconds': time.perf_counter() - started,
        'source_sha256': {str(p): digest(b) for p, b in snapshots.items()},
        'artifact_sha256': {'encoded.jsonl': digest(raw)},
        'vocabulary_sha256': codec.vocabulary_sha256,
        'device': 'cpu', 'seed': None, 'downloaded_bytes': 0,
        'local_output_budget_bytes': budget, 'free_disk_bytes': shutil.disk_usage(ancestor).free,
        'neural_training': False, 'tensorization': False,
        'coverage': manifest.get('coverage', []),
        'scope': 'lossless scalar graph packing only; no loss/BPB or held-out evaluation',
        'model_input_policy': 'entities retain roles/incidence and typed fields; metadata is audit-only',
        'missing': ['tensor feature encoding', 'conditional masks', 'neural training', 'array modalities']}
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'encoded.jsonl').open('xb') as stream:
        stream.write(raw)
    with (args.output / 'measurement.json').open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
