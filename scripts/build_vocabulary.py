#!/usr/bin/env python
"""Fit a bounded ConvexTok vocabulary from a checksum-pinned train-only slice."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import resource
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import scipy
from mora3.data.schema import GraphRecord
from mora3.tokenization.convex_vocab import fit_vocabulary, substring_candidates
from mora3.tokenization.shortest_path import ByteTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--field', required=True, help='Exact available string measurement name')
    parser.add_argument('--max-length', type=int, default=3)
    parser.add_argument('--vocab-size', type=int, default=283, help='Includes 256 bytes + 3 controls')
    parser.add_argument('--rounding', choices=('det', 'bias', 'int'), default='det')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output already exists; use a new run directory')
    if shutil.disk_usage(args.manifest.parent).free < 200 * 2**30:
        raise ValueError('Must retain 200 GiB free disk')
    digest = lambda raw: hashlib.sha256(raw).hexdigest()
    # Bound input reads before loading. No dataset scanning or source mutation.
    source = args.manifest.parent / 'records.jsonl'
    if source.stat().st_size > 10 * 2**20 or args.manifest.stat().st_size > 2**20:
        raise ValueError('Input exceeds bounded reference fit limits')
    manifest_raw = args.manifest.read_bytes()
    manifest = json.loads(manifest_raw)
    raw = source.read_bytes()
    if digest(raw) != manifest['checksum_files']['records.jsonl']:
        raise ValueError('Source checksum mismatch')
    records = [GraphRecord.from_bytes(line) for line in raw.splitlines() if line.strip()]
    if len({r.record_id for r in records}) != len(records):
        raise ValueError('Duplicate record IDs')
    fields = []
    for record in records:
        if manifest['split_assignments'].get(record.record_id) != 'train':
            raise ValueError('Every input record must have an explicit train assignment')
        for kind, entities in (('node', record.nodes), ('edge', record.edges)):
            for index, entity in enumerate(entities):
                for mi, measurement in enumerate(entity.measurements):
                    if (measurement.name == args.field and measurement.available
                            and isinstance(measurement.value, str)):
                        fields.append((record.record_id, kind, index, mi, measurement.value.encode('utf-8')))
    corpus = tuple(f[-1] for f in fields)
    if sum(map(len, corpus)) > 100_000:
        raise ValueError('Training payload byte limit exceeded')
    controls = ('pad', 'node', 'edge')
    started = time.perf_counter()
    candidates = substring_candidates(corpus, max_length=args.max_length)
    result = fit_vocabulary(corpus, candidates, vocab_size=args.vocab_size,
                            controls=controls, rounding=args.rounding)
    elapsed = time.perf_counter() - started
    tokenizer = ByteTokenizer(result.learned, controls=controls)
    encoded = []
    for rid, kind, index, mi, payload in fields:
        ids = tokenizer.encode(payload)
        if tokenizer.decode(ids) != payload:
            raise RuntimeError('Byte round-trip failed')
        encoded.append({'record_id': rid, 'entity': kind, 'index': index,
                        'measurement_index': mi, 'token_ids': ids})
    unchanged = source.read_bytes() == raw and args.manifest.read_bytes() == manifest_raw
    if not unchanged:
        raise RuntimeError('Source changed during fitting')
    args.output.mkdir(parents=True, exist_ok=False)
    def save(name, content):
        with (args.output / name).open('x') as stream:
            stream.write(content)
        return digest((args.output / name).read_bytes())
    vocabulary = {'version': 1, 'learned_hex': [t.hex() for t in result.learned],
                  'controls': controls, 'byte_fallback': 256, 'size': tokenizer.size,
                  'rounding': args.rounding,
                  'candidate_scores': [[t.hex(), score] for t, score in result.scores]}
    checksums = {'vocabulary.json': save('vocabulary.json', json.dumps(vocabulary, indent=2) + '\n'),
                 'encoded.jsonl': save('encoded.jsonl', ''.join(json.dumps(e) + '\n' for e in encoded))}
    metrics = asdict(result)
    metrics.pop('learned'); metrics.pop('scores')
    metrics.update({'source': str(source), 'source_sha256': digest(raw),
        'manifest': str(args.manifest), 'manifest_sha256': digest(manifest_raw),
        'source_unchanged': unchanged, 'records': len(records), 'fields': len(fields),
        'field_name': args.field, 'payload_bytes': sum(map(len, corpus)),
        'candidate_count': len(candidates), 'max_candidate_length': args.max_length,
        'vocabulary_budget': args.vocab_size, 'vocabulary_size': tokenizer.size,
        'rounding': args.rounding, 'all_roundtrips_exact': True,
        'elapsed_seconds': elapsed, 'peak_process_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'scipy_version': scipy.__version__, 'solver': 'scipy.optimize.linprog/highs',
        'device': 'cpu', 'seed': None, 'neural_training': False, 'vocabulary_learning': True,
        'evaluation_scope': 'training slice only; no held-out or full-corpus leakage claim',
        'candidate_scope': 'all within-field substrings of byte lengths 2 through max_candidate_length',
        'local_output_budget_bytes': 10 * 2**20, 'downloaded_bytes': 0,
        'free_disk_bytes': shutil.disk_usage(args.output).free, 'artifact_sha256': checksums})
    save('measurement.json', json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
