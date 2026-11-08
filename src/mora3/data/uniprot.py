"""Bounded UniProtKB scalar projection; database annotations are not validation.

Raw response retains residue features, evidence codes and other omitted fields.
No network IO, structure imputation, similarity search or held-out split claim.
"""
import hashlib
import json
from pathlib import Path
import re
import shutil
from .schema import GraphRecord, Node, Edge, Measurement
from .provenance import Provenance, LeakageKey, require_text
from .splits import leakage_groups, audit_splits


def _results(raw):
    def unique(pairs):
        d = {}
        for k, v in pairs:
            if k in d:
                raise ValueError('duplicate JSON key')
            d[k] = v
        return d
    try:
        results = json.loads(raw, object_pairs_hook=unique)['results']
        if not isinstance(results, list) or not 1 <= len(results) <= 64:
            raise ValueError('Require 1..64 records')
        ids = [e['primaryAccession'] for e in results]
        if any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
            raise ValueError('Invalid or duplicate accession')
        return results
    except (KeyError, TypeError) as exc:
        raise ValueError('Malformed search response') from exc


def convert(api_bytes, accession, release):
    """Project one accession, binding provenance to the entire exact response."""
    if not isinstance(release, str) or not re.fullmatch(r'\d{4}_\d{2}', release):
        raise ValueError('Explicit UniProt release required')
    if not isinstance(accession, str) or not re.fullmatch(r'(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})', accession):
        raise ValueError('Invalid accession')
    entries = [e for e in _results(api_bytes) if e['primaryAccession'] == accession]
    if len(entries) != 1:
        raise ValueError('Missing accession')
    e = entries[0]
    try:
        if e['entryType'] != 'UniProtKB reviewed (Swiss-Prot)':
            raise ValueError('Reviewed entry required')
        score = e['annotationScore']
        if type(score) not in (int, float) or score not in (4, 5):
            raise ValueError('Annotation score must be 4 or 5')
        sequence = e['sequence']['value']
        length = e['sequence']['length']
        if (not isinstance(sequence, str) or not re.fullmatch('[ACDEFGHIKLMNPQRSTVWYBXZJUO]+', sequence)
                or type(length) is not int or length != len(sequence) or length > 8192):
            raise ValueError('Invalid sequence/length or bounded adapter limit exceeded')
        audit = e['entryAudit']
        versions = [audit[k] for k in ('entryVersion', 'sequenceVersion')]
        if any(type(v) is not int or v < 1 for v in versions):
            raise ValueError('Invalid source versions')
        name = e['proteinDescription']['recommendedName']['fullName']['value']
        require_text(name, 'recommended name')
        annotations = [Measurement('name', name, None, 'observed', True),
            Measurement('annotation_score', int(score), 'dimensionless', 'observed', True)]
        texts = [t['value'] for c in e.get('comments', []) if c['commentType'] == 'FUNCTION'
                 for t in c.get('texts', [])]
        for i, value in enumerate(texts):
            require_text(value, 'function text')
            annotations.append(Measurement(f'function_{i}', value, None, 'observed', True))
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('Malformed UniProt entry') from exc
    sha = hashlib.sha256(api_bytes).hexdigest()
    provenance = Provenance('UniProtKB', f'{release};entry:{versions[0]};sequence:{versions[1]}',
                            accession, sha, 'CC-BY-4.0')
    nodes = (Node('protein', (Measurement('sequence', sequence, None, 'observed', True),
                              Measurement('structure', None, 'angstrom', 'missing', False))),
             Node('protein_annotation', tuple(annotations)))
    return GraphRecord('uniprot:' + accession, nodes, (Edge(0, 1, 'has_annotation', True),),
        (provenance,), (LeakageKey('sequence:protein:sha256:v1', hashlib.sha256(sequence.encode()).hexdigest()),))


def curate(source, output, *, expected_sha256, release):
    """Offline, train-only curation with a reject report and no source writes."""
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError('Output exists; use a fresh directory')
    if source.stat().st_size > 4 * 2**20:
        raise ValueError('Source exceeds 4 MiB budget')
    raw = source.read_bytes()
    digest = lambda b: hashlib.sha256(b).hexdigest()
    if digest(raw) != expected_sha256:
        raise ValueError('Source checksum mismatch')
    if shutil.disk_usage(source.parent).free < 200 * 2**30 + 20 * 2**20:
        raise ValueError('Must retain 200 GiB free')
    records, rejected = [], []
    for entry in _results(raw):
        acc = entry['primaryAccession']
        try:
            records.append(convert(raw, acc, release))
        except ValueError as exc:
            rejected.append({'accession': acc, 'reason': str(exc)})
    if not records:
        raise ValueError('No accepted records')
    packed = b''.join(r.canonical_bytes() + b'\n' for r in records)
    if len(packed) > 10 * 2**20:
        raise ValueError('Output exceeds budget')
    splits = {r.record_id: 'train' for r in records}
    manifest = dict(adapter='uniprot-sequence-annotation-v1', release=release,
        source_path=str(source.resolve()), source_sha256=digest(raw),
        accepted=len(records), rejected=rejected, seed=None,
        graph_nodes=sum(len(r.nodes) for r in records), graph_edges=sum(len(r.edges) for r in records),
        sequence_residues=sum(len(r.nodes[0].measurements[0].value) for r in records),
        leakage_groups=leakage_groups(records), split_assignments=splits,
        split_violations=audit_splits(records, splits),
        split_audit_scope='exact sequence and accession only; train-only; homology audit pending',
        coverage=['protein sequence', 'database recommended name/function/annotation score'],
        missing=['structure', 'homology clusters', 'held-out evaluation', 'full modality coverage'],
        attribution='UniProt Consortium; https://www.uniprot.org/; CC-BY-4.0 https://creativecommons.org/licenses/by/4.0/; scalar projection of original response',
        annotation_policy='database-supplied, not experimental validation; evidence codes retained in source response',
        training_ready=False, smoke_candidate=True, local_output_budget_bytes=20*2**20,
        checksum_files={'records.jsonl': digest(packed)})
    if source.read_bytes() != raw:
        raise RuntimeError('Source changed during curation')
    manifest['source_unchanged'] = True
    output.mkdir(parents=True, exist_ok=False)
    (output / 'records.jsonl').write_bytes(packed)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest
