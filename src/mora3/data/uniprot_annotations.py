"""Lossless UniProt prediction and evidence-bound sequence-generation views.

This reader supports admitted train-only diagnostics, not a neural annotation
objective or family-held-out evaluation. Source/identity metadata never enter
predictor inputs. Complete nested targets remain available without truncation.
"""
import gzip
import hashlib
import io
import json
from dataclasses import dataclass, replace
from pathlib import Path

IDENTITY_KEYS = frozenset({'primaryAccession', 'secondaryAccessions', 'uniProtkbId',
                          'entryType', 'entryAudit', 'annotationScore', 'sequence'})
TASK = 'source_full_UniProt_annotation_prediction'


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode()


@dataclass(frozen=True)
class AnnotationPredictionView:
    accession: str
    sequence: str
    target_bytes: bytes
    provenance_bytes: bytes
    family_group: None = None
    heldout_eligible: bool = False

    def inputs(self):
        return {'canonical_sequence': self.sequence}

    def targets(self):
        return json.loads(self.target_bytes)

    def provenance(self):
        return json.loads(self.provenance_bytes)


def prediction_view(row, entry, *, _generation=False):
    """Validate an envelope against its exact source entry; no inferred joins.

    The internal generation branch shares identical admission/identity checks.
    """
    payload = row['native_payload']
    if hashlib.sha256(payload).hexdigest() != row['native_payload_sha256']:
        raise ValueError('native payload hash mismatch')
    task = json.loads(payload)
    if (task.get('schema') != 'mora3.CorpusTask/v1' or task.get('task') != (GENERATION_TASK if _generation else TASK)
            or row['task_name'] != (GENERATION_TASK if _generation else TASK) or row['native_schema'] != task['schema']):
        raise ValueError('unsupported annotation task')
    receipt = json.loads(row['admission_receipt_json'])
    if (row['admitted_for_task'] is not True or row['split'] != 'train'
            or receipt.get('license') != 'CC-BY-4.0' or receipt.get('reviewed') is not True
            or receipt.get('canonical_sequence_exact_baseline') is not True):
        raise ValueError('train-only admission/rights gate failed')
    seq = entry['sequence']['value']
    sha = hashlib.sha256(seq.encode()).hexdigest()
    identity = task['audit']['identity']
    expected = dict(primary_accession=entry['primaryAccession'], sequence_sha256=sha,
                    sequence_length=len(seq), annotation_score=entry['annotationScore'],
                    entry_version=entry['entryAudit']['entryVersion'],
                    sequence_version=entry['entryAudit']['sequenceVersion'])
    if (identity != expected or row['record_id'] != expected['primary_accession']
            or not seq or entry['sequence']['length'] != len(seq)
            or entry['entryType'] != 'UniProtKB reviewed (Swiss-Prot)'
            or entry['annotationScore'] not in (4, 5)):
        raise ValueError('source identity/sequence mismatch')
    ref = task['audit']['source_ref']
    if (ref != receipt['source_ref'] or ref['source_release'] != '2026_03'
            or ref['raw_file_sha256'] != row['source_sha256']
            or ref['entry_version'] != expected['entry_version']
            or ref['sequence_version'] != expected['sequence_version']):
        raise ValueError('source provenance mismatch')
    groups = json.loads(row['group_ids_json'])
    if not {'accession:' + expected['primary_accession'], 'sequence:' + sha} <= set(groups):
        raise ValueError('exact identity groups missing; no homology inference')
    if not _generation and task['conditions'] != {'canonical_sequence': seq}:
        raise ValueError('prediction inputs must contain only canonical sequence')
    packet = {k: v for k, v in entry.items() if k not in IDENTITY_KEYS}
    properties = []
    for key, kind, unit in [('length', 'sequence_length', 'residue'),
                            ('molWeight', 'sequence_mass', 'Da')]:
        if key in entry['sequence']:
            value = entry['sequence'][key]
            properties.append(dict(parameter_type=kind, value=value, unit=unit,
                evidence_role='provider_computed_from_canonical_sequence',
                source_pointer='/sequence/' + key,
                source_value_json=_canonical(value).decode(), measurement_context=None))
    targets = dict(annotation_packet=packet, provider_computed_sequence_properties=properties)
    if not _generation and task['targets'] != targets:
        raise ValueError('lossless source target mismatch')
    provenance = dict(identity=identity, source_ref=ref, groups=groups,
                      missing=task['missing'], split='train', license=receipt['license'],
                      missing_is_negative=False, GO_coverage='KB_filtered_partial')
    if _generation:
        conditions = generation_conditions(entry)
        if (task['conditions'] != conditions or task['targets'] != {'canonical_sequence': seq}
                or not any(v for k, v in conditions.items() if k != 'organism_taxon_id')):
            raise ValueError('generation source conditions/target mismatch')
        provenance['condition_role'] = 'source_observed_annotations'
        return AnnotationGenerationView(expected['primary_accession'], _canonical(conditions),
                                        _canonical(task['targets']), _canonical(provenance))
    return AnnotationPredictionView(expected['primary_accession'], seq,
                                    _canonical(targets), _canonical(provenance))


def _verified_bytes(root, receipt):
    path = (root / receipt['path']).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('source path outside project')
    raw = path.read_bytes()
    if len(raw) != receipt['bytes'] or hashlib.sha256(raw).hexdigest() != receipt['sha256']:
        raise ValueError('source/shard integrity mismatch')
    return raw


def _authoritative_groups(root, master, shard, rows):
    """Resolve complete, verified task allocation; groups are audit metadata only."""
    import pyarrow.parquet as pq
    source = master['shards'].get(shard['path'])
    if source is None or any(source.get(k) != shard[k] for k in ('path', 'bytes', 'sha256', 'rows')):
        raise ValueError('master/source shard binding mismatch')
    receipt = master['lookup'].get(shard['path'])
    if receipt is None:
        raise ValueError('missing authoritative group sidecar')
    group_rows = pq.read_table(io.BytesIO(_verified_bytes(root, receipt))).to_pylist()
    groups = {r['task_id']: r['effective_groups_json'] for r in group_rows}
    ids = [r['task_id'] for r in rows]
    if (len(group_rows) != receipt['rows'] or len(groups) != len(group_rows)
            or len(set(ids)) != len(ids) or set(groups) != set(ids)):
        raise ValueError('authoritative group/task allocation mismatch')
    for row in rows:
        effective = json.loads(groups[row['task_id']])
        original = json.loads(row['group_ids_json'])
        if (not isinstance(effective, list) or not effective
                or any(not isinstance(g, str) or not g for g in effective)
                or len(set(effective)) != len(effective) or not set(original) <= set(effective)):
            raise ValueError('invalid authoritative groups or original groups lost')
    return groups, receipt


def read_prediction_views(manifest, *, root, limit, _generation=False,
                          authoritative_master=None, master_sha256=None):
    """Read bounded diagnostics from immutable typed snapshots, verifying raw IO.

    Each consumed shard and raw page is verified before use. Only the requested
    task direction is consumed; generation rows fail closed on unsafe conditions.
    """
    import pyarrow.parquet as pq
    if type(limit) is not int or not 1 <= limit <= 1500:
        raise ValueError('explicit diagnostic bound 1..1500 required')
    root = Path(root)
    snapshot_bytes = Path(manifest).read_bytes()
    m = json.loads(snapshot_bytes)
    master = None
    if authoritative_master is not None:
        raw = Path(authoritative_master).read_bytes()
        if master_sha256 is None or hashlib.sha256(raw).hexdigest() != master_sha256:
            raise ValueError('authoritative master integrity mismatch')
        document = json.loads(raw)
        shards = {s['path']: s for s in document['shards']}
        if len(shards) != len(document['shards']):
            raise ValueError('duplicate master source shards')
        master = dict(shards=shards, lookup=document['effective_group_lookup'])
    elif master_sha256 is not None:
        raise ValueError('master hash requires authoritative master')
    if (m.get('source_release') != '2026_03' or m.get('license') != 'CC-BY-4.0'
            or m.get('source_only_train_split') is not True
            or m.get('homology_and_source_parent_heldout_unresolved') is not True):
        raise ValueError('unsupported snapshot admission/split policy')
    pages = {p['path']: p for p in m['source_page_receipts']}
    cached_page = None
    entries = None
    count = 0
    seen = set()
    for shard in m['task_shards']:
        rows = pq.read_table(io.BytesIO(_verified_bytes(root, shard))).to_pylist()
        if len(rows) != shard['rows'] or shard['split'] != 'train':
            raise ValueError('shard allocation mismatch')
        resolved = None
        if master is not None:
            resolved, group_receipt = _authoritative_groups(root, master, shard, rows)
        for row in rows:
            if row['task_name'] != (GENERATION_TASK if _generation else TASK):
                continue
            task = json.loads(row['native_payload'])
            ref = task['audit']['source_ref']
            page = pages[ref['raw_file']]
            if ref['raw_file_sha256'] != page['sha256'] or page['source_release'] != m['source_release']:
                raise ValueError('raw page binding mismatch')
            if cached_page != ref['raw_file']:
                raw = _verified_bytes(root, page)
                body = gzip.decompress(raw) if page['compression'] == 'gzip' else raw
                entries = json.loads(body)['results']
                if len(entries) != page['entries']:
                    raise ValueError('raw page entry count mismatch')
                cached_page = ref['raw_file']
            index = ref['results_index']
            if type(index) is not int or not 0 <= index < len(entries):
                raise ValueError('invalid source index')
            view = prediction_view(row, entries[index], _generation=_generation)
            if resolved is not None:
                provenance = view.provenance()
                provenance['source_groups'] = provenance['groups']
                provenance['groups'] = json.loads(resolved[row['task_id']])
                provenance['authoritative_groups'] = dict(
                    master_sha256=master_sha256,
                    snapshot_sha256=hashlib.sha256(snapshot_bytes).hexdigest(),
                    source_shard=shard, group_sidecar=group_receipt,
                    task_id=row['task_id'],
                    scope='audit_only_detected_guards_not_exhaustive_homology')
                view = replace(view, provenance_bytes=_canonical(provenance))
            if view.accession in seen:
                raise ValueError('duplicate prediction accession')
            seen.add(view.accession)
            yield view
            count += 1
            if count == limit:
                return

GENERATION_TASK = 'source_UniProt_annotation_conditioned_canonical_sequence'


def generation_conditions(entry):
    """Reconstruct canonical-scope, assertion-local source conditions.

    No accession, residue labels, sequence descriptors or evidence reference IDs
    are model inputs. Evidence remains in the immutable raw source binding.
    """
    import math
    import re
    def direct(obj):
        return any(x.get('evidenceCode') == 'ECO:0000269'
                   for x in obj.get('evidences', []) if isinstance(x, dict))
    functions, positive, negative, ecs, kinetics = [], [], [], [], []
    def ec_walk(obj):
        if isinstance(obj, dict):
            for k, value in obj.items():
                if k == 'ecNumbers' and isinstance(value, list):
                    ecs.extend(str(x['value']) for x in value
                               if isinstance(x, dict) and 'value' in x and direct(x))
                elif k == 'ecNumber':
                    if isinstance(value, dict) and value.get('value') is not None and direct(value):
                        ecs.append(str(value['value']))
                else:
                    ec_walk(value)
        elif isinstance(obj, list):
            for value in obj:
                ec_walk(value)
    description = entry.get('proteinDescription', {})
    for key in ('recommendedName', 'alternativeNames'):
        ec_walk(description.get(key, {}))
    for c in entry.get('comments', []):
        if c.get('molecule'):
            continue
        ec_walk(c)
        if c.get('commentType') == 'FUNCTION':
            for t in c.get('texts', []):
                value = t.get('value')
                if (direct(t) and isinstance(value, str) and value
                        and not re.search(r'\b[ACDEFGHIKLMNPQRSTVWYOU]{8,}\b', value)):
                    functions.append(value)
        if c.get('commentType') == 'BIOPHYSICOCHEMICAL PROPERTIES':
            for key, kind, field in [('michaelisConstants', 'MichaelisConstant', 'constant'),
                                     ('maximumVelocities', 'maximumVelocity', 'velocity')]:
                for x in c.get('kineticParameters', {}).get(key, []):
                    value, unit = x.get(field), x.get('unit')
                    if direct(x) and type(value) in (int, float) and math.isfinite(value) and isinstance(unit, str) and unit:
                        kinetics.append(dict(parameter_type=kind, source_value=value,
                            source_unit=unit, substrate_label=x.get('substrate'),
                            assay_pH=None, assay_temperature=None))
    for x in entry.get('uniProtKBCrossReferences', []):
        if x.get('database') != 'GO' or x.get('isoformId'):
            continue
        props = {p['key']: p.get('value') for p in x.get('properties', []) if 'key' in p}
        evidence = props.get('GoEvidenceType')
        if isinstance(evidence, str) and evidence.split(':', 1)[0] in {'EXP','IDA','IPI','IMP','IGI','IEP','HTP','HDA','HMP','HGI','HEP'}:
            qualifier = props.get('Qualifier', props.get('GoAnnotationQualifier'))
            (negative if isinstance(qualifier, str) and 'NOT' in qualifier.split('|') else positive).append(x.get('id'))
    return dict(source_experimentally_supported_function_texts=sorted(set(functions)),
                source_experimental_associated_GO_ids=sorted(set(positive)),
                source_explicit_experimental_NOT_GO_ids=sorted(set(negative)),
                source_experimentally_supported_EC_ids=sorted(set(ecs)),
                source_kinetic_conditions=kinetics,
                organism_taxon_id=entry.get('organism', {}).get('taxonId'))


@dataclass(frozen=True)
class AnnotationGenerationView:
    accession: str
    condition_bytes: bytes
    target_bytes: bytes
    provenance_bytes: bytes
    family_group: None = None
    heldout_eligible: bool = False

    def inputs(self):
        return json.loads(self.condition_bytes)

    def targets(self):
        return json.loads(self.target_bytes)

    def provenance(self):
        return json.loads(self.provenance_bytes)


def generation_view(row, entry):
    return prediction_view(row, entry, _generation=True)


def read_generation_views(manifest, *, root, limit, authoritative_master=None,
                          master_sha256=None):
    return read_prediction_views(manifest, root=root, limit=limit, _generation=True,
        authoritative_master=authoritative_master, master_sha256=master_sha256)
