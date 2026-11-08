"""Exact raw GOA packet verification, before sequence-source/admission binding.

This layer never grants training eligibility. Authoritative corpus groups and
independent KB/form sequence verification remain required by a native reader.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from .uniprot_annotations import _canonical


@dataclass(frozen=True)
class VerifiedGOAPacket:
    condition_bytes: bytes
    target_bytes: bytes
    provenance_bytes: bytes

    def inputs(self):return json.loads(self.condition_bytes)
    def targets(self):return json.loads(self.target_bytes)
    def provenance(self):return json.loads(self.provenance_bytes)


def verify_goa_packet(row, *, root):
    """Verify source assertion multiset and exact-subject input isolation.

    Reads only explicitly referenced JSON under root; no source discovery. The
    sequence digest checks envelope consistency, not original KB sequence truth.
    """
    raw=row['native_payload']
    if hashlib.sha256(raw).hexdigest()!=row['native_payload_sha256']:
        raise ValueError('payload hash mismatch')
    d=json.loads(raw);c=d['conditions'];t=d['targets'];a=d['audit'];b=a['binding']
    receipt=json.loads(row['admission_receipt_json']);subject=b['subject']
    if (d.get('schema')!='mora3.CorpusTask/v1' or row['native_schema']!=d['schema']
            or d.get('task')!='source_GOA_assertion_packet_prediction' or row['task_name']!=d['task']
            or row['record_id']!=subject or not subject.startswith('UniProtKB:')
            or b.get('canonical_labels_inherited') is not False
            or row['split']!='train' or row['admitted_for_task'] is not True
            or receipt.get('license')!='CC-BY-4.0'
            or receipt.get('qualified_parent_entry') is not True
            or receipt.get('canonical_labels_inherited') is not False
            or receipt.get('semantic_not_residue_or_experimental_response_task') is not True):
        raise ValueError('GOA exact subject/train-only rights mismatch')
    if set(c)!={'protein_sequence'} or not isinstance(c['protein_sequence'],str) or not c['protein_sequence']:
        raise ValueError('sequence-only conditions required')
    seq=c['protein_sequence'];sha=hashlib.sha256(seq.encode()).hexdigest()
    groups=json.loads(row['group_ids_json'])
    if row['full_axis_values']!=len(seq) or not {'accession:'+b['parent_accession'],'sequence:'+sha}<=set(groups):
        raise ValueError('sequence envelope mismatch')
    if set(t)!={'source_GO_annotations','typed_GO_annotations'}:
        raise ValueError('complete raw and typed targets required')
    root=Path(root).resolve();cache={};sources=[];assertions=[]
    for ref in a['source_refs']:
        path=(root/ref['raw_file']).resolve()
        if not path.is_relative_to(root):raise ValueError('source outside project')
        if path not in cache:
            payload=path.read_bytes();digest=hashlib.sha256(payload).hexdigest()
            cache[path]=(digest,json.loads(payload))
        digest,source=cache[path]
        if digest!=ref['raw_sha256']:raise ValueError('raw source hash mismatch')
        index=ref['results_index']
        if type(index) is not int or not 0<=index<len(source['results']):
            raise ValueError('raw source index mismatch')
        assertion=source['results'][index]
        if assertion.get('geneProductId')!=subject:raise ValueError('raw subject mismatch')
        assertions.append(assertion);sources.append(dict(ref))
    if not assertions or sorted(map(_canonical,assertions))!=sorted(map(_canonical,t['source_GO_annotations'])):
        raise ValueError('raw assertion multiset mismatch')
    provenance=dict(subject=subject,source_refs=sources,source_audit=a,
                    envelope={k:v for k,v in row.items() if k!='native_payload'},
                    sequence_source_verified=False,training_eligibility_granted=False,
                    heldout_eligible=False,paired_cellular_observation=False)
    return VerifiedGOAPacket(_canonical(c),_canonical(t),_canonical(provenance))


def bind_canonical_sequence(packet, *, root, page_receipt, results_index):
    """Check current canonical KB sequence independently of the GOA envelope.

    Explicitly excludes forms: their exact coordinate bindings need a separate
    reader. This verifies sequence identity, never historical GOA sequence
    equivalence, corpus admission, experimental status or generation eligibility.
    """
    import gzip
    root = Path(root).resolve()
    path = (root / page_receipt['path']).resolve()
    if not path.is_relative_to(root):
        raise ValueError('KB source outside project')
    if page_receipt.get('source_release') != '2026_03':
        raise ValueError('KB release mismatch')
    raw = path.read_bytes()
    if (len(raw) != page_receipt['bytes'] or
            hashlib.sha256(raw).hexdigest() != page_receipt['sha256']):
        raise ValueError('KB source integrity mismatch')
    compression = page_receipt.get('compression')
    if compression not in ('none', 'gzip'):
        raise ValueError('unsupported KB compression')
    entries = json.loads(gzip.decompress(raw) if compression == 'gzip' else raw)['results']
    if (len(entries) != page_receipt['entries'] or type(results_index) is not int
            or not 0 <= results_index < len(entries)):
        raise ValueError('KB source allocation mismatch')
    entry = entries[results_index]
    provenance = packet.provenance()
    accession = entry['primaryAccession']
    sequence = entry['sequence']
    if (provenance['subject'] != 'UniProtKB:' + accession
            or provenance['source_audit']['binding']['parent_accession'] != accession
            or sequence['value'] != packet.inputs()['protein_sequence']
            or sequence['length'] != len(sequence['value'])):
        raise ValueError('exact canonical KB sequence mismatch')
    provenance.update(sequence_source_verified=True,
        canonical_sequence_source=dict(page_receipt=page_receipt, results_index=results_index),
        historical_GOA_sequence_equivalence_verified=False)
    return VerifiedGOAPacket(packet.condition_bytes, packet.target_bytes, _canonical(provenance))


def read_group_bound_goa_packets(master_path, *, root, master_sha256,
                                 shard_path, limit):
    """Read one explicit GOA shard with authoritative audit-only leakage groups.

    Verifies the entire shard/task allocation before yielding a bounded slice.
    This is not final native training admission: independent sequence, form,
    ontology and objective applicability checks remain necessary. Master versions
    are pinned by caller hash, never silently discovered or upgraded.
    """
    import io
    import pyarrow.parquet as pq
    from .uniprot_annotations import _verified_bytes, _authoritative_groups
    if type(limit) is not int or not 1 <= limit <= 1500:
        raise ValueError('explicit diagnostic bound 1..1500 required')
    root = Path(root).resolve()
    path = (root / master_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError('master outside project')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != master_sha256:
        raise ValueError('authoritative master integrity mismatch')
    document = json.loads(raw)
    shards = {s['path']: s for s in document['shards']}
    if len(shards) != len(document['shards']):
        raise ValueError('duplicate master source shards')
    if shard_path not in shards:
        raise ValueError('source shard absent from master')
    shard = shards[shard_path]
    rows = pq.read_table(io.BytesIO(_verified_bytes(root, shard))).to_pylist()
    if len(rows) != shard['rows']:
        raise ValueError('source shard row count mismatch')
    groups, receipt = _authoritative_groups(root,
        dict(shards=shards, lookup=document['effective_group_lookup']), shard, rows)
    for row in rows[:limit]:
        packet = verify_goa_packet(row, root=root)
        provenance = packet.provenance()
        provenance.update(source_groups=json.loads(row['group_ids_json']),
            groups=json.loads(groups[row['task_id']]),
            authoritative_groups=dict(master_sha256=master_sha256,
                source_shard=shard, group_sidecar=receipt, task_id=row['task_id'],
                scope='audit_only_detected_guards_not_exhaustive_homology'))
        yield VerifiedGOAPacket(packet.condition_bytes, packet.target_bytes,
                                _canonical(provenance))


@dataclass(frozen=True)
class ApplicableGOAPacket:
    """Raw packet plus immutable full-axis supervision; no predictor metadata."""
    packet: VerifiedGOAPacket
    axis_ids: tuple
    labels: tuple
    observed: tuple
    status: str


def bind_goa_applicability(packet, *, root, registry, ontology_manifest,
                           ontology_sha256):
    """Compose verified canonical packets with pinned source axes/reference.

    Current ontology is separate reference metadata: no historical ID rewriting,
    ancestor propagation or added negatives. Conflicting coarse term polarities
    quarantine the entire packet explicitly. This opt-in policy never changes
    historical trainers; native scheduler adoption/recovery is a separate gate.
    Call with packets from the source/group reader and canonical sequence binder.
    """
    import io
    import pyarrow.parquet as pq
    from .uniprot_annotations import _verified_bytes
    from mora3.model.uniprot_go import goa_targets
    provenance = packet.provenance()
    if (provenance.get('sequence_source_verified') is not True
            or not provenance.get('authoritative_groups')
            or not provenance.get('groups')
            or provenance['subject'] != 'UniProtKB:' + provenance['source_audit']['binding']['parent_accession']):
        raise ValueError('verified canonical sequence and authoritative groups required')
    root = Path(root).resolve()
    path = (root / ontology_manifest).resolve()
    if not path.is_relative_to(root):
        raise ValueError('ontology manifest outside project')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ontology_sha256:
        raise ValueError('ontology manifest integrity mismatch')
    document = json.loads(raw)
    if (document.get('schema') != 'mora3.GOCurrentReferenceAxisBinding/v1'
            or document.get('same_snapshot') is not False
            or any(document.get(k) is not True for k in (
                'no_automatic_obsolete_or_alt_ID_remapping',
                'no_GO_ancestry_label_propagation','no_dense_negative_labels'))):
        raise ValueError('ontology reference policy mismatch')
    axes = [r for r in registry.binding()['axis_files'] if r['axis'] == 'GO']
    source = document['source_axis']
    if len(axes) != 1 or any(axes[0][k] != source[k] for k in ('path','bytes','sha256')):
        raise ValueError('ontology/registry exact source axis mismatch')
    axis = registry.axis('GO')
    axis_rows = pq.read_table(io.BytesIO(_verified_bytes(root, source))).to_pylist()
    binding = document['binding']
    rows = pq.read_table(io.BytesIO(_verified_bytes(root, binding))).to_pylist()
    if (tuple(r['axis_id'] for r in axis_rows) != axis
            or tuple(r['axis_id'] for r in rows) != axis or len(rows) != binding['rows']
            or any(r.get('raw_annotation_ID_not_remapped') is not True
                   or r.get('annotation_source_versions_separate') is not True for r in rows)):
        raise ValueError('complete ordered ontology axis binding required')
    status = 'applicable'
    try:
        y, mask = goa_targets([packet.targets()['source_GO_annotations']], axis,
                             subjects=[provenance['subject']])
        labels = tuple(y[0].tolist()); observed = tuple(mask[0].tolist())
    except ValueError as exc:
        if str(exc) != 'conflicting source GOA polarity':
            raise
        status = 'quarantined_polarity_conflict'
        labels = (0.,) * len(axis); observed = (False,) * len(axis)
    provenance.update(goa_objective=dict(
        policy='exact-subject-coarse-GOA-observed-BCE-v1',
        registry=registry.binding(), ontology_manifest_sha256=ontology_sha256,
        ontology_binding=binding, ontology_version=document['ontology_version'],
        ontology_timestamp=document['ontology_timestamp'], same_snapshot=False,
        status=status, unknown_is_negative=False, extensions_supported=False,
        forms_supported=False, ontology_propagation=False), generation_eligible=False)
    result = VerifiedGOAPacket(packet.condition_bytes, packet.target_bytes,
                               _canonical(provenance))
    return ApplicableGOAPacket(result, axis, labels, observed, status)
