"""Reviewed, same-snapshot admission for a finite MCSA source-list component.

The expected review-packet digest is a trusted coordinator/run-configuration
input. Hashes establish integrity, not an external signature or blanket research
approval. Constructing a boolean or changing a candidate flag never qualifies a
batch. This token pins and reconstructs actual native rows; it does not issue a
full-project training, optimizer, checkpoint, or model-family certificate.
"""
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path
import re
from types import MappingProxyType

from .mcsa_weak_sites import (WeakRoleVocabulary, WeakSiteBatch, SourceListSupervision,
    parse_weak_site_candidate, collate_weak_site_candidates)

FAMILY = 'MCSA-source-role-list-current-reference'
TASK = 'source_role_list_reconstruction_on_current_reference'
NATIVE = 'Mora3.MCSAWeakReferenceSiteCandidate/v1'
_SEMANTICS = dict(schema='Mora3.MCSASourceRoleListModelBinding/v1', source_family=FAMILY,
    task_name=TASK, native_schema=NATIVE, condition_inputs=['canonical_sequences'],
    target_history='teacher-forced-ordered-role-prefix-BOS-input-only',
    targets='observed-ordered-source-role-list-including-EOS',
    source_position='output-selector-only-never-encoder-marker', unknown_positions='no-target-no-loss',
    biological_negatives=False,
    encoder_sequence_contract='complete-native-sequence-no-crop-or-truncation')


def _sha(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def _wire(value):
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
            separators=(',', ':')).encode()
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError('literal finite JSON values required') from exc


def _pairs(values):
    result = {}
    for key, value in values:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def _float(text):
    value = float(text)
    if not math.isfinite(value):
        raise ValueError('nonfinite JSON number')
    return value


def _json(raw):
    try:
        return json.loads(raw, object_pairs_hook=_pairs, parse_float=_float,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON literal')))
    except (TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError('strict UTF8 JSON required') from exc


def _identity(path):
    stat = path.stat()
    return tuple(getattr(stat, key) for key in ('st_dev', 'st_ino', 'st_mode', 'st_nlink',
        'st_uid', 'st_gid', 'st_size', 'st_mtime_ns', 'st_ctime_ns'))


def _read(reference, identities):
    if (type(reference) is not dict or type(reference.get('path')) is not str or not reference['path']
            or type(reference.get('bytes')) is not int or reference['bytes'] <= 0 or not _sha(reference.get('sha256'))):
        raise ValueError('exact path/positive byte length/SHA256 reference required')
    path = Path(reference['path'])
    before = _identity(path)
    raw = path.read_bytes()
    if (_identity(path) != before or len(raw) != reference['bytes']
            or hashlib.sha256(raw).hexdigest() != reference['sha256']):
        raise ValueError('pinned source artifact changed or SHA/size mismatch')
    identities[str(path)] = before
    return raw


def _same(left, right, message):
    if _wire(left) != _wire(right):
        raise ValueError(message)


def _live_code_hashes():
    data = Path(__file__).resolve().parent
    return dict(model_source_sha256=hashlib.sha256((data.parent / 'model/mcsa_role_list.py').read_bytes()).hexdigest(),
        residue_source_sha256=hashlib.sha256((data.parent / 'model/residue_objective.py').read_bytes()).hexdigest(),
        reader_source_sha256=hashlib.sha256((data / 'mcsa_weak_sites.py').read_bytes()).hexdigest())


def _complete_sequence_stream_length(sequence, tokenizer):
    """Exact passive graph-codec entity-stream length, with no tensor/model."""
    from mora3.data.schema import GraphRecord, Node, Measurement
    from mora3.data.provenance import Provenance
    from mora3.tokenization.graph_payload import GraphPayloadCodec
    raw = sequence.encode('ascii')
    digest = hashlib.sha256(raw).hexdigest()
    record = GraphRecord('provided-sequence-condition:' + digest,
        (Node('protein', (Measurement('sequence', sequence, None, 'observed', True),)),), (),
        (Provenance('provided-sequence-condition', 'request-content-sha256-v1',
            digest, digest, 'no-rights-inferred'),))
    if tokenizer.decode(tokenizer.encode(raw)) != raw:
        raise ValueError('complete source sequence cannot roundtrip fitted byte vocabulary')
    lengths = []
    for entity in GraphPayloadCodec(tokenizer).encode(record)['entities']:
        length = len(_wire({'kind': entity['kind'], 'directed': entity['directed']}))
        for field in sorted(entity['measurements'], key=lambda f: f['name']):
            length += 2 + len(_wire({key: field[key] for key in ('name', 'units', 'status', 'value_type')}))
            length += len(field['token_ids'])
        lengths.append(length)
    if len(lengths) != 1:
        raise ValueError('exact single protein sequence condition entity required')
    return lengths[0]


class MCSARoleListAdmission:
    """Immutable compiled source rows from one explicitly reviewed packet."""
    __slots__ = ('_packet_sha256', '_binding', '_axis', '_prediction_classes', '_input_classes',
        '_records', '_identities', '_implementation_sha256')

    def __init__(self, *args, **kwargs):
        raise TypeError('use from_review_packet with a trusted expected coordinator digest')

    def __setattr__(self, name, value):
        raise AttributeError('reviewed source admission is immutable')

    @property
    def review_packet_sha256(self):
        return self._packet_sha256

    @classmethod
    def from_review_packet(cls, path, expected_sha256, *, vocabulary):
        if cls is not MCSARoleListAdmission or type(vocabulary) is not WeakRoleVocabulary or not _sha(expected_sha256):
            raise ValueError('exact admission class, frozen role vocabulary and trusted review SHA required')
        path = Path(path)
        identities = {}
        packet_identity = _identity(path)
        raw = path.read_bytes()
        if _identity(path) != packet_identity or hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError('review packet SHA256 mismatch')
        packet = _json(raw)
        if (type(packet) is not dict or packet.get('schema') != 'Mora3.MCSARoleListReviewedSourceAdmission/v1'
                or packet.get('review_state') != 'accepted-scoped-source'
                or type(packet.get('coordinator_review')) is not dict
                or packet['coordinator_review'].get('owner') != 'root-coordinator'):
            raise ValueError('explicit accepted coordinator source review required')
        implementation_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if packet.get('admission_implementation_sha256') != implementation_sha:
            raise ValueError('review did not bind the live admission implementation')
        binding = packet.get('approved_model_binding')
        if type(binding) is not dict:
            raise ValueError('exact reviewed semantic/model binding required')
        required = set(_SEMANTICS) | {'axis_sha256', 'max_target_length', 'payload_vocabulary_sha256',
            'model_source_sha256', 'residue_source_sha256', 'reader_source_sha256', 'payload_max_tokens'}
        if set(binding) != required:
            raise ValueError('literal complete source task/model binding required')
        for key, value in _SEMANTICS.items():
            _same(binding[key], value, 'source task semantic binding differs')
        if (binding['axis_sha256'] != vocabulary.axis_sha256
                or type(binding['max_target_length']) is not int or binding['max_target_length'] < 1
                or not _sha(binding['payload_vocabulary_sha256'])
                or type(binding['payload_max_tokens']) is not int or binding['payload_max_tokens'] < 1):
            raise ValueError('frozen source axis/finite list length/payload vocabulary and positive complete-sequence bound required')
        _same({key: binding[key] for key in _live_code_hashes()}, _live_code_hashes(), 'live model/residue/reader code differs from review')
        payload_ref = packet.get('payload_vocabulary_artifact')
        payload_document = _json(_read(payload_ref, identities))
        if payload_ref['sha256'] != binding['payload_vocabulary_sha256']:
            raise ValueError('actual fitted payload artifact SHA differs from model binding')
        fit_receipt = _json(_read(packet.get('payload_fit_receipt'), identities))
        if (fit_receipt.get('schema') != 'Mora3.MCSACanonicalConditionPayloadFit/v1'
                or fit_receipt.get('status') != 'complete-source-fit'
                or fit_receipt.get('fit_fields') != ['conditions.canonical_sequence']
                or fit_receipt.get('target_fields_used') is not False
                or fit_receipt.get('heldout_rows_used') is not False):
            raise ValueError('completed condition-only source payload fit required')
        _same(fit_receipt.get('payload_vocabulary_artifact'), payload_ref, 'fit receipt and actual payload artifact differ')
        fit_input_ref = fit_receipt.get('fit_input')
        fit_inputs = [_json(line) for line in _read(fit_input_ref, identities).splitlines()]
        underlying_fit = _json(_read(fit_receipt.get('underlying_fit_receipt'), identities))
        if (underlying_fit.get('schema') != 'Mora3.SourceConditionTokenizerFitReceipt/v1'
                or underlying_fit.get('completed') is not True
                or underlying_fit.get('artifact_sha256') != payload_ref['sha256']
                or underlying_fit.get('artifact_bytes') != payload_ref['bytes']
                or underlying_fit.get('independent_all602_shortest_path_and_lossless_roundtrip') is not True
                or underlying_fit.get('input_and_code_unchanged') is not True):
            raise ValueError('actual completed fit artifact and independent DP/roundtrip receipt required')
        fit_scope = payload_document.get('fit_scope')
        if (type(fit_scope) is not dict or fit_scope.get('input_sha256') != fit_input_ref['sha256']
                or fit_scope.get('allcorpus_fit') is not False or fit_scope.get('multilingual_fit') is not False
                or fit_scope.get('universal_byte_coverage') is not True):
            raise ValueError('precise source fit scope/byte coverage required; no broader fit inference')
        _same(underlying_fit.get('source_fit_scope'), fit_scope, 'payload artifact and actual fit scope differ')
        review = packet['coordinator_review']
        if review.get('approved_model_binding_sha256') != hashlib.sha256(_wire(binding)).hexdigest():
            raise ValueError('coordinator model semantic digest differs')
        component_ref = packet.get('component_manifest')
        component = _json(_read(component_ref, identities))
        _same(fit_receipt.get('source_component_manifest'), component_ref, 'payload fit and source component snapshots differ')
        mask_ref = packet.get('final_mask_manifest')
        mask = _json(_read(mask_ref, identities))
        proof_ref = packet.get('final_mask_component_verification')
        proof = _json(_read(proof_ref, identities))
        _same(proof.get('component_manifest'), component_ref, 'component and independent proof are different snapshots')
        _same(proof.get('final_mask_manifest'), mask_ref, 'mask and independent proof are different snapshots')
        _same(proof.get('exact_parent_master'), mask.get('parent_master'), 'parent-master snapshot differs')
        if (proof.get('exact_final_mask_complete_both_directions') is not True
                or proof.get('all_candidate_payload_bytes_task_IDs_full_axes_and_aligned_groups_equal') is not True
                or proof.get('all_candidate_component_guards_role_disjoint') is not True):
            raise ValueError('complete independent mask/native-row/group proof required')
        _read(mask['parent_master'], identities)
        if (component.get('source_family') != FAMILY or component.get('task_name') != TASK
                or component.get('native_schema') != NATIVE
                or component.get('scientific_task_admission') is not False
                or component.get('model_adapter_supported') is not False):
            raise ValueError('exact nonrelabelled native candidate component required')
        if (review.get('component_manifest_sha256') != component_ref['sha256']
                or review.get('final_mask_manifest_sha256') != mask_ref['sha256']
                or review.get('independent_verification_sha256') != proof_ref['sha256']):
            raise ValueError('coordinator source/mask/proof review differs')
        # Pin both closure endpoints: source roles must not be a newly invented
        # or independently relabelled split relative to the reviewed component.
        closure_ref = component['current_protein_gene_guard_closure']
        closure = _json(_read(closure_ref, identities))
        closure_check = _json(_read(component['independent_current_closure'], identities))
        _same(closure_check.get('producer_manifest'), closure_ref, 'current closure proof snapshot differs')
        if closure_check.get('all1003_MCSA_source_components_and_roles_unchanged') is not True:
            raise ValueError('reviewed MCSA source roles must have closed native guards')
        _same(mask.get('current_protein_gene_closure'), closure_ref, 'component and mask guard closure differ')
        vocab_ref = component['source_vocabulary']
        vocab_document = _json(_read(vocab_ref, identities))
        _same(WeakRoleVocabulary.from_document(vocab_document).roles, vocabulary.roles, 'reviewed role vocabulary differs')
        candidate_ref = component['source_candidate_file']
        _same(vocab_document.get('source_candidate_file'), candidate_ref, 'role axis and candidate source snapshots differ')
        native_manifest = _json(_read(component['source_native_manifest'], identities))
        _same(native_manifest.get('source_candidate_file'), candidate_ref, 'native source and component candidate snapshots differ')
        _same(native_manifest.get('vocabulary'), vocab_ref, 'native source and component vocabulary snapshots differ')
        candidates = {}
        native_bytes = {}
        for line in _read(candidate_ref, identities).splitlines():
            native = _json(line)
            record = parse_weak_site_candidate(native, vocabulary)
            rid = record.audit['record_id']
            if rid in candidates:
                raise ValueError('duplicate canonical candidate record')
            candidates[rid] = record
            native_bytes[rid] = line
        if (not candidates or len(candidates) != component.get('source_entry_views')
                or len(candidates) != proof.get('source_native_candidate_records')):
            raise ValueError('complete selected native candidate cohort required')
        import pyarrow.parquet as pq
        sidecars = {}
        for item in component['group_sidecars']:
            key = item['source_shard']['path']
            if key in sidecars:
                raise ValueError('duplicate source-shard group-sidecar binding')
            sidecars[key] = item
        seen, groups = set(), {}
        split_counts, site_counts, symbol_counts = Counter(), Counter(), Counter()
        fit_coordinates = {}
        for shard in component['shards']:
            rows = pq.read_table(io.BytesIO(_read(shard, identities))).to_pylist()
            _same(sidecars[shard['path']]['source_shard'], shard, 'aligned group sidecar binds a different source shard')
            aligned = pq.read_table(io.BytesIO(_read(sidecars[shard['path']]['authoritative_sidecar'], identities))).to_pylist()
            if len(rows) != shard['rows'] or len(aligned) != len(rows):
                raise ValueError('complete native shard/group row conservation required')
            for source_ordinal, (row, group) in enumerate(zip(rows, aligned)):
                rid = row['record_id']
                if rid not in candidates or rid in seen:
                    raise ValueError('native shard candidate coverage/uniqueness differs')
                seen.add(rid)
                record = candidates[rid]
                payload = row['native_payload']
                if (payload != native_bytes[rid] or hashlib.sha256(payload).hexdigest() != row['native_payload_sha256']
                        or row['task_id'] != hashlib.sha256(b'Mora3.MCSA.source-role-list/v1\0' + payload).hexdigest()
                        or row['source_family'] != FAMILY or row['task_name'] != TASK or row['native_schema'] != NATIVE
                        or row['admitted_for_task'] is not False or row['model_adapter_supported'] is not False
                        or row['split'] != record.audit['proposed_split'] or row['split'] != shard['split']
                        or group['task_id'] != row['task_id'] or group['effective_groups_json'] != row['group_ids_json']):
                    raise ValueError('exact candidate bytes/identity/role/aligned groups required; flags cannot promote')
                group_keys = _json(row['group_ids_json'])
                if type(group_keys) is not list or not group_keys or any(type(key) is not str or not key for key in group_keys):
                    raise ValueError('nonempty exact conservative group keys required')
                for key in group_keys:
                    if key in groups and groups[key] != row['split']:
                        raise ValueError('reviewed guard crosses source split')
                    groups[key] = row['split']
                fit_coordinates[rid] = dict(source_record_id=rid, source_task_id=row['task_id'],
                    source_native_payload_sha256=row['native_payload_sha256'], source_shard=shard,
                    source_shard_row_ordinal=source_ordinal, original_split=row['split'],
                    canonical_sequence_sha256=hashlib.sha256(record.canonical_sequence.encode()).hexdigest(),
                    current_sequence_provenance=record.audit['current_uniprot_provenance'])
                single = collate_weak_site_candidates([record], vocabulary)
                split_counts[row['split']] += 1
                site_counts[row['split']] += len(single.source_list_supervision)
                symbol_counts[row['split']] += sum(len(target.target_ids) for target in single.source_list_supervision)
        if seen != set(candidates):
            raise ValueError('missing selected candidate shard records')
        _same(dict(split_counts), component['source_splits'], 'source split count differs')
        _same(dict(site_counts), component['sparse_mapping_views_by_split'], 'sparse target count differs')
        _same(dict(symbol_counts), component['role_EOS_symbols_by_split'], 'complete ordered role/EOS count differs')
        _same(dict(split_counts), proof['component_split_counts'], 'independent source role count differs')
        # Reconstruct the fit cohort from actual native train rows, including
        # repeated sequences as distinct source views; no heldout/target export.
        expected_train = {rid for rid, record in candidates.items() if record.audit['proposed_split'] == 'train'}
        fit_seen = set()
        fit_characters = 0
        fit_bytes = 0
        for item in fit_inputs:
            if type(item) is not dict or set(item) != {'conditions', 'audit'} or type(item['audit']) is not dict:
                raise ValueError('condition-only fit export with separate exact audit coordinates required')
            rid = item['audit'].get('source_record_id')
            if rid not in expected_train or rid in fit_seen:
                raise ValueError('complete unique train-source fit views required; heldouts cannot fit')
            record = candidates[rid]
            _same(item['conditions'], record.condition_inputs(), 'fit conditions differ from complete canonical train sequence')
            _same(item['audit'], fit_coordinates[rid], 'fit source provenance/file/row/payload coordinates differ')
            fit_seen.add(rid)
            fit_characters += len(record.canonical_sequence)
            fit_bytes += len(record.canonical_sequence.encode())
        if (fit_seen != expected_train or type(fit_receipt.get('train_views')) is not int
                or fit_receipt['train_views'] != len(expected_train)
                or fit_receipt.get('full_canonical_sequence_characters') != fit_characters
                or type(underlying_fit.get('all_source_records')) is not int
                or underlying_fit['all_source_records'] != len(expected_train)
                or underlying_fit.get('all_source_bytes') != fit_bytes):
            raise ValueError('actual complete source fit count/full sequence bytes differ from receipt')
        # Load the exact mask once, so a receipt flag alone cannot create a token.
        mask_rows = pq.read_table(io.BytesIO(_read(mask['mask'], identities))).to_pylist()
        coordinates = set()
        for row in mask_rows:
            if type(row.get('protected')) is not bool or type(row.get('unknown')) is not bool:
                raise ValueError('exact typed mask protection and unknown dispositions required')
            coordinate = row['source_shard'], row['source_row_ordinal']
            if coordinate in coordinates:
                raise ValueError('duplicate reviewed mask coordinate')
            coordinates.add(coordinate)
        if (len(mask_rows) != proof['mask_taskviews'] or len(mask_rows) != mask['combined_mask_taskviews']
                or sum(row['protected'] is True for row in mask_rows) != proof['protected_reference_exclusions']
                or sum(row['protected'] is False for row in mask_rows) != proof['unknown_reference_quarantines']):
            raise ValueError('actual source mask differs from same-snapshot proof')
        # Tokenization is validation of this frozen fit, never a new fit on
        # heldout sequences. Check every complete source condition, including
        # exact graph metadata/field boundaries consumed by the encoder.
        from mora3.tokenization.shortest_path import ByteTokenizer
        try:
            payload_tokenizer = ByteTokenizer(tuple(bytes.fromhex(value)
                for value in payload_document['learned_hex']), controls=payload_document['controls'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('complete byte-token payload vocabulary required') from exc
        compiled = {}
        for rid, record in candidates.items():
            if _complete_sequence_stream_length(record.canonical_sequence, payload_tokenizer) > binding['payload_max_tokens']:
                raise ValueError('reviewed payload bound would reject a complete source sequence')
            single = collate_weak_site_candidates([record], vocabulary)
            target_rows = tuple((target.position_0based, target.decoder_input_ids, target.target_ids,
                target.source_residue_ordinal, target.source_mapping_ordinal) for target in single.source_list_supervision)
            if max(len(target[2]) for target in target_rows) > binding['max_target_length']:
                raise ValueError('reviewed model would truncate a complete source role list')
            compiled[rid] = (record.canonical_sequence, record.audit['proposed_split'], _wire(record.audit), target_rows)
        if _identity(path) != packet_identity:
            raise ValueError('review packet changed during source compilation')
        identities[str(path)] = packet_identity
        self = object.__new__(cls)
        for key, value in dict(_packet_sha256=expected_sha256, _binding=_wire(binding), _axis=vocabulary.axis_sha256,
                _prediction_classes=vocabulary.prediction_class_count, _input_classes=vocabulary.decoder_input_class_count,
                _records=MappingProxyType(compiled), _identities=MappingProxyType(identities),
                _implementation_sha256=implementation_sha).items():
            object.__setattr__(self, key, value)
        return self

    def require_batch(self, batch, *, model_binding, split):
        if any(not hasattr(self, name) for name in self.__slots__):
            raise ValueError('admission must be compiled from reviewed native source rows')
        if (type(self) is not MCSARoleListAdmission or type(batch) is not WeakSiteBatch
                or batch.task_admitted is not False or split not in ('train', 'validation', 'test')
                or type(batch.canonical_sequences) is not tuple or not batch.canonical_sequences
                or type(batch.audit) is not tuple or len(batch.audit) != len(batch.canonical_sequences)
                or type(batch.source_list_supervision) is not tuple
                or batch.vocabulary_axis_sha256 != self._axis
                or type(batch.prediction_class_count) is not int or batch.prediction_class_count != self._prediction_classes
                or type(batch.decoder_input_class_count) is not int or batch.decoder_input_class_count != self._input_classes):
            raise ValueError('exact source candidate batch/axis and explicit source split required')
        if _wire(model_binding) != self._binding:
            raise ValueError('model task semantics, source implementation or payload vocabulary differs from review')
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != self._implementation_sha256:
            raise ValueError('admission implementation changed after review')
        for path, identity in self._identities.items():
            if _identity(Path(path)) != identity:
                raise ValueError('reviewed source/component/mask artifact identity changed')
        expected = []
        for index, (sequence, audit) in enumerate(zip(batch.canonical_sequences, batch.audit)):
            if type(sequence) is not str or type(audit) is not dict or audit.get('record_id') not in self._records:
                raise ValueError('native source record identity required')
            actual_sequence, actual_split, actual_audit, rows = self._records[audit['record_id']]
            if sequence != actual_sequence or split != actual_split or _wire(audit) != actual_audit:
                raise ValueError('source sequence/audit/original split does not match immutable reviewed record')
            expected.extend((index, *row) for row in rows)
        observed = []
        for row in batch.source_list_supervision:
            if (type(row) is not SourceListSupervision
                    or any(type(getattr(row, key)) is not int for key in ('record_index', 'position_0based', 'source_residue_ordinal', 'source_mapping_ordinal'))
                    or type(row.decoder_input_ids) is not tuple or type(row.target_ids) is not tuple
                    or any(type(value) is not int for value in (*row.decoder_input_ids, *row.target_ids))):
                raise ValueError('exact native sparse source coordinates/ordered role tokens required')
            observed.append((row.record_index, row.position_0based, row.decoder_input_ids, row.target_ids,
                row.source_residue_ordinal, row.source_mapping_ordinal))
        if observed != expected:
            raise ValueError('complete native sparse rows/order/prefix/EOS differ; no truncation or relabelling allowed')
