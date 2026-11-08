"""Synthetic sealed packets exercise binding, never qualify project source data."""
from dataclasses import replace
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import pyarrow as pa
import pyarrow.parquet as pq

import mora3.data.mcsa_role_list_admission as admission_module
from mora3.data.mcsa_role_list_admission import MCSARoleListAdmission
from mora3.data.mcsa_weak_sites import (
    ROLE_FIELDS, WeakRoleVocabulary, parse_weak_site_candidate, collate_weak_site_candidates)


def wire(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()


def reference(path):
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def write(path, value):
    path.write_bytes(wire(value))
    return reference(path)


def role():
    return dict(emo='EMO_1', function='acceptor', function_type='interaction', group_function='')


def native(mid, sequence, split):
    accession = 'P' + str(mid)
    mapping = dict(resid=2, code='Cys', is_reference=True, uniprot_id=accession)
    annotation = dict(mcsa_id=mid, roles=[role(), role()], residue_sequences=[mapping])
    return dict(schema=admission_module.NATIVE, record_id='mcsa-weak-reference:' + str(mid),
        conditions=dict(canonical_sequence=sequence),
        audit=dict(source_mcsa_id=mid, reference_uniprot_id=accession,
            current_uniprot_provenance=dict(primary_accession=accession,
                canonical_sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest()),
            original_MCSA_sequence_version=None, full_source_sequence_identity_proven=False,
            quarantined_mapping_contexts=[dict(source_only='unchanged')]),
        targets=dict(source_annotated_site_roles=[dict(position_1based=2, source_residue_ordinal=0,
            source_mapping_ordinal=0, source_mapping=mapping, native_annotation=annotation,
            evidence_role='weak_source_curated_annotation_on_current_accession_reference',
            experimental_catalytic_truth=False, physical_original_sequence_identity_proven=False)]),
        loss_positions_1based=[2], unlisted_residue_loss_mask=False, unlisted_residues_are_negative=False,
        source_annotation_not_physical_or_experimental_truth=True,
        source_version_unknown_is_explicit_weak_evidence_not_label_repair=True,
        proposed_group='group:' + str(mid), proposed_split=split, group_key_evidence=[['fixture', str(mid)]],
        task_admission=False, training=False)


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.vocab = WeakRoleVocabulary((tuple(role()[f] for f in ROLE_FIELDS),))
        self.natives = [native(i, sequence, split) for i, sequence, split in
            [(1, 'ACD', 'train'), (2, 'ACE', 'validation'), (3, 'ACF', 'test')]]
        self.records = [parse_weak_site_candidate(row, self.vocab) for row in self.natives]
        candidates = self.root / 'candidates.jsonl'
        candidates.write_bytes(b''.join(wire(row) + b'\n' for row in self.natives))
        candidate_ref = reference(candidates)
        vocab_ref = write(self.root / 'vocabulary.json', dict(
            schema='Mora3.MCSAExactSourceRoleListVocabulary/v1', role_fields=list(ROLE_FIELDS),
            roles=[dict(id=0, **role())], EOS_target_id=1, BOS_input_only_id=2,
            prediction_class_count=2, decoder_input_class_count=3, source_candidate_file=candidate_ref))
        source_ref = write(self.root / 'source-manifest.json', dict(source_candidate_file=candidate_ref, vocabulary=vocab_ref))
        closure_ref = write(self.root / 'closure.json', dict(scope='synthetic sealed native fixture'))
        closure_check = write(self.root / 'closure-check.json', dict(producer_manifest=closure_ref,
            all1003_MCSA_source_components_and_roles_unchanged=True))
        shards, sidecars = [], []
        for i, candidate in enumerate(self.natives):
            raw = wire(candidate)
            tid = hashlib.sha256(b'Mora3.MCSA.source-role-list/v1\0' + raw).hexdigest()
            row = dict(record_id=candidate['record_id'], task_id=tid, native_payload=raw,
                native_payload_sha256=hashlib.sha256(raw).hexdigest(), source_family=admission_module.FAMILY,
                task_name=admission_module.TASK, native_schema=admission_module.NATIVE,
                split=candidate['proposed_split'], admitted_for_task=False, model_adapter_supported=False,
                group_ids_json=json.dumps([candidate['proposed_group']]))
            p = self.root / ('shard%d.parquet' % i)
            pq.write_table(pa.Table.from_pylist([row]), p)
            shard = reference(p) | dict(rows=1, split=candidate['proposed_split'])
            gp = self.root / ('groups%d.parquet' % i)
            pq.write_table(pa.Table.from_pylist([dict(task_id=tid, effective_groups_json=row['group_ids_json'])]), gp)
            shards.append(shard)
            sidecars.append(dict(source_shard=shard, authoritative_sidecar=reference(gp)))
        self.component_ref = write(self.root / 'component.json', dict(
            source_family=admission_module.FAMILY, task_name=admission_module.TASK, native_schema=admission_module.NATIVE,
            scientific_task_admission=False, model_adapter_supported=False,
            current_protein_gene_guard_closure=closure_ref, independent_current_closure=closure_check,
            source_vocabulary=vocab_ref, source_candidate_file=candidate_ref, source_native_manifest=source_ref,
            source_entry_views=3, shards=shards, group_sidecars=sidecars,
            source_splits=dict(train=1, validation=1, test=1),
            sparse_mapping_views_by_split=dict(train=1, validation=1, test=1),
            role_EOS_symbols_by_split=dict(train=3, validation=3, test=3)))
        parent_ref = write(self.root / 'parent-master.json', dict(scope='synthetic parent'))
        mask_path = self.root / 'mask.parquet'
        pq.write_table(pa.Table.from_pylist([dict(source_shard='synthetic-native', source_row_ordinal=0,
            protected=True, unknown=False)]), mask_path)
        self.mask_ref = write(self.root / 'mask-manifest.json', dict(parent_master=parent_ref,
            current_protein_gene_closure=closure_ref, mask=reference(mask_path), combined_mask_taskviews=1))
        self.proof_ref = write(self.root / 'independent-proof.json', dict(component_manifest=self.component_ref,
            final_mask_manifest=self.mask_ref, exact_parent_master=parent_ref,
            exact_final_mask_complete_both_directions=True,
            all_candidate_payload_bytes_task_IDs_full_axes_and_aligned_groups_equal=True,
            all_candidate_component_guards_role_disjoint=True,
            source_native_candidate_records=3, component_split_counts=dict(train=1, validation=1, test=1),
            mask_taskviews=1, protected_reference_exclusions=1, unknown_reference_quarantines=0))
        fit_input_path = self.root / 'fit-input.jsonl'
        train_raw = wire(self.natives[0])
        train_audit = dict(source_record_id=self.natives[0]['record_id'],
            source_task_id=hashlib.sha256(b'Mora3.MCSA.source-role-list/v1\0' + train_raw).hexdigest(),
            source_native_payload_sha256=hashlib.sha256(train_raw).hexdigest(), source_shard=shards[0],
            source_shard_row_ordinal=0, original_split='train',
            canonical_sequence_sha256=hashlib.sha256(b'ACD').hexdigest(),
            current_sequence_provenance=self.natives[0]['audit']['current_uniprot_provenance'])
        fit_input_path.write_bytes(wire(dict(conditions=self.natives[0]['conditions'], audit=train_audit)) + b'\n')
        self.fit_input_ref = reference(fit_input_path)
        fit_scope = dict(cohort='synthetic-source-fixture', input_sha256=self.fit_input_ref['sha256'],
            allcorpus_fit=False, multilingual_fit=False, universal_byte_coverage=True)
        self.payload_ref = write(self.root / 'payload-vocabulary.json', dict(version=1, fit_scope=fit_scope, learned_hex=[], controls=['pad','node','edge']))
        raw_fit = write(self.root / 'underlying-fit.json', dict(schema='Mora3.SourceConditionTokenizerFitReceipt/v1',
            completed=True, artifact_sha256=self.payload_ref['sha256'], artifact_bytes=self.payload_ref['bytes'],
            all_source_records=1, all_source_bytes=3,
            independent_all602_shortest_path_and_lossless_roundtrip=True, input_and_code_unchanged=True,
            source_fit_scope=fit_scope))
        self.fit_receipt = dict(schema='Mora3.MCSACanonicalConditionPayloadFit/v1', status='complete-source-fit',
            source_component_manifest=self.component_ref, payload_vocabulary_artifact=self.payload_ref,
            fit_input=self.fit_input_ref, train_views=1, full_canonical_sequence_characters=3,
            fit_fields=['conditions.canonical_sequence'], target_fields_used=False, heldout_rows_used=False,
            underlying_fit_receipt=raw_fit)
        self.fit_receipt_ref = write(self.root / 'payload-fit-receipt.json', self.fit_receipt)
        self.binding = dict(schema='Mora3.MCSASourceRoleListModelBinding/v1',
            source_family=admission_module.FAMILY, task_name=admission_module.TASK, native_schema=admission_module.NATIVE,
            axis_sha256=self.vocab.axis_sha256, max_target_length=3, condition_inputs=['canonical_sequences'],
            target_history='teacher-forced-ordered-role-prefix-BOS-input-only',
            targets='observed-ordered-source-role-list-including-EOS',
            source_position='output-selector-only-never-encoder-marker', unknown_positions='no-target-no-loss',
            biological_negatives=False, payload_vocabulary_sha256=self.payload_ref['sha256'],
            encoder_sequence_contract='complete-native-sequence-no-crop-or-truncation', payload_max_tokens=8192,
            **admission_module._live_code_hashes())
        self.packet = dict(schema='Mora3.MCSARoleListReviewedSourceAdmission/v1',
            review_state='accepted-scoped-source',
            admission_implementation_sha256=hashlib.sha256(Path(admission_module.__file__).read_bytes()).hexdigest(),
            approved_model_binding=self.binding, component_manifest=self.component_ref,
            payload_vocabulary_artifact=self.payload_ref, payload_fit_receipt=self.fit_receipt_ref,
            final_mask_manifest=self.mask_ref, final_mask_component_verification=self.proof_ref,
            coordinator_review=dict(owner='root-coordinator',
                approved_model_binding_sha256=hashlib.sha256(wire(self.binding)).hexdigest(),
                component_manifest_sha256=self.component_ref['sha256'],
                final_mask_manifest_sha256=self.mask_ref['sha256'], independent_verification_sha256=self.proof_ref['sha256']))

    def token(self, packet=None):
        path = self.root / 'review.json'
        reference = write(path, self.packet if packet is None else packet)
        return MCSARoleListAdmission.from_review_packet(path, reference['sha256'], vocabulary=self.vocab)

    def batch(self, index=0):
        return collate_weak_site_candidates([self.records[index]], self.vocab)

    def test_real_rows_and_each_source_split_are_bound(self):
        token = self.token()
        for i, split in enumerate(('train', 'validation', 'test')):
            token.require_batch(self.batch(i), model_binding=self.binding, split=split)

    def test_training_and_eval_roles_cannot_be_relabelled(self):
        token = self.token()
        for index, split in [(0, 'test'), (1, 'train'), (2, 'validation')]:
            with self.subTest(index=index, split=split), self.assertRaises(ValueError):
                token.require_batch(self.batch(index), model_binding=self.binding, split=split)

    def test_boolean_or_uncompiled_object_is_not_admission(self):
        with self.assertRaises(TypeError):
            MCSARoleListAdmission(True)
        token = object.__new__(MCSARoleListAdmission)
        with self.assertRaises(ValueError):
            token.require_batch(self.batch(), model_binding=self.binding, split='train')
        token = self.token()
        with self.assertRaises(ValueError):
            token.require_batch(replace(self.batch(), task_admitted=True), model_binding=self.binding, split='train')

    def test_accepted_review_and_trusted_digest_are_required(self):
        for state in ('pending-scoped-review', True, None):
            packet = copy.deepcopy(self.packet)
            packet['review_state'] = state
            with self.subTest(state=state), self.assertRaises(ValueError):
                self.token(packet)
        path = self.root / 'review.json'
        write(path, self.packet)
        with self.assertRaises(ValueError):
            MCSARoleListAdmission.from_review_packet(path, '0' * 64, vocabulary=self.vocab)

    def test_full_sequence_and_source_audit_cannot_be_substituted(self):
        token = self.token()
        batch = self.batch()
        for change in ('sequence', 'record_id', 'quarantine'):
            altered = replace(batch, canonical_sequences=('AAA',)) if change == 'sequence' else copy.deepcopy(batch)
            if change == 'record_id':
                altered.audit[0]['record_id'] = self.records[1].audit['record_id']
            elif change == 'quarantine':
                altered.audit[0]['quarantined_mapping_contexts'] = []
            with self.subTest(change=change), self.assertRaises(ValueError):
                token.require_batch(altered, model_binding=self.binding, split='train')

    def test_roles_order_duplicates_complete_rows_and_EOS_are_bound(self):
        token = self.token()
        batch = self.batch()
        row = batch.source_list_supervision[0]
        changes = [(), (row, row), (replace(row, target_ids=(0, 1)),),
            (replace(row, target_ids=(1, 0, 0)),),
            (replace(row, decoder_input_ids=(2, 1, 0)),),
            (replace(row, source_mapping_ordinal=1),),
            (replace(row, position_0based=True),)]
        for rows in changes:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                token.require_batch(replace(batch, source_list_supervision=rows), model_binding=self.binding, split='train')

    def test_model_code_semantics_and_vocabulary_cannot_drift(self):
        token = self.token()
        for key, value in [('model_source_sha256', '0' * 64), ('payload_vocabulary_sha256', 'b' * 64),
                ('biological_negatives', True), ('max_target_length', 2)]:
            binding = dict(self.binding, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                token.require_batch(self.batch(), model_binding=binding, split='train')

    def test_changed_native_file_is_refused_before_use(self):
        token = self.token()
        path = self.root / 'candidates.jsonl'
        path.write_bytes(path.read_bytes() + b'\n')
        with self.assertRaises(ValueError):
            token.require_batch(self.batch(), model_binding=self.binding, split='train')

    def test_mask_or_component_snapshot_cannot_be_mixed(self):
        packet = copy.deepcopy(self.packet)
        bad_mask = write(self.root / 'other-mask.json', dict(other='snapshot'))
        packet['final_mask_manifest'] = bad_mask
        packet['coordinator_review']['final_mask_manifest_sha256'] = bad_mask['sha256']
        with self.assertRaises(ValueError):
            self.token(packet)

    def test_candidate_envelopes_cannot_be_promoted_by_flag(self):
        component = json.loads((self.root / 'component.json').read_text())
        shard = component['shards'][0]
        path = Path(shard['path'])
        rows = pq.read_table(path).to_pylist()
        rows[0]['admitted_for_task'] = True
        # Deliberate synthetic mutation; production artifacts are never rewritten.
        pq.write_table(pa.Table.from_pylist(rows), path)
        component['shards'][0].update(reference(path))
        component['group_sidecars'][0]['source_shard'] = dict(component['shards'][0])
        component_ref = write(self.root / 'component-mutated.json', component)
        packet = copy.deepcopy(self.packet)
        packet['component_manifest'] = component_ref
        packet['coordinator_review']['component_manifest_sha256'] = component_ref['sha256']
        # Even a consistently updated local synthetic proof cannot legalize flag
        # promotion: the reader validates actual native candidate envelope state.
        proof = json.loads((self.root / 'independent-proof.json').read_text())
        proof['component_manifest'] = component_ref
        packet['final_mask_component_verification'] = write(self.root / 'proof-mutated.json', proof)
        packet['coordinator_review']['independent_verification_sha256'] = packet['final_mask_component_verification']['sha256']
        with self.assertRaises(ValueError):
            self.token(packet)

    def test_json_duplicate_keys_and_nonfinite_do_not_qualify(self):
        for raw in (b'{"schema":1,"schema":2}', b'{"x":NaN}', b'{"x":1e9999}'):
            path = self.root / 'malformed.json'
            path.write_bytes(raw)
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                MCSARoleListAdmission.from_review_packet(path, hashlib.sha256(raw).hexdigest(), vocabulary=self.vocab)

    def test_actual_payload_artifact_cannot_be_just_a_coordinator_hash(self):
        packet = copy.deepcopy(self.packet)
        packet['payload_vocabulary_artifact'] = write(self.root / 'different-payload.json', dict(other=True))
        with self.assertRaises(ValueError):
            self.token(packet)
        packet = copy.deepcopy(self.packet)
        packet.pop('payload_fit_receipt')
        with self.assertRaises(ValueError):
            self.token(packet)

    def test_actual_completed_fit_counts_and_condition_only_scope_required(self):
        for key, value in [('status', 'running'), ('train_views', 2),
                ('full_canonical_sequence_characters', 4), ('target_fields_used', True), ('heldout_rows_used', True)]:
            receipt = dict(self.fit_receipt, **{key: value})
            packet = copy.deepcopy(self.packet)
            packet['payload_fit_receipt'] = write(self.root / ('bad-fit-' + key + '.json'), receipt)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.token(packet)

    def test_fit_input_requires_actual_source_train_sequence_and_coordinates(self):
        original = json.loads((self.root / 'fit-input.jsonl').read_text())
        for change in ('sequence', 'heldout', 'targets'):
            item = copy.deepcopy(original)
            if change == 'sequence':
                item['conditions']['canonical_sequence'] = 'AAA'
            elif change == 'heldout':
                item['audit']['source_record_id'] = self.natives[1]['record_id']
            else:
                item['targets'] = {'gold': 'must not fit'}
            path = self.root / ('changed-input-' + change + '.jsonl')
            path.write_bytes(wire(item) + b'\n')
            new_input = reference(path)
            # Keep all integrity metadata consistent to test actual source-row
            # comparison, not merely the first receipt SHA check.
            new_scope = dict(cohort='synthetic-source-fixture', input_sha256=new_input['sha256'],
                allcorpus_fit=False, multilingual_fit=False, universal_byte_coverage=True)
            payload_ref = write(self.root / ('payload-' + change + '.json'), dict(version=1, fit_scope=new_scope))
            raw_fit = json.loads((self.root / 'underlying-fit.json').read_text())
            raw_fit.update(artifact_sha256=payload_ref['sha256'], artifact_bytes=payload_ref['bytes'], source_fit_scope=new_scope)
            receipt = dict(self.fit_receipt, fit_input=new_input, payload_vocabulary_artifact=payload_ref,
                underlying_fit_receipt=write(self.root / ('raw-fit-' + change + '.json'), raw_fit))
            packet = copy.deepcopy(self.packet)
            packet.update(payload_vocabulary_artifact=payload_ref,
                payload_fit_receipt=write(self.root / ('fit-' + change + '.json'), receipt))
            packet['approved_model_binding']['payload_vocabulary_sha256'] = payload_ref['sha256']
            packet['coordinator_review']['approved_model_binding_sha256'] = hashlib.sha256(wire(packet['approved_model_binding'])).hexdigest()
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.token(packet)

    def test_compiled_source_token_cannot_be_mutated(self):
        token = self.token()
        with self.assertRaises(AttributeError):
            token._axis = 'changed'
        with self.assertRaises(TypeError):
            token._records['invented'] = ('AAA', 'train', b'{}', ())

    def test_complete_encoder_contract_is_required_and_cannot_be_removed(self):
        for value in ('cropped', None, True):
            packet = copy.deepcopy(self.packet)
            packet['approved_model_binding']['encoder_sequence_contract'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.token(packet)
        for key in ('encoder_sequence_contract', 'payload_max_tokens'):
            packet = copy.deepcopy(self.packet)
            del packet['approved_model_binding'][key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.token(packet)

    def test_payload_bound_is_positive_exact_and_covers_complete_native_stream(self):
        for limit in (True, 0, -1, 1.0, 3):
            packet = copy.deepcopy(self.packet)
            packet['approved_model_binding']['payload_max_tokens'] = limit
            packet['coordinator_review']['approved_model_binding_sha256'] = hashlib.sha256(
                wire(packet['approved_model_binding'])).hexdigest()
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                self.token(packet)

    def test_complete_stream_bound_includes_metadata_not_only_raw_AA_tokens(self):
        from mora3.tokenization.shortest_path import ByteTokenizer
        tokenizer = ByteTokenizer((), controls=('pad','node','edge'))
        length = admission_module._complete_sequence_stream_length('ACD', tokenizer)
        self.assertGreater(length, len(tokenizer.encode(b'ACD')))
        packet = copy.deepcopy(self.packet)
        packet['approved_model_binding']['payload_max_tokens'] = length
        packet['coordinator_review']['approved_model_binding_sha256'] = hashlib.sha256(
            wire(packet['approved_model_binding'])).hexdigest()
        token = self.token(packet)
        token.require_batch(self.batch(), model_binding=packet['approved_model_binding'], split='train')

    def test_payload_bound_covers_validation_and_test_without_refitting(self):
        from unittest.mock import patch
        original = admission_module._complete_sequence_stream_length
        calls = []
        def measured(sequence, tokenizer):
            calls.append(sequence)
            return 9000 if sequence == 'ACF' else original(sequence, tokenizer)
        with patch.object(admission_module, '_complete_sequence_stream_length', measured):
            with self.assertRaisesRegex(ValueError, 'complete source sequence'):
                self.token()
        self.assertEqual(calls, ['ACD','ACE','ACF'])


if __name__ == '__main__':
    unittest.main()
