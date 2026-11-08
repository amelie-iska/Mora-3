import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from mora3.data.mcsa_weak_sites import (
    ROLE_FIELDS, WeakRoleVocabulary, collate_weak_site_candidates,
    parse_weak_site_candidate, read_weak_site_candidates,
)


def role(emo='EMO_1', function='acceptor'):
    return dict(emo=emo, function=function, function_type='interaction', group_function='')


def vocabulary():
    return WeakRoleVocabulary(tuple(sorted(tuple(role(k, v)[f] for f in ROLE_FIELDS)
        for k, v in [('EMO_1', 'acceptor'), ('EMO_2', 'donor')])))


def candidate(roles=None):
    mapping = dict(resid=2, code='Cys', uniprot_id='P1', is_reference=True)
    roles = [role('EMO_2', 'donor'), role(), role('EMO_2', 'donor')] if roles is None else roles
    site = dict(position_1based=2, source_residue_ordinal=0, source_mapping_ordinal=0,
        source_mapping=mapping, native_annotation=dict(mcsa_id=1, roles=roles,
            residue_sequences=[mapping], private_audit_target='never a condition'),
        evidence_role='weak_source_curated_annotation_on_current_accession_reference',
        experimental_catalytic_truth=False, physical_original_sequence_identity_proven=False)
    return dict(schema='Mora3.MCSAWeakReferenceSiteCandidate/v1', record_id='mcsa-weak-reference:1',
        conditions=dict(canonical_sequence='ACD'),
        audit=dict(source_mcsa_id=1, reference_uniprot_id='P1', original_MCSA_sequence_version=None,
            full_source_sequence_identity_proven=False, quarantined_mapping_contexts=[{'disagreement': True}],
            current_uniprot_provenance=dict(primary_accession='P1',
                canonical_sequence_sha256=hashlib.sha256(b'ACD').hexdigest())),
        targets=dict(source_annotated_site_roles=[site]), loss_positions_1based=[2],
        unlisted_residue_loss_mask=False, unlisted_residues_are_negative=False,
        source_annotation_not_physical_or_experimental_truth=True,
        source_version_unknown_is_explicit_weak_evidence_not_label_repair=True,
        proposed_group='source-group:1', proposed_split='test',
        group_key_evidence=[['source', 'P1']], task_admission=False, training=False)


class WeakSiteSourceListTests(unittest.TestCase):
    def test_exact_order_and_duplicates_and_termination(self):
        vocab = vocabulary()
        record = parse_weak_site_candidate(candidate(), vocab)
        site = record.source_lists[0]
        self.assertEqual((site.position_0based, site.amino_acid), (1, 'C'))
        self.assertEqual(site.role_ids, (1, 0, 1))
        batch = collate_weak_site_candidates([record], vocab)
        target = batch.source_list_supervision[0]
        self.assertEqual(target.decoder_input_ids, (3, 1, 0, 1))
        self.assertEqual(target.target_ids, (1, 0, 1, 2))
        self.assertEqual((batch.prediction_class_count, batch.decoder_input_class_count), (3, 4))
        self.assertFalse(batch.task_admitted)

    def test_empty_observed_list_has_EOS_only_not_negative_residue_labels(self):
        vocab = vocabulary()
        batch = collate_weak_site_candidates([parse_weak_site_candidate(candidate([]), vocab)], vocab)
        self.assertEqual(batch.source_list_supervision[0].decoder_input_ids, (3,))
        self.assertEqual(batch.source_list_supervision[0].target_ids, (2,))
        self.assertEqual(len(batch.source_list_supervision), 1)
        self.assertEqual(batch.source_list_supervision[0].position_0based, 1)

    def test_conditions_exclude_all_target_source_and_positive_position_metadata(self):
        vocab = vocabulary()
        record = parse_weak_site_candidate(candidate(), vocab)
        self.assertEqual(record.condition_inputs(), {'canonical_sequence': 'ACD'})
        batch = collate_weak_site_candidates([record], vocab)
        self.assertEqual(batch.condition_inputs(), {'canonical_sequences': ('ACD',)})
        self.assertIsNone(record.audit['original_MCSA_sequence_version'])
        self.assertFalse(record.audit['full_source_sequence_identity_proven'])
        self.assertEqual(record.audit['quarantined_mapping_contexts'], [{'disagreement': True}])

    def test_extra_condition_field_is_refused(self):
        record = candidate()
        record['conditions']['known_source_site'] = 2
        with self.assertRaises(ValueError):
            parse_weak_site_candidate(record, vocabulary())

    def test_actual_mismatch_or_bad_numbering_cannot_be_supervised(self):
        for position, code in [(0, 'Cys'), (True, 'Cys'), (4, 'Cys'), (2, 'Asp')]:
            with self.subTest(position=position, code=code):
                record = candidate()
                site = record['targets']['source_annotated_site_roles'][0]
                site['source_mapping'].update(resid=position, code=code)
                site['position_1based'] = position
                with self.assertRaises(ValueError):
                    parse_weak_site_candidate(record, vocabulary())

    def test_original_mapping_pointer_is_not_inferred(self):
        record = candidate()
        record['targets']['source_annotated_site_roles'][0]['source_mapping_ordinal'] = 1
        with self.assertRaises(ValueError):
            parse_weak_site_candidate(record, vocabulary())

    def test_unknown_tuple_or_hidden_ontology_coercion_refused(self):
        for changed in (dict(function='renamed'), dict(emo='EMO_other'), dict(extra='x')):
            record = candidate()
            record['targets']['source_annotated_site_roles'][0]['native_annotation']['roles'][0].update(changed)
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                parse_weak_site_candidate(record, vocabulary())

    def test_no_candidate_training_activation_or_negative_mask(self):
        for field, value in [('task_admission', True), ('training', True),
            ('unlisted_residues_are_negative', True), ('unlisted_residue_loss_mask', True)]:
            record = candidate()
            record[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                parse_weak_site_candidate(record, vocabulary())

    def test_source_coordinates_and_loss_positions_exact(self):
        record = candidate()
        record['loss_positions_1based'] = [1, 2]
        with self.assertRaises(ValueError):
            parse_weak_site_candidate(record, vocabulary())
        record = candidate()
        record['targets']['source_annotated_site_roles'].append(copy.deepcopy(record['targets']['source_annotated_site_roles'][0]))
        with self.assertRaises(ValueError):
            parse_weak_site_candidate(record, vocabulary())

    def test_different_equal_sized_role_axis_cannot_collate(self):
        record = parse_weak_site_candidate(candidate(), vocabulary())
        other = WeakRoleVocabulary(tuple(sorted((('EMO_X', 'x', 'x', ''), ('EMO_Y', 'y', 'y', '')))))
        with self.assertRaises(ValueError):
            collate_weak_site_candidates([record], other)

    def test_frozen_vocabulary_document_axis_and_tokens(self):
        vocab = vocabulary()
        doc = dict(schema='Mora3.MCSAExactSourceRoleListVocabulary/v1', role_fields=list(ROLE_FIELDS),
            roles=[dict(id=i, **dict(zip(ROLE_FIELDS, role))) for i, role in enumerate(vocab.roles)],
            EOS_target_id=2, BOS_input_only_id=3, prediction_class_count=3, decoder_input_class_count=4)
        self.assertEqual(WeakRoleVocabulary.from_document(doc), vocab)
        doc['EOS_target_id'] = True
        with self.assertRaises(ValueError):
            WeakRoleVocabulary.from_document(doc)

    def test_pinned_file_and_duplicate_source_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'candidate.jsonl'
            raw = (json.dumps(candidate()) + '\n').encode()
            path.write_bytes(raw)
            self.assertEqual(len(read_weak_site_candidates(path, hashlib.sha256(raw).hexdigest(), vocabulary())), 1)
            with self.assertRaises(ValueError):
                read_weak_site_candidates(path, '0' * 64, vocabulary())
            path.write_bytes(raw + raw)
            with self.assertRaises(ValueError):
                read_weak_site_candidates(path, hashlib.sha256(raw + raw).hexdigest(), vocabulary())


if __name__ == '__main__':
    unittest.main()
