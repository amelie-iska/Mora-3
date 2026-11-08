"""Synthetic position/mapping controls; no source files or neural construction."""
import copy
import hashlib
import unittest
from mora3.data import mcsa_sequence_context as api


class MCSASequenceTests(unittest.TestCase):
    def mapping(self, **changes):
        d=dict(uniprot_id='P12345', code='Asp', is_reference=True, resid=2)
        d.update(changes)
        return d

    def provenance(self, seq='ADG'):
        return dict(primary_accession='P12345',canonical_sequence_sha256=hashlib.sha256(seq.encode()).hexdigest(),
            source_release='2026_03',sequence_version=3,entry_version=4,
            raw_sequence_source=dict(path='/synthetic/source.tsv.gz',sha256='a'*64,bytes=10,row_index=0),
            original_entry_ref=dict(path='/synthetic/entry.parquet',sha256='b'*64,bytes=10,row_index=0))

    def entry(self):
        return dict(mcsa_id=7,reference_uniprot_id='P12345',is_reference_uniprot_id=True,
            protein=dict(sequences=[dict(uniprot_id='P12345')]),
            residues=[dict(roles=[dict(function='synthetic annotated role')],ptm='',
                residue_chains=[dict(pdb_id='TEST',auth_resid='2A',resid=2,chain_name='B')],
                residue_sequences=[self.mapping()])])

    def test_literal_1based_code_agreement_is_not_full_version_proof(self):
        result=api.check_residue_mapping(self.mapping(),'ADG','P12345')
        self.assertEqual(result['disposition'],'POSITION_CODE_AGREES_CURRENT_SEQUENCE')
        self.assertEqual(result['source_position'],2)
        self.assertIsNone(result['original_MCSA_sequence_version'])
        self.assertFalse(result['full_source_sequence_identity_proven'])

    def test_mismatch_stays_mismatch_without_editing_source(self):
        mapping=self.mapping(code='Glu');before=copy.deepcopy(mapping)
        result=api.check_residue_mapping(mapping,'ADG','P12345')
        self.assertEqual(result['disposition'],'SOURCE_CODE_CURRENT_SEQUENCE_DISAGREEMENT')
        self.assertEqual(mapping,before)

    def test_zero_negative_boolean_string_and_out_of_range_positions(self):
        for position in (0,-1,True,'2',4):
            with self.subTest(position=position):
                result=api.check_residue_mapping(self.mapping(resid=position),'ADG','P12345')
                self.assertNotEqual(result['disposition'],'POSITION_CODE_AGREES_CURRENT_SEQUENCE')

    def test_modified_or_ambiguous_code_not_silently_normalized(self):
        for code in ('D-Asp','TPO','Asx','X',' Asp '):
            with self.subTest(code=code):
                result=api.check_residue_mapping(self.mapping(code=code),'ADG','P12345')
                self.assertEqual(result['disposition'],'SOURCE_RESIDUE_CODE_UNQUALIFIED')

    def test_alternative_and_other_accession_mappings_remain_separate(self):
        self.assertEqual(api.check_residue_mapping(self.mapping(is_reference=False),'ADG','P12345')['disposition'],
            'ALTERNATIVE_SOURCE_MAPPING_NOT_SELECTED_REFERENCE')
        self.assertEqual(api.check_residue_mapping(self.mapping(uniprot_id='Q00000'),'ADG','P12345')['disposition'],
            'DIFFERENT_SOURCE_ACCESSION')

    def test_all_target_annotation_and_audit_changes_leave_conditions_unchanged(self):
        original=self.entry();changed=copy.deepcopy(original)
        changed['mcsa_id']=9;changed['residues'][0]['roles']=[dict(function='changed hidden target')]
        changed['residues'][0]['residue_chains'][0]['auth_resid']='999'
        a=api.sequence_context(original,'ADG',self.provenance())
        b=api.sequence_context(changed,'ADG',self.provenance())
        self.assertEqual(a['conditions'],b['conditions'])
        self.assertNotEqual(a['targets'],b['targets'])
        self.assertEqual(a['conditions'],dict(canonical_sequence='ADG'))

    def test_chain_insertion_codes_and_unknown_fields_preserved_as_source_assertions(self):
        entry=self.entry();entry['residues'][0]['future_source_field']={'kept':[None,False,0]}
        result=api.sequence_context(entry,'ADG',self.provenance())
        target=result['targets']['source_residue_annotations'][0]
        self.assertEqual(target['native_annotation'],entry['residues'][0])
        self.assertEqual(target['native_annotation']['residue_chains'][0]['auth_resid'],'2A')
        self.assertFalse(result['task_admission'])
        self.assertFalse(result['unlisted_residues_are_negative'])

    def test_corrupt_current_sequence_hash_refuses(self):
        p=self.provenance();p['canonical_sequence_sha256']='0'*64
        with self.assertRaises(ValueError):api.sequence_context(self.entry(),'ADG',p)

    def test_container_residue_code_stays_unqualified(self):
        for code in ([],{}):
            with self.subTest(code=code):
                r=api.check_residue_mapping(self.mapping(code=code),'ADG','P12345')
                self.assertEqual(r['disposition'],'SOURCE_RESIDUE_CODE_UNQUALIFIED')

    def test_explicit_source_full_sequence_compares_bytes_without_imputing_version(self):
        entry=self.entry();entry['protein']['sequences'][0]['sequence']='ADG'
        result=api.sequence_context(entry,'ADG',self.provenance())
        self.assertTrue(result['audit']['original_MCSA_full_sequence_available'])
        self.assertTrue(result['audit']['source_original_vs_current_full_sequence_equality'])
        self.assertIsNone(result['audit']['original_MCSA_sequence_version'])

    def test_conflicting_or_different_source_full_sequences_do_not_qualify(self):
        for sequences in ([dict(uniprot_id='P12345',sequence='AEG')],
                          [dict(uniprot_id='P12345',sequence='ADG'),dict(uniprot_id='P12345',sequence='AEG')]):
            with self.subTest(sequences=sequences):
                entry=self.entry();entry['protein']['sequences']=sequences
                result=api.sequence_context(entry,'ADG',self.provenance())
                self.assertFalse(result['audit']['full_source_sequence_identity_proven'])


if __name__=='__main__':unittest.main()
