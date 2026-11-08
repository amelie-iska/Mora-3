"""Data integrity and leakage regressions for the opt-in SCRI extension."""
import copy
import hashlib
import json
import unittest

from mora3.extensions.scri.contracts import (
    NativeValue, SourceEvidence, StateRecord, TransitionPair, validate_record,
    compile_transition, leakage_components, split_components, validate_splits,
)


def evidence(**changes):
    values = dict(dataset_id='study-A', revision='r1', artifact_sha256='a'*64,
                  source_locator='matrix.h5ad#obs/cell-1', license_id='CC-BY-4.0',
                  evidence_kind='synthetic_fixture', study_id='study-A',
                  sample_id='replicate-1', donor_id='donor-1',
                  experiment_id='experiment-1', lineage_id=None)
    return SourceEvidence(**(values | changes))


def record(key='cell-1', **changes):
    values = dict(record_id=key, source=evidence(), organism='human',
                  cell_type='fibroblast', time_value=0.0, time_unit='hour',
                  time_reference='treatment-start',
                  measurements=(NativeValue('RNA', 'GENE_A', 2.0, 'log1p-count', True),
                                NativeValue('RNA', 'GENE_B', None, 'log1p-count', False)),
                  coordinates=None, coordinate_unit=None, coordinate_frame=None,
                  intervention_ids=(), official_split=None)
    return StateRecord(**(values | changes))


class ContractsTests(unittest.TestCase):
    def test_absent_gene_is_not_a_zero_target(self):
        r = record()
        validate_record(r)
        self.assertIsNone(r.measurements[1].value)
        with self.assertRaisesRegex(ValueError, 'missing'):
            NativeValue('RNA', 'GENE_B', 0.0, 'log1p-count', False)

    def test_nonfinite_and_boolean_numbers_refused(self):
        for x in (float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                NativeValue('RNA', 'G', x, 'count', True)

    def test_negative_counts_and_fractional_counts_refused(self):
        for x in (-1, 0.5):
            with self.assertRaises(ValueError):
                NativeValue('RNA', 'G', x, 'count', True)

    def test_unpaired_destructive_assays_do_not_become_longitudinal_pairs(self):
        before = record()
        after = record('cell-2', time_value=24.0)
        with self.assertRaisesRegex(ValueError, 'lineage'):
            TransitionPair(before, after, 'same_cell', 'b'*64)

    def test_cross_donor_pairs_refused(self):
        after = record('cell-2', source=evidence(donor_id='donor-2'), time_value=24.0)
        with self.assertRaisesRegex(ValueError, 'donor'):
            TransitionPair(record(), after, 'matched_population', 'b'*64)

    def test_native_axes_and_units_must_match(self):
        after = record('cell-2', time_value=24.0,
                       measurements=(NativeValue('RNA','GENE_A',3.0,'count',True),))
        with self.assertRaisesRegex(ValueError, 'axis'):
            compile_transition(TransitionPair(record(), after, 'matched_population', 'b'*64))

    def test_target_values_and_source_ids_not_in_predictor(self):
        after = record('secret-outcome-id', time_value=24.0, intervention_ids=('drug-X',),
                       measurements=(NativeValue('RNA','GENE_A',12345.0,'log1p-count',True),
                                     NativeValue('RNA','GENE_B',None,'log1p-count',False)))
        task = compile_transition(TransitionPair(record(), after, 'matched_population', 'b'*64))
        rendered = json.dumps(task['condition'], sort_keys=True)
        self.assertNotIn('12345', rendered)
        self.assertNotIn('secret-outcome-id', rendered)
        self.assertNotIn('donor-1', rendered)
        self.assertEqual(task['target']['values'], [12345.0, None])
        self.assertEqual(task['target']['observed'], [True, False])
        self.assertEqual(task['target']['causal_claim'], False)

    def test_simulation_cannot_be_joined_to_experimental_target(self):
        after = record('cell-2', source=evidence(evidence_kind='simulation'),time_value=24.0)
        with self.assertRaisesRegex(ValueError, 'evidence'):
            TransitionPair(record(), after, 'matched_population', 'b'*64)

    def test_component_transitivity_prevents_indirect_leakage(self):
        a=record('a',source=evidence(sample_id='s1',donor_id='d1',experiment_id='e1',source_locator='a'))
        b=record('b',source=evidence(sample_id='s2',donor_id='d1',experiment_id='e2',source_locator='b'))
        c=record('c',source=evidence(sample_id='s2',donor_id='d2',experiment_id='e3',source_locator='c'))
        groups=leakage_components((a,b,c), holdout_axes=('donor','sample'))
        self.assertEqual(len(groups),1)
        self.assertEqual(set(groups[0]),{'a','b','c'})

    def test_unknown_donor_is_not_a_fake_unique_donor(self):
        r=record(source=evidence(donor_id=None))
        with self.assertRaisesRegex(ValueError, 'donor'):
            leakage_components((r,),holdout_axes=('donor',))

    def test_conflicting_official_splits_refused(self):
        a=record('a',official_split='train')
        b=record('b',official_split='test')
        with self.assertRaisesRegex(ValueError,'official'):
            split_components((a,b),holdout_axes=('donor',),seed=17)

    def test_split_result_is_order_independent(self):
        rows=tuple(record(str(i),source=evidence(study_id=f's{i}',sample_id=f's{i}',
                  donor_id=f'd{i}',experiment_id=f'e{i}',source_locator=f'obs/{i}')) for i in range(20))
        x=split_components(rows,holdout_axes=('study','donor'),seed=17)
        y=split_components(tuple(reversed(rows)),holdout_axes=('study','donor'),seed=17)
        self.assertEqual(x,y)
        validate_splits(rows,x,holdout_axes=('study','donor'))
        x[rows[0].record_id]='bad'
        with self.assertRaises(ValueError):
            validate_splits(rows,x,holdout_axes=('study','donor'))


if __name__ == '__main__':
    unittest.main()
