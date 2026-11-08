"""Compatibility and scientific-semantics checks for the pinned safe CSVs."""
from hashlib import sha256
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from mora3.extensions.scri.tasmorpho import (
    CASES, COUNT_HEADER, MAX_CSV_BYTES, SUPPORTED_SOURCE_COMMIT,
    UnsupportedTasmorphoSemantics, load_reconstructed_counts, load_summary,
    parse_reconstructed_counts_bytes, parse_summary_bytes,
)


# Synthetic cases exercising the original CSV schema. No upstream payload is embedded.
SUMMARY = b"""case,E_mem,E_gj,E_pump,E_total,d_mean_vmem_mV,d_wound_gj
control,0,0,0,0,0,0
na,1.5,0.25,0.25,2,20,0
k,0.2,0.1,0.1,0.4,-10,0.01
gj,0.5,2,0.5,3,0,0.3
na_k,2,0.25,0.25,2.5,9,0.03
na_gj,3,2.5,0.5,6,22,0.32
k_gj,0.5,2.5,0.5,3.5,-10,0.29
"""


def summary(payload=SUMMARY, **overrides):
    kwargs = dict(source_commit=SUPPORTED_SOURCE_COMMIT,
                  run_id='synthetic-schema-example', expected_sha256=sha256(payload).hexdigest(),
                  source_locator='https://example.invalid/synthetic/betse_summary.csv')
    kwargs.update(overrides)
    return parse_summary_bytes(payload, **kwargs)


def edited_summary(case, column, value):
    rows = list(csv.reader(io.StringIO(SUMMARY.decode())))
    index = rows[0].index(column)
    for row in rows[1:]:
        if row[0] == case:
            row[index] = str(value)
    stream = io.StringIO(newline='')
    csv.writer(stream, lineterminator='\n').writerows(rows)
    return stream.getvalue().encode()


def count_fixture():
    # Reconstruct the CSV fields from audited script03's CountDatum constants;
    # the upstream pinned tree does not commit this generated count CSV.
    source_rows = [
        ('8OH_immediate_DH', 143, 573, 'primary_fit_target',
         'Immediate double-headed outcomes after 8-OH'),
        ('8OH_challenge_DH_among_immediate_SH', 36, 155, 'primary_fit_target',
         'Challenge double-headed outcomes after water re-cut of 8-OH immediate SH worms'),
        ('nigericin_immediate_DH', 17, 132, 'condition_setting',
         'Immediate double-headed outcomes after nigericin'),
        ('monensin_immediate_DH', 11, 89, 'condition_setting',
         'Immediate double-headed outcomes after monensin'),
        ('8OH_challenge_DH_among_immediate_DH', 100, 100, 'consistency_check_only',
         'Consistency check only; not used in reduced fit'),
    ]
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(COUNT_HEADER)
    for label, successes, trials, role, description in source_rows:
        writer.writerow((label, successes, trials, successes / trials, role, description))
    return stream.getvalue().encode()


def counts(payload=None, **overrides):
    payload = count_fixture() if payload is None else payload
    kwargs = dict(source_commit=SUPPORTED_SOURCE_COMMIT, run_id='count-reconstruction',
                  expected_sha256=sha256(payload).hexdigest(),
                  source_locator='file:///reviewed-output/reconstructed_count_table.csv')
    kwargs.update(overrides)
    return parse_reconstructed_counts_bytes(payload, **kwargs)


class SummaryCompatibilityTests(unittest.TestCase):
    def test_synthetic_summary_reconstructs_grounding_and_units(self):
        result = summary()
        self.assertEqual(tuple(row.case for row in result.cases), CASES)
        self.assertEqual(result.grounding.Q, ((2, .5), (.5, 3)))
        self.assertEqual(result.grounding.R, ((.02, 0), (0, .3)))
        self.assertAlmostEqual(result.grounding.K[0][0], .02**2 * 12/23)
        self.assertAlmostEqual(result.grounding.K[0][1], -.02 * .3 * 2/23)
        self.assertAlmostEqual(result.grounding.K[1][1], .3**2 * 8/23)
        wire = result.wire()
        self.assertEqual(wire['evidence_kind'], 'simulation')
        self.assertEqual(wire['native_units']['E_total'], 's')
        self.assertEqual(wire['native_units']['d_mean_vmem_mV'], 'mV')
        self.assertEqual(wire['grounding']['response_units'][0], 'V')
        self.assertFalse(wire['grounding']['eigenaxis_exported'])
        self.assertFalse(wire['grounding']['morphology_axis_identified'])
        self.assertIsNone(wire['native_drug_mapping'])
        self.assertIsNone(wire['source']['random_seed'])
        self.assertIsNone(wire['source']['geometry_sha256'])
        self.assertIsNone(wire['source']['data_license_id'])
        self.assertFalse(wire['training_admitted'])
        json.dumps(wire, allow_nan=False)

    def test_unknown_revision_and_false_seed_are_rejected(self):
        for overrides in ({'source_commit': '0' * 40}, {'random_seed': True},
                          {'random_seed': -1}, {'geometry_sha256': 'unknown'}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                summary(**overrides)

    def test_artifact_digest_is_verified(self):
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            summary(SUMMARY.replace(b',20,0', b',21,0'),
                    expected_sha256=sha256(SUMMARY).hexdigest())

    def test_native_header_is_exact(self):
        bad_headers = [
            b'case,E_mem,E_gj,E_pump,E_total,d_mean_vmem_mV,d_mean_vmem_mV',
            b'case,E_mem,E_gj,E_pump,E_total,d_mean_vmem_V,d_wound_gj',
            SUMMARY.splitlines()[0] + b',cell_id',
        ]
        for header in bad_headers:
            with self.subTest(header=header), self.assertRaisesRegex(ValueError, 'header'):
                summary(header + b'\n' + SUMMARY.split(b'\n', 1)[1])

    def test_missing_duplicate_and_unrecognized_cases_fail(self):
        for payload in (SUMMARY + SUMMARY.splitlines()[1] + b'\n',
                        b'\n'.join(SUMMARY.splitlines()[:-1]) + b'\n',
                        SUMMARY.replace(b'na_k,', b'8OH,'),
                        SUMMARY.replace(b'na_k,', b'na,')):
            with self.subTest(payload=payload[-80:]), self.assertRaisesRegex(ValueError, 'seven native cases'):
                summary(payload)

    def test_nan_infinite_negative_and_inconsistent_energy_fail(self):
        for column, value in (('d_wound_gj', 'NaN'), ('E_mem', 'inf'),
                              ('E_pump', '-1'), ('E_total', '9.0')):
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                summary(edited_summary('na', column, value))

    def test_control_must_be_exactly_zero(self):
        with self.assertRaisesRegex(ValueError, 'zero control'):
            summary(edited_summary('control', 'd_mean_vmem_mV', '1e-15'))

    def test_indefinite_q_is_not_silently_regularized(self):
        # Every component is finite, nonnegative and summing, but the pair
        # energy implies a non-positive-definite quadratic form.
        rows = list(csv.reader(io.StringIO(SUMMARY.decode())))
        for row in rows[1:]:
            if row[0] == 'na_gj':
                row[1:5] = ['10', '0', '0', '10']
        stream = io.StringIO(newline='')
        csv.writer(stream, lineterminator='\n').writerows(rows)
        with self.assertRaisesRegex(ValueError, 'positive definite'):
            summary(stream.getvalue().encode())

    def test_case_order_is_normalized_without_dropping_fields(self):
        lines = SUMMARY.splitlines()
        reversed_payload = b'\n'.join([lines[0], *reversed(lines[1:])]) + b'\n'
        result = summary(reversed_payload)
        self.assertEqual(result.cases, summary().cases)
        self.assertNotEqual(result.provenance.artifact_sha256, summary().provenance.artifact_sha256)

    def test_shared_run_partition_is_stable_across_artifact_order(self):
        lines = SUMMARY.splitlines()
        alternate = b'\n'.join([lines[0], *reversed(lines[1:])]) + b'\n'
        self.assertEqual(summary().provenance.partition_group,
                         summary(alternate).provenance.partition_group)
        self.assertNotEqual(summary().provenance.partition_group,
                            summary(run_id='separate-run').provenance.partition_group)

    def test_aggregate_cases_cannot_become_cells_or_graphs(self):
        result = summary()
        for method in (result.to_state_records, result.to_cell_graph):
            with self.assertRaises(UnsupportedTasmorphoSemantics):
                method()

    def test_file_loader_does_not_invent_remote_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'betse_summary.csv'
            path.write_bytes(SUMMARY)
            result = load_summary(path, source_commit=SUPPORTED_SOURCE_COMMIT,
                                  run_id='local-test', expected_sha256=sha256(SUMMARY).hexdigest())
            self.assertEqual(result.provenance.source_locator, path.as_uri())

    def test_malformed_csv_or_binary_artifacts_are_rejected(self):
        for payload in (b'\x80\x04dill', b'\xff', SUMMARY + b'\n',
                        SUMMARY.replace(b'control,', b'control,\x00'),
                        SUMMARY + b'"unclosed'):
            with self.subTest(payload=payload[-20:]), self.assertRaises(ValueError):
                summary(payload)

    def test_file_loaders_reject_oversize_and_symlink_paths_before_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            actual = root / 'actual.csv'
            actual.write_bytes(SUMMARY)
            link = root / 'link.csv'
            link.symlink_to(actual)
            directory_link = root / 'linked-directory'
            directory_link.symlink_to(root, target_is_directory=True)
            large = root / 'oversize.csv'
            large.write_bytes(b'x' * (MAX_CSV_BYTES + 1))
            for loader in (load_summary, load_reconstructed_counts):
                for path in (link, directory_link / 'actual.csv', large):
                    with self.subTest(loader=loader.__name__, path=path), self.assertRaisesRegex(
                            ValueError, 'bounded regular nonsymlink'):
                        loader(path, source_commit=SUPPORTED_SOURCE_COMMIT,
                               run_id='guard-test', expected_sha256=sha256(SUMMARY).hexdigest())


class CountEvidenceTests(unittest.TestCase):
    def test_native_count_schema_preserves_conditional_denominators_and_roles(self):
        result = counts()
        by_label = {row.label: row for row in result.counts}
        self.assertEqual(by_label['8OH_challenge_DH_among_immediate_SH'].denominator_population,
                         'immediate_SH')
        self.assertEqual(by_label['8OH_challenge_DH_among_immediate_SH'].trials, 155)
        self.assertEqual(by_label['8OH_challenge_DH_among_immediate_DH'].denominator_population,
                         'immediate_DH')
        self.assertEqual(by_label['8OH_challenge_DH_among_immediate_DH'].role,
                         'consistency_check_only')
        self.assertEqual(by_label['monensin_immediate_DH'].role, 'condition_setting')
        wire = result.wire()
        self.assertEqual(wire['observation_scope'], 'published_aggregate_count_reconstruction')
        self.assertFalse(wire['primary_source_counts_independently_verified'])
        self.assertFalse(wire['training_admitted'])
        self.assertIsNone(wire['native_drug_mapping'])
        self.assertNotIn('partition_group', wire)
        self.assertTrue(any('8OH chemical identity conflicts' in value
                            for value in wire['admission_blockers']))
        json.dumps(wire, allow_nan=False)

    def test_changing_counts_proportions_or_fit_roles_requires_new_source_review(self):
        payload = count_fixture()
        for bad in (payload.replace(b',143,573,', b',144,573,'),
                    payload.replace(b',143,573,', b',143.0,573,'),
                    payload.replace(b'0.24956369982547993', b'0.9'),
                    payload.replace(b'condition_setting', b'held_out_validation'),
                    payload.replace(b',100,100,1.0,', b',100,100,nan,')):
            self.assertNotEqual(payload, bad, 'test must actually alter the fixture')
            with self.subTest(payload=bad[:120]), self.assertRaises(ValueError):
                counts(bad)

    def test_model_probability_table_is_not_experimental_count_evidence(self):
        payload = (b'quantity,estimate,std_error_delta,lower_95,upper_95,status,interpretation\n'
                   b'p_nigericin_challenge_given_SH,0.149,0.01,0.1,0.2,held_out_prediction,prediction\n')
        with self.assertRaisesRegex(ValueError, 'header'):
            counts(payload)

    def test_count_file_loader_keeps_primary_reference_unverified(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'reconstructed_count_table.csv'
            payload = count_fixture()
            path.write_bytes(payload)
            result = load_reconstructed_counts(
                path, source_commit=SUPPORTED_SOURCE_COMMIT, run_id='fixture',
                expected_sha256=sha256(payload).hexdigest(),
                primary_source_locator='https://pmc.ncbi.nlm.nih.gov/articles/PMC5443973/')
            self.assertEqual(result.primary_source_locator,
                             'https://pmc.ncbi.nlm.nih.gov/articles/PMC5443973/')
            self.assertFalse(result.wire()['primary_source_counts_independently_verified'])


if __name__ == '__main__':
    unittest.main()
