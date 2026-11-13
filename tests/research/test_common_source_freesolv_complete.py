"""All original FreeSolv train rows, with held-out provenance only; no updates."""
from hashlib import sha256
from pathlib import Path
import json
import struct

import pytest
from mora3.research.common_source_freesolv import (
    audit_original_freesolv_complete_train,
    freesolv_declared_group_census,
    _reference_freesolv_numeric,
)

ROOT = Path('/home/iska/Documents/amelie/bio/Mora-3')


@pytest.fixture(scope='module')
def complete():
    import pyarrow as pa
    import pyarrow.parquet as pq
    from mora3.data.finite_union_native_rows import FiniteUnionNativeView
    heldout = {'f82fa116530c9423fba100a4117fd602ec221d8d8b1e49578f28d8adb0826cfa',
               '6c1dc7d2c89505f8c6980ceea4a7c4e2b9f925268ee0f52c93718382098bde16'}
    observed = set()
    original_read = pq.read_table
    original_require = FiniteUnionNativeView.require_native_rows

    def read(source, *args, **kwargs):
        if isinstance(source, pa.BufferReader):
            position = source.tell()
            source.seek(0); digest = sha256(source.read()).hexdigest(); source.seek(position)
            if digest in heldout:
                assert kwargs['columns'] == ['task_id', 'record_id', 'source_family', 'source_revision',
                                              'split', 'task_name', 'native_schema']
                observed.add(digest)
        return original_read(source, *args, **kwargs)

    def require(view, batch, *, split):
        if '/FreeSolv/' in batch.source_shard['path']:
            assert split == batch.original_split == 'train', 'No held-out native target decoding is permitted'
        return original_require(view, batch, split=split)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pq, 'read_table', read)
        patch.setattr(FiniteUnionNativeView, 'require_native_rows', require)
        result = audit_original_freesolv_complete_train(ROOT)
    assert observed == heldout
    return result


def test_complete_original_train_route_not_another_two_row_demonstration(complete):
    d = complete
    assert d['schema'] == 'Mora3.FreeSolvCompleteOriginalTrainReplay/v1'
    assert d['original_role_rows'] == {'train': 556, 'validation': 42, 'test': 44}
    assert d['native_train_rows'] == d['native_observed_train_targets'] == 556
    assert d['candidate_common_all_train_rows_equal'] is True
    assert d['all_train_numeric_references_equal'] is True
    assert d['both_train_iterators_exhausted'] is True
    assert d['frozen_per_trial_window_limit_unchanged'] == 2
    assert d['train_final_cursor_position'] == [3, 0]
    assert d['source_boundary_restarts_checked'] == 4
    assert [d['rows'][i]['next_position'] for i in (255, 511, 555)] == [[1, 0], [2, 0], [3, 0]]
    assert len(d['rows']) == len({(r['source_shard_sha256'], r['source_ordinal']) for r in d['rows']}) == 556
    for row in d['rows']:
        assert row['source_unit'] == 'kcal/mol'
        assert row['candidate_cursor_sha256'] != row['common_cursor_sha256']
        assert row['original_value_float64_bits'] == row['independent_value_float64_bits']
        assert row['original_uncertainty_float64_bits'] == row['independent_uncertainty_float64_bits']
        assert len(row['native_payload_sha256']) == len(row['condition_sha256']) == 64


def test_all_original_role_groups_preserved_without_promoting_a_clean_benchmark(complete):
    d = complete
    g = d['declared_group_census']
    assert g['role_rows'] == {'train': 556, 'validation': 42, 'test': 44}
    assert g['distinct_molecule_groups_by_role'] == {'train': 553, 'validation': 42, 'test': 44}
    assert g['distinct_molecule_groups_total'] == 639
    assert g['same_molecule_extra_rows_by_role'] == {'train': 3, 'validation': 0, 'test': 0}
    assert g['cross_role_connected_components'] == []
    assert g['source_local_declared_groups_disjoint'] is True
    assert g['global_chemical_split_accepted'] is False
    assert d['heldout_target_payload_decoded'] is False
    assert d['heldout_metadata_rows'] == 86
    assert d['heldout_decoded_columns'] == ['task_id', 'record_id', 'source_family', 'source_revision', 'split', 'task_name', 'native_schema']
    assert len(d['source_files']) == 5
    for key in ('full_expanded_corpus_ready', 'complete62_route_replay', 'fresh_owner_validation_reference',
                'global_split_admitted', 'training_admission'):
        assert d[key] is False
    assert d['optimizer_updates'] == d['actual_proxy_trials'] == d['actual_target_runs'] == 0


def _row(index, role, component, molecule, scaffold):
    return dict(record_id=f'mobley_{index}', task_id=f'{index:064x}', split=role,
                groups=(f'component:{component:064x}', f'molecule:{molecule:064x}', f'scaffold:{scaffold:064x}'))


def test_declared_group_transitive_closure_reports_a_bridge_across_roles():
    rows = (_row(1, 'train', 10, 20, 30), _row(2, 'train', 11, 21, 30),
            _row(3, 'validation', 11, 22, 31), _row(4, 'test', 12, 23, 32))
    d = freesolv_declared_group_census(rows)
    assert d['connected_components'] == 2
    assert len(d['cross_role_connected_components']) == 1
    assert d['cross_role_connected_components'][0]['roles'] == ['train', 'validation']
    assert d['cross_role_connected_components'][0]['record_count'] == 3
    assert d['source_local_declared_groups_disjoint'] is False
    assert d == freesolv_declared_group_census(tuple(reversed(rows)))
    # A within-train repeated molecule is retained, not silently deduplicated.
    duplicate = rows + (_row(5, 'train', 10, 20, 30),)
    assert freesolv_declared_group_census(duplicate)['same_molecule_extra_rows_by_role']['train'] == 1


@pytest.mark.parametrize('damage', ('duplicate_record', 'duplicate_task', 'bad_role', 'bool_group', 'missing_axis', 'unknown_axis'))
def test_declared_group_census_rejects_incomplete_or_ambiguous_original_identity(damage):
    rows = [_row(1, 'train', 10, 20, 30), _row(2, 'validation', 11, 21, 31)]
    if damage == 'duplicate_record': rows[1]['record_id'] = rows[0]['record_id']
    elif damage == 'duplicate_task': rows[1]['task_id'] = rows[0]['task_id']
    elif damage == 'bad_role': rows[1]['split'] = 'tuning'
    elif damage == 'bool_group': rows[1]['groups'] = (True, *rows[1]['groups'][1:])
    elif damage == 'missing_axis': rows[1]['groups'] = rows[1]['groups'][:2]
    else: rows[1]['groups'] = ('context:' + 'f'*64, *rows[1]['groups'][1:])
    with pytest.raises(ValueError): freesolv_declared_group_census(tuple(rows))


def _packet():
    return dict(schema='mora3.CorpusTask/v1', task='experimental_hydration_free_energy',
                conditions={'smiles': 'O'}, targets={'expt': -6.0, 'unit': 'kcal/mol'}, missing=[],
                audit={'source_row': {'expt': -6.0, 'd_expt': 0.2},
                       'label_contract': {'default_target_fields': ['expt'],
                         'label_evidence': {'expt': 'experimental', 'd_expt': 'experimental_uncertainty_may_be_default'}}})


def test_independent_numeric_reference_preserves_native_bits_lexeme_and_uncertainty_class():
    raw = json.dumps(_packet()).encode().replace(b'-6.0', b'-6.000e0')
    d = _reference_freesolv_numeric(raw)
    assert d['value_float64_bits'] == struct.pack('>d', -6.0).hex()
    assert d['uncertainty_float64_bits'] == struct.pack('>d', 0.2).hex()
    assert d['numeric_lexeme'] == '-6.000e0'
    assert d['uncertainty_kind'] == 'experimental_uncertainty_may_be_default'


@pytest.mark.parametrize('damage', ('bool', 'nonfinite', 'wrong_unit', 'source_mismatch', 'predicted', 'invented_uncertainty', 'target_in_conditions'))
def test_independent_numeric_reference_refuses_relabeling_or_target_exposure(damage):
    p = _packet()
    if damage == 'bool': p['targets']['expt'] = p['audit']['source_row']['expt'] = True
    elif damage == 'nonfinite': p['targets']['expt'] = float('nan')
    elif damage == 'wrong_unit': p['targets']['unit'] = 'kJ/mol'
    elif damage == 'source_mismatch': p['audit']['source_row']['expt'] = 1.0
    elif damage == 'predicted': p['audit']['label_contract']['label_evidence']['expt'] = 'computed'
    elif damage == 'invented_uncertainty': p['audit']['label_contract']['label_evidence']['d_expt'] = 'known_standard_deviation'
    else: p['conditions']['expt'] = -6.0
    with pytest.raises(ValueError): _reference_freesolv_numeric(json.dumps(p).encode())


def test_original_checkout_is_required_before_any_source_read(tmp_path):
    with pytest.raises(ValueError): audit_original_freesolv_complete_train(tmp_path)
