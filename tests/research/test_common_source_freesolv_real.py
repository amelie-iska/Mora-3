"""Actual original FreeSolv scalar route on both frozen N29 snapshot bindings.

No optimization, training admission, biological inference or altered source data.
"""
from pathlib import Path
import pytest
from mora3.research.common_source_freesolv import audit_original_freesolv_two_windows

ROOT=Path('/home/iska/Documents/amelie/bio/Mora-3')


def test_original_freesolv_two_windows_and_exact_native_scalar_truth():
    result=audit_original_freesolv_two_windows(ROOT)
    assert result['schema']=='Mora3.OriginalFreeSolvTwoWindowSourceReplay/v1'
    assert result['task_key']=='experimental_hydration_free_energy'
    assert result['windows_replayed']==result['native_source_rows']==2
    assert result['source_family']=='FreeSolv' and result['split']=='train'
    assert result['candidate_and_common_source_equal'] is True
    assert result['native_conditions_and_targets_equal'] is True
    assert result['second_window_restart_exact'] is True
    assert result['native_view_policy_sha256']=='c3812ecfcfea8a01f81769bf5ef6fb3880e04a0ac054a6349e9272db14765393'
    assert len(result['rows'])==2
    assert all(row['original_target_kind']=='reported_point'
               and row['original_source_unit']=='kcal/mol'
               and row['exact_numeric_source_projection']
               and row['native_payload_sha256']
               and row['source_ordinal']>=0
               and row['source_payload_bytes']>0
               and row['candidate_next_position']==row['common_next_position']
               and row['candidate_cursor_sha256']!=row['common_cursor_sha256']
               for row in result['rows'])
    assert result['native_observed_targets']==2
    assert len({(row['source_shard_sha256'],row['source_ordinal'])for row in result['rows']})==2
    assert result['complete62_route_replay'] is False
    assert result['validation_baseline_measured'] is False
    assert result['global_split_admitted'] is False
    assert result['training_admission'] is False
    assert result['optimizer_updates']==result['actual_campaign_runs']==0


@pytest.mark.parametrize('task,windows',[
    ('native-prior:dna',2),('experimental_hydration_free_energy',1),
    ('experimental_hydration_free_energy',3),('experimental_hydration_free_energy',True),
    ('experimental_hydration_free_energy',0),('source_reported_experimental_logD7_4',2)])
def test_free_solv_refuses_changed_task_or_source_scan_budget(task,windows):
    with pytest.raises(ValueError):
        audit_original_freesolv_two_windows(ROOT,task_key=task,max_windows=windows)
