"""Two REAL original train windows: paired cursors, target counts and restart."""
from pathlib import Path
import pytest
from mora3.research.common_source_native_replay import audit_common_source_two_prior_windows

ROOT=Path('/home/iska/Documents/amelie/bio/Mora-3')


def test_actual_two_native_train_windows_and_exact_independent_restarts():
    result=audit_common_source_two_prior_windows(ROOT)
    assert result['schema']=='Mora3.CommonSourceTwoNativeWindows/v1'
    assert result['task_key']=='native-prior:dna'
    assert result['windows_replayed']==2
    assert result['actual_native_rows']==2
    assert len(result['rows'])==2
    assert result['original_rows_and_payloads_identical'] is True
    assert result['source_role']=='train'
    assert result['both_rebound_route_cursors_conserved'] is True
    assert result['resumed_second_window_exact_match'] is True
    assert result['independent_target_count_conserved'] is True
    assert result['observed_train_target_tokens']==sum(r['admitted_targets'] for r in result['rows'])
    assert result['source_target_tokens']==sum(r['source_target_tokens'] for r in result['rows'])
    assert all(r['source_target_tokens']>0 and r['native_payload_bytes']>0 for r in result['rows'])
    assert result['rows'][0]['preparation_status']=='observed'
    assert result['rows'][1]['preparation_status']=='budget_refused'
    assert result['rows'][1]['original_target_bytes']>4096
    assert result['rows'][1]['admitted_targets']==0
    assert all(r['candidate_next_position']==r['common_next_position'] for r in result['rows'])
    assert result['rows'][0]['native_payload_sha256']!=result['rows'][1]['native_payload_sha256']
    assert (result['rows'][0]['source_shard_sha256'],result['rows'][0]['source_ordinal']) != (result['rows'][1]['source_shard_sha256'],result['rows'][1]['source_ordinal'])
    assert result['training_admission'] is False
    assert result['complete62_route_replay'] is False
    assert result['global_split_accepted'] is False
    assert result['optimizer_updates']==result['actual_campaign_runs']==0


@pytest.mark.parametrize('task,max_windows',[
    ('native-prior:dna',1),('native-prior:dna',3),('native-prior:protein',2),
    ('native-prior:dna',True),('native-prior:dna',0),
])
def test_two_window_contract_refuses_changed_scope_before_source_io(task,max_windows):
    with pytest.raises(ValueError,match='two'):
        audit_common_source_two_prior_windows(ROOT,task_key=task,max_windows=max_windows)


def test_cursor_transition_rejects_backward_equal_or_mismatched_coordinates():
    from mora3.research.common_source_native_replay import _checked_two_window_transition
    for a,b in [((0,0),(0,0)),((0,5),(0,4)),((1,0),(0,100)),((0,0),(1,-1))]:
        with pytest.raises(ValueError):_checked_two_window_transition(a,b)
    assert _checked_two_window_transition((0,0),(0,1))==(0,1)
    assert _checked_two_window_transition((0,900),(1,0))==(1,0)


def test_cursor_position_binding_refuses_cross_snapshot_replay():
    from mora3.research.common_source_native_replay import _checked_rebound_position
    with pytest.raises(ValueError):_checked_rebound_position((0,1),(0,2))
    assert _checked_rebound_position((0,1),(0,1))==(0,1)
