"""Two genuine original native SMILES windows under the reviewed common 29-head snapshot.

Read-only provenance/cursor/target verification; never counts as training.
"""
from pathlib import Path
import pytest
from mora3.research.common_source_native_replay import (
    audit_common_source_two_native_smiles_windows,
)
ROOT=Path('/home/iska/Documents/amelie/bio/Mora-3')


def test_two_real_native_smiles_windows_preserve_original_targets_and_cursor_restart():
    receipt=audit_common_source_two_native_smiles_windows(ROOT)
    assert receipt['schema']=='Mora3.CommonSourceTwoNativeWindows/v1'
    assert receipt['task_key']=='native-prior:native_smiles'
    assert receipt['source_role']=='train'
    assert receipt['windows_replayed']==2 and receipt['actual_native_rows']==2
    assert len(receipt['rows'])==2
    assert receipt['bounded_original_update_scan_budget']==2
    assert receipt['original_rows_and_payloads_identical'] is True
    assert receipt['both_rebound_route_cursors_conserved'] is True
    assert receipt['resumed_second_window_exact_match'] is True
    assert receipt['independent_target_count_conserved'] is True
    assert receipt['source_target_tokens']==sum(r['source_target_tokens'] for r in receipt['rows'])
    assert receipt['observed_train_target_tokens']==sum(r['admitted_targets'] for r in receipt['rows'])
    assert all(r['source_target_tokens']>0 for r in receipt['rows'])
    assert all(r['native_payload_bytes']>0 and len(r['native_payload_sha256'])==64 for r in receipt['rows'])
    assert all(r['candidate_next_position']==r['common_next_position'] for r in receipt['rows'])
    assert all(r['preparation_status'] in ('observed','budget_refused') for r in receipt['rows'])
    assert all((r['admitted_targets']==r['source_target_tokens']) if r['preparation_status']=='observed'
        else (r['admitted_targets']==0 and r['original_target_bytes']>4096) for r in receipt['rows'])
    assert not receipt['complete62_route_replay']
    assert not receipt['global_split_accepted']
    assert not receipt['matched_validation_baseline_measured']
    assert not receipt['training_admission']
    assert receipt['optimizer_updates']==receipt['actual_campaign_runs']==0


@pytest.mark.parametrize('task,max_windows',[
    ('native-prior:dna',2),('native-prior:protein',2),
    ('native-prior:native_smiles',1),('native-prior:native_smiles',3),
    ('native-prior:native_smiles',True),
])
def test_smiles_scope_refuses_changed_task_or_budget_before_source_io(task,max_windows):
    with pytest.raises(ValueError):
        audit_common_source_two_native_smiles_windows(ROOT,task_key=task,max_windows=max_windows)
