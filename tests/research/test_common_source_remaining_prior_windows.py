"""Actual source: four still-unqualified native priors, each exactly two windows.

All outcomes remain read-only. Explicitly exclude protein: its frozen view digest
is not the current base view and may not be rebased by assertion.
"""
from pathlib import Path
import pytest
from mora3.research.common_source_native_replay import audit_common_source_two_remaining_prior_windows

ROOT=Path('/home/iska/Documents/amelie/bio/Mora-3')
ROUTES=('native-prior:rfam_rna','native-prior:rnacentral_rna',
        'native-prior:smiles_to_selfies','native-prior:selfies_to_smiles')


@pytest.mark.parametrize('task',ROUTES)
def test_original_two_windows_with_exact_native_codec_and_restart(task):
    result=audit_common_source_two_remaining_prior_windows(ROOT,task_key=task)
    assert result['schema']=='Mora3.CommonSourceTwoNativeWindows/v1'
    assert result['task_key']==task and result['source_role']=='train'
    assert result['windows_replayed']==result['actual_native_rows']==2
    assert len(result['rows'])==2
    assert result['original_rows_and_payloads_identical'] is True
    assert result['both_rebound_route_cursors_conserved'] is True
    assert result['resumed_second_window_exact_match'] is True
    assert result['independent_target_count_conserved'] is True
    assert result['bounded_original_update_scan_budget']==2
    assert result['source_target_tokens']==sum(x['source_target_tokens']for x in result['rows'])
    assert result['observed_train_target_tokens']==sum(x['admitted_targets']for x in result['rows'])
    assert all(x['source_target_tokens']>0 and x['native_payload_bytes']>0 for x in result['rows'])
    assert all(x['candidate_next_position']==x['common_next_position']for x in result['rows'])
    assert all(x['preparation_status']in('observed','budget_refused')for x in result['rows'])
    assert all(x['admitted_targets']==x['source_target_tokens'] if x['preparation_status']=='observed'
               else x['admitted_targets']==0 and x['original_target_bytes']>4096 for x in result['rows'])
    assert result['candidate_source_snapshot_sha256']!=result['common_source_snapshot_sha256']
    assert result['original_source_corpus_exhausted']is None
    assert result['complete62_route_replay']is False
    assert result['global_split_accepted']is False
    assert result['matched_validation_baseline_measured']is False
    assert result['training_admission']is False
    assert result['optimizer_updates']==result['actual_campaign_runs']==0


@pytest.mark.parametrize('task,max_windows',[
    ('native-prior:protein',2),('native-prior:dna',2),('native-prior:native_smiles',2),
    ('native-prior:rfam_rna',1),('native-prior:rnacentral_rna',3),
    ('native-prior:selfies_to_smiles',True),('native-prior:smiles_to_selfies',0),
])
def test_no_silent_source_view_or_window_policy_override(task,max_windows):
    with pytest.raises(ValueError,match='[Rr]oute|[Bb]udget|[Ww]indow|[Ss]cope'):
        audit_common_source_two_remaining_prior_windows(ROOT,task_key=task,max_windows=max_windows)


def test_protein_has_a_distinct_frozen_original_view_binding():
    from mora3.research.primary_campaign_source_rebind import load_inactive_common_source_rebind
    from mora3.research.primary_source_factory_preflight import document
    from mora3.research.finite_view_compatibility import BASE_POLICY
    context=load_inactive_common_source_rebind(ROOT)
    original=document(ROOT,context.evidence)
    protein=[r for r in original['route_bindings']if r['task_key']=='native-prior:protein']
    assert len(protein)==1
    assert protein[0]['common_view_policy_sha256']=='2888d9f11038f70f9b25a049c8a006d87ca9918de6e94adcb71ec5ee1c9fd34b'
    assert BASE_POLICY=='c3812ecfcfea8a01f81769bf5ef6fb3880e04a0ac054a6349e9272db14765393'
    assert protein[0]['common_view_policy_sha256']!=BASE_POLICY
    with pytest.raises(ValueError):
        audit_common_source_two_remaining_prior_windows(ROOT,task_key='native-prior:protein')
