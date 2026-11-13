"""Actual ChEMBL37 AZ logD7.4 source semantics, not laboratory causality.

Reads two bounded original source rows; source-of-truth data are immutable.
"""
from pathlib import Path
import pytest
from mora3.research.common_source_logd import audit_original_logd_two_windows

ROOT=Path('/home/iska/Documents/amelie/bio/Mora-3')
TASK='source_reported_experimental_logD7_4'


def test_real_logd_original_rows_and_source_semantic_qualification():
    result=audit_original_logd_two_windows(ROOT)
    assert result['schema']=='Mora3.OriginalAZLogDTwoWindowSourceReplay/v1'
    assert result['task_key']==TASK
    assert result['source_family']=='ChEMBL37-AZ-experimental-logD7_4'
    assert result['split']=='train' and result['windows_replayed']==2
    assert result['native_source_rows']==result['native_observed_targets']==2
    assert result['candidate_and_common_source_equal'] is True
    assert result['native_conditions_and_targets_equal'] is True
    assert result['second_window_restart_exact'] is True
    assert len(result['rows'])==2
    assert len({(row['source_shard_sha256'],row['source_ordinal']) for row in result['rows']})==2
    assert [row['original_numeric_lexeme'] for row in result['rows']]==['1.8','1.8']
    assert len({row['condition_sha256'] for row in result['rows']})==2
    # Distinct source rows and conditions do NOT establish independent assays.
    for row in result['rows']:
        assert row['original_target_kind']=='reported_nominal_coordinate'
        assert row['literal_source_unit'] is None
        assert row['source_relation'] is None
        assert row['source_coordinate_scope']=='declared_native_nominal_logD7_4_coordinate_relation_unspecified'
        assert row['exact_numeric_source_projection'] is True
        assert row['candidate_next_position']==row['common_next_position']
        assert row['candidate_cursor_sha256']!=row['common_cursor_sha256']
        assert row['original_numeric_lexeme']
        assert row['original_native_payload_sha256']
        assert row['assay_pH']==7.4
        assert row['source_semantics']!='independently-verified-experimental-effect'
    assert result['validation_baseline_measured'] is False
    assert result['global_split_admitted'] is False
    assert result['training_admission'] is False
    assert result['optimizer_updates']==result['actual_campaign_runs']==0


@pytest.mark.parametrize('task,windows',[
    ('experimental_hydration_free_energy',2),(TASK,1),(TASK,3),
    (TASK,True),(TASK,0),('curated_kinetic_annotation_value',2)])
def test_original_logd_strict_source_scope_and_window_budget(task,windows):
    with pytest.raises(ValueError):
        audit_original_logd_two_windows(ROOT,task_key=task,max_windows=windows)
