"""Read actual original UniProt kinetic rows, never assume enzyme kinetic causality."""
from pathlib import Path
import pytest
from mora3.research.common_source_uniprot_kinetics import audit_original_uniprot_kinetics_two_windows

ROOT=Path('/home/iska/Documents/amelie/bio/Mora-3')
TASK='curated_kinetic_annotation_value'


def test_original_two_kinetic_windows_preserve_exact_native_measurement_semantics():
    result=audit_original_uniprot_kinetics_two_windows(ROOT)
    assert result['schema']=='Mora3.OriginalUniProtKineticsTwoWindowSourceReplay/v1'
    assert result['task_key']==TASK and result['source_family']=='UniProt-kinetics'
    assert result['split']=='train' and result['windows_replayed']==2
    assert result['native_source_rows']==result['native_observed_targets']==2
    assert result['candidate_and_common_source_equal'] is True
    assert result['native_conditions_and_targets_equal'] is True
    assert result['second_window_restart_exact'] is True
    assert len(result['rows'])==2
    assert [row['original_numeric_lexeme'] for row in result['rows']]==['2.6','11.7']
    assert [row['measurement_type'] for row in result['rows']]==['Km','Km']
    assert [row['literal_source_unit'] for row in result['rows']]==['uM','uM']
    assert len({(row['source_shard_sha256'],row['source_ordinal']) for row in result['rows']})==2
    assert len({row['condition_sha256']for row in result['rows']})==2
    for row in result['rows']:
        assert row['original_target_kind']=='reported_point'
        assert row['measurement_type']!='Vmax'
        assert row['quantity_coordinate']=='source_curated_reported_Km'
        assert row['kinetic_alias_role']=='substrate_alias'
        assert row['assay_method']is None and row['assayed_construct']is None
        assert row['physical_missingness']==['assay_method','assay_pH','assay_temperature',
                                            'assayed_construct','parent_comment_molecule_scope']
        assert row['candidate_next_position']==row['common_next_position']
        assert row['candidate_cursor_sha256']!=row['common_cursor_sha256']
        assert row['native_source_fingerprint'] and row['provider_lexemes_sha256']
        assert row['native_numeric_coordinate_exact'] is True
        assert row['kinetic_unit_converted'] is False
        assert row['training_objective_invoked'] is False
    assert result['validation_baseline_measured'] is False
    assert result['global_split_admitted'] is False
    assert result['training_admission'] is False
    assert result['optimizer_updates']==result['actual_campaign_runs']==0


@pytest.mark.parametrize('task,windows',[
    ('source_reported_experimental_logD7_4',2),(TASK,1),(TASK,3),
    (TASK,True),(TASK,0),('experimental_hydration_free_energy',2)])
def test_kinetic_replay_refuses_foreign_tasks_and_nonfrozen_window_count(task,windows):
    with pytest.raises(ValueError):
        audit_original_uniprot_kinetics_two_windows(ROOT,task_key=task,max_windows=windows)
