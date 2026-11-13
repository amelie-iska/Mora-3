"""Genuine original sci-Plex2 source-cell paired route fidelity; no training."""
from pathlib import Path
import pytest
from mora3.research.common_source_gperturb_pair import (
    audit_original_gperturb_two_direction_pairs, require_paired_source_semantics,
)

ROOT=Path('/home/iska/Documents/amelie/bio/Mora-3')


def test_genuine_two_original_cells_have_exact_opposite_condition_target_roles():
    doc=audit_original_gperturb_two_direction_pairs(ROOT)
    assert doc['schema']=='Mora3.GPerturbSourcePairedRoutesTwoWindows/v1'
    assert doc['source_family']=='GPerturb-sciPlex2'
    assert doc['source_cells']==2 and doc['original_route_rows']==4
    assert doc['original_source_targets']==10004
    assert doc['direction_targets']=={'response':10000,'intervention':4}
    assert doc['candidate_common_source_equal'] is True
    assert doc['independent_second_window_restarts'] is True
    assert doc['shared_original_source_cells'] is True
    assert doc['source_group_heldout_accepted'] is False
    assert doc['physical_concentration_resolved'] is False
    assert doc['paired_causal_treatment_evidence'] is False
    assert doc['training_admission'] is False
    assert doc['optimizer_updates']==doc['actual_campaign_runs']==0
    assert [p['source_index'] for p in doc['paired_rows']]==[0,1]
    assert [p['intervention']['identity']for p in doc['paired_rows']]==['Dex','Nutlin']
    assert [p['intervention']['source_coordinate']for p in doc['paired_rows']]==[.001,.005]
    assert [p['response_ordinal']for p in doc['paired_rows']]==[0,2]
    assert [p['intervention_ordinal']for p in doc['paired_rows']]==[1,3]
    assert len({p['cell_id_hash'] for p in doc['paired_rows']})==2
    assert len({p['response_expression_sha256'] for p in doc['paired_rows']})==2
    for p in doc['paired_rows']:
        assert p['expression_axis_length']==5000
        assert p['intervention']['physical_concentration'] is None
        assert p['source_context']==dict(study='GSM4150377',cell_line='A549',
            exposure_hours=24,baseline=None,desired_response=None)
        assert p['condition_target_roles']==dict(
            response={'conditions':['context','intervention'],'targets':['expression']},
            intervention={'conditions':['context','expression'],'targets':['intervention']})
        assert p['response_expression_sha256']==p['intervention_condition_expression_sha256']
        assert p['target_sidecar_training_ready'] is False
        assert p['observational_train_only'] is True
        assert p['original_source_revision']
        assert p['candidate_common_cursor_positions_equal'] is True
        for direction in ('response','intervention'):
            assert len(p[f'{direction}_native_source_fingerprint'])==64
            assert p[f'candidate_{direction}_cursor_sha256']!=p[f'common_{direction}_cursor_sha256']
            assert tuple(p[f'{direction}_next_position'])


@pytest.mark.parametrize('directions,windows',[
    (('response',),2),(('intervention','response'),1),
    (('intervention','response'),3),(('intervention','response'),True),
    (('intervention','response'),0),(('response','response'),2),
])
def test_refuse_nonfrozen_source_scope_before_opening_any_row(directions,windows):
    with pytest.raises(ValueError):
        audit_original_gperturb_two_direction_pairs(ROOT,directions=directions,max_windows=windows)


def test_refuse_wrong_original_checkout_before_native_source_reader(tmp_path):
    with pytest.raises(ValueError):audit_original_gperturb_two_direction_pairs(tmp_path)


def test_pair_semantics_refuse_swapped_labels_study_or_source_cells():
    base=dict(source_index=0,cell_id='original-cell',source_shard_sha256='a'*64,
        source_context=dict(study='GSM4150377',cell_line='A549',exposure_hours=24,
            baseline=None,desired_response=None),expression_sha256='b'*64,
        axis_sha256='c'*64,intervention=dict(identity='Dex',source_coordinate=.001,
            physical_concentration=None,units='source_scaled_coordinate',is_vehicle=False),
        original_group='study:GSE139944:GSM4150377')
    left=dict(base,direction='response',condition_keys=('context','intervention'),
        target_keys=('expression',))
    right=dict(base,direction='intervention',condition_keys=('context','expression'),
        target_keys=('intervention',))
    assert require_paired_source_semantics(left,right)['source_index']==0
    for changes in (dict(source_index=1),dict(cell_id='different'),
                    dict(expression_sha256='d'*64),dict(axis_sha256='e'*64),
                    dict(source_context=dict(base['source_context'],exposure_hours=48)),
                    dict(target_keys=('expression',)),dict(original_group='unknown'),
                    dict(intervention=dict(base['intervention'],physical_concentration=.001))):
        with pytest.raises(ValueError):
            require_paired_source_semantics(left,right|changes)
