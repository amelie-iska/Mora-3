"""Full original DeepBacs train census and source-group provenance; zero training."""
from pathlib import Path

import pytest

from mora3.research.common_source_deepbacs_census import (
    audit_original_deepbacs_complete_train,
    require_role_group_census,
)

ROOT = Path('/home/iska/Documents/amelie/bio/Mora-3')


def test_complete_original_train_and_group_roles():
    evidence = audit_original_deepbacs_complete_train(ROOT)
    assert evidence['schema'] == 'Mora3.DeepBacsCompleteOriginalTrainCensus/v1'
    assert evidence['original_train_acquisitions'] == 7
    assert evidence['original_train_task_rows'] == 14
    assert evidence['original_test_acquisitions_metadata_only'] == 5
    assert evidence['original_test_task_rows_metadata_only'] == 10
    assert evidence['original_validation_task_rows'] == 0
    assert evidence['complete_both_native_train_routes'] is True
    assert evidence['candidate_common_all_original_train_rows_equal'] is True
    assert evidence['direction_target_positions'] == {
        'image_to_mask': 5497856, 'mask_to_image': 1835008,
    }
    assert evidence['original_source_target_positions'] == 7332864
    assert [x['forward_ordinal'] for x in evidence['original_train_pairs']] == list(range(0, 14, 2))
    assert [x['reverse_ordinal'] for x in evidence['original_train_pairs']] == list(range(1, 14, 2))
    assert evidence['train_test_provider_series_shared'] is True
    assert evidence['independent_heldout_accepted'] is False
    assert evidence['training_admission'] is False
    assert evidence['optimizer_updates'] == evidence['actual_campaign_runs'] == 0


def test_census_rejects_wrong_paired_group_and_false_split_overlap():
    train = {'train': [('first', 'parent/a'), ('second','parent/b')],
             'test': [('third', 'parent/c')]}
    result = require_role_group_census(train)
    assert result['literal_parent_group_overlap'] == []
    with pytest.raises(ValueError, match='duplicate'):
        require_role_group_census({'train': [('a','parent/x'), ('a','parent/x')], 'test': []})
    with pytest.raises(ValueError, match='role'):
        require_role_group_census({'train': [('a','parent/x')]})
    overlap = require_role_group_census({'train': [('a','parent/x')], 'test': [('b','parent/x')]})
    assert overlap['literal_parent_group_overlap'] == ['parent/x']
    assert overlap['independent_heldout_accepted'] is False


def test_source_scope_does_not_accept_changed_root(tmp_path):
    with pytest.raises(ValueError):
        audit_original_deepbacs_complete_train(tmp_path)
