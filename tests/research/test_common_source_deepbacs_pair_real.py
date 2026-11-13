"""Genuine DeepBacs forward/reverse source rasters; no gradient or training."""
from copy import deepcopy
from pathlib import Path

import pytest

from mora3.research.common_source_deepbacs_pair import (
    audit_original_deepbacs_two_direction_pairs,
    require_deepbacs_source_pair,
)

ROOT = Path('/home/iska/Documents/amelie/bio/Mora-3')


def test_two_original_rasters_replay_under_both_frozen_snapshots():
    result = audit_original_deepbacs_two_direction_pairs(ROOT)
    assert result['schema'] == 'Mora3.DeepBacsOriginalDirectionalPairs/v1'
    assert result['original_raster_count'] == 2
    assert result['original_route_rows'] == 4
    assert result['original_source_target_positions'] == 2095104
    assert result['direction_target_positions'] == {
        'image_to_mask': 1570816, 'mask_to_image': 524288,
    }
    assert result['candidate_common_original_rows_equal'] is True
    assert result['independent_second_window_restarts'] is True
    assert result['training_admission'] is False
    assert result['actual_campaign_runs'] == result['optimizer_updates'] == 0
    assert result['source_test_development_exposed'] is True
    assert result['global_group_split_accepted'] is False
    assert result['matched_owner_validation_complete'] is False
    assert [p['forward_ordinal'] for p in result['paired_rows']] == [0, 2]
    assert [p['reverse_ordinal'] for p in result['paired_rows']] == [1, 3]
    assert len({p['image_sha256'] for p in result['paired_rows']}) == 2
    for row in result['paired_rows']:
        assert row['image_shape'] == [512, 512]
        assert row['image_dtype'] == '>u2' and row['label_dtype'] == '>u2'
        assert row['source_image_bytes'] == row['source_mask_bytes'] == 524288
        assert row['forward_target_positions'] == 785408
        assert row['reverse_target_positions'] == 262144
        assert row['source_license'] == 'CC-BY-4.0'
        # Provider file/member digests must not be conflated with decoded pixels.
        assert row['decoded_image_sha256'] != row['image_sha256']
        assert row['decoded_mask_sha256'] != row['mask_sha256']
        assert row['condition_target_roles'] == {
            'image_to_mask': {'conditions': ['image'], 'targets': ['labels']},
            'mask_to_image': {'conditions': ['labels'], 'targets': ['image']},
        }
        assert row['candidate_common_cursor_positions_equal'] is True
        for direction in ('image_to_mask', 'mask_to_image'):
            assert row[f'{direction}_source_fingerprint']
            assert row[f'candidate_{direction}_cursor_sha256'] != row[f'common_{direction}_cursor_sha256']


@pytest.mark.parametrize('directions,limit', [
    (('image_to_mask',), 2),
    (('mask_to_image', 'image_to_mask'), 2),
    (('image_to_mask', 'mask_to_image'), 1),
    (('image_to_mask', 'mask_to_image'), 3),
    (('image_to_mask', 'mask_to_image'), True),
])
def test_reject_unregistered_source_scope(directions, limit):
    with pytest.raises(ValueError):
        audit_original_deepbacs_two_direction_pairs(ROOT, directions=directions, max_windows=limit)


def test_refuse_wrong_original_root(tmp_path):
    with pytest.raises(ValueError):
        audit_original_deepbacs_two_direction_pairs(tmp_path)


def test_pair_requires_exact_source_identity_and_inverse_roles():
    base = dict(
        direction='image_to_mask', source_id='parent/field-1',
        original_split='train', parent_group='parent/field-1',
        source_shard_sha256='a'*64, image_sha256='b'*64, mask_sha256='c'*64,
        decoded_image_sha256='d'*64, decoded_mask_sha256='e'*64,
        image_shape=[512,512], image_dtype='>u2', label_dtype='>u2',
        source_license='CC-BY-4.0', source_revision='Zenodo5550933.v1',
        source_image_bytes=524288, source_mask_bytes=524288,
        source_ordinal=0, observed_positions=785408,
        condition_keys=('image',), target_keys=('labels',),
    )
    reverse = dict(base, direction='mask_to_image', source_ordinal=1,
        observed_positions=262144, condition_keys=('labels',), target_keys=('image',))
    assert require_deepbacs_source_pair(base, reverse)['image_sha256'] == 'b'*64
    for key, value in [
        ('source_id','different'), ('parent_group','other'), ('mask_sha256','d'*64),
        ('image_shape',[512,511]), ('source_shard_sha256','e'*64),
        ('source_license','no-rights'), ('original_split','test'),
        ('target_keys',('labels',)), ('source_ordinal',0),
        ('source_image_bytes',23), ('observed_positions',0),
    ]:
        altered = deepcopy(reverse)
        altered[key] = value
        with pytest.raises(ValueError):
            require_deepbacs_source_pair(base, altered)
