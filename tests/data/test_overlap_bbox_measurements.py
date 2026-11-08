"""Exact opt-in pruning must preserve complete independent-slot results."""
from dataclasses import replace
import pytest
import torch
from test_segmentation_measurements import image
from mora3.data.segmentation_measurements import measure_predicted_mask_set


def measure(masks, **kw):
    return measure_predicted_mask_set(masks, replace(image(), spatial_shape=tuple(masks.shape[1:])),
        checkpoint='diagnostic:v1', mask_asset='new-mask.npy', max_mask_bytes=masks.numel(), **kw)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('kind', ['empty', 'duplicate', 'touching', 'disjoint', 'holes', 'random'])
def test_exact_bbox_results_and_nonmutation(device, kind):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    m = torch.zeros(5, 12, 15, dtype=torch.bool)
    if kind != 'empty': m[0, 1:5, 1:5] = True
    if kind == 'duplicate': m[1] = m[0]
    if kind == 'touching': m[1, 1:5, 5:8] = True
    if kind == 'disjoint': m[1, 8:11, 10:14] = True
    if kind == 'holes':
        m[1, 1:5, 1:5] = True
        m[0, 2:4, 2:4] = False
        m[1, 1, 1] = False
    if kind == 'random': m = torch.rand((5,12,15), generator=torch.Generator().manual_seed(35701)) > .65
    m = m.to(device); before = m.clone()
    expected = measure(m)
    assert measure(m, overlap_method='bbox_exact') == expected
    assert torch.equal(m, before)
    if kind == 'duplicate': assert expected['edges'][0]['overlap_px'] == 16
    if kind in ('empty', 'touching', 'disjoint'): assert not expected['edges']


@pytest.mark.parametrize('bad', [None, True, 1, [], 'bbox', 'approximate'])
def test_invalid_method_refuses(bad):
    with pytest.raises(ValueError, match='overlap method'):
        measure(torch.zeros(1,2,3,dtype=torch.bool), overlap_method=bad)
