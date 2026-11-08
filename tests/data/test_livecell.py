import copy
import numpy as np
import pytest
from mora3.data.livecell import LiveCellExample


def example(**changes):
    d=dict(image=np.arange(30,dtype=np.uint8).reshape(5,6),
           image_info={'id':7,'file_name':'A172_Phase_C7_1_02d16h00m_2.tif','height':5,'width':6},
           annotations=[{'id':9,'image_id':7,'iscrowd':0,'segmentation':[[1,1,4,1,4,3,1,3]]}],
           split='train',source_sha256='a'*64)
    d.update(changes)
    return LiveCellExample.from_coco(**d)


def test_alignment_and_exact_polygons():
    x=example()
    assert x.polygons==(((1.,1.,4.,1.,4.,3.,1.,3.),),)
    assert x.context['parent_image']=='A172_Phase_C7_1_02d16h00m'
    assert x.context['well']=='C7' and x.context['pixel_size_um'] is None
    assert x.project('image_to_mask')['inputs']['image'].shape==(5,6)


def test_hidden_masks_ids_and_counts_do_not_enter_image_inputs():
    x=example(); ann=[{'id':999,'image_id':7,'iscrowd':0,'segmentation':[[0,0,2,0,2,2]]}]
    y=example(annotations=ann,source_sha256='b'*64)
    for z in (x,y):
        p=z.project('image_to_mask')
        assert set(p['inputs'])=={'image'}
        assert 'polygons' in p['targets']
    assert x.input_metadata('image_to_mask')==y.input_metadata('image_to_mask')
    np.testing.assert_array_equal(x.project('image_to_mask')['inputs']['image'],y.project('image_to_mask')['inputs']['image'])
    assert x.canonical_bytes()!=y.canonical_bytes()


def test_two_masks_roundtrip_and_copy_safety():
    x=example(); y=LiveCellExample.from_bytes(x.canonical_bytes())
    assert x.canonical_bytes()==y.canonical_bytes()
    p=x.project('mask_to_image'); assert set(p['inputs'])=={'polygons'}
    assert set(p['targets'])=={'image'}
    p['targets']['image'][0,0]=255
    assert x.project('image_to_mask')['inputs']['image'][0,0]==0


@pytest.mark.parametrize('change',[
 {'image':np.zeros((6,5),dtype=np.uint8)}, {'split':'unspecified'},
 {'source_sha256':'no'},
 {'annotations':[{'id':1,'image_id':8,'iscrowd':0,'segmentation':[[0,0,1,0,1,1]]}]},
 {'annotations':[{'id':1,'image_id':7,'iscrowd':0,'segmentation':[[0,0,8,0,1,1]]}]},
 {'annotations':[{'id':1,'image_id':7,'iscrowd':1,'segmentation':{'counts':'abc'}}]},
 {'image':np.full((5,6),np.nan)},
])
def test_rejects_invalid_or_unsupported(change):
    with pytest.raises(ValueError): example(**change)


def test_official_holdout_cannot_train():
    for split in ('val','test'):
        x=example(split=split)
        with pytest.raises(ValueError,match='split'): x.require_training()
    example().require_training()


def test_gpu_projection_matches_cpu():
    import torch
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    x=example()
    p=x.project('image_to_mask',device='cuda')
    np.testing.assert_array_equal(p['inputs']['image'].cpu().numpy(),x.project('image_to_mask')['inputs']['image'].numpy())
