"""Independent plane geometry: overlapping memberships must never be flattened."""
from dataclasses import replace
import pytest
import torch
from test_segmentation_measurements import image
from mora3.data import segmentation_measurements as measurements


def run(masks, im=None, **kw):
    return measurements.measure_predicted_mask_set(masks, image() if im is None else im,
        checkpoint='segmenter:v2', mask_asset='new-prediction.pt', max_mask_bytes=1024, **kw)


@pytest.mark.parametrize('device', ['cpu','cuda'])
def test_overlapping_duplicate_and_empty_slots_keep_native_geometry(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    masks=torch.tensor([[[1,1,0],[0,1,0]],[[0,1,1],[0,0,0]],
                        [[0,0,0],[0,0,0]],[[1,1,0],[0,1,0]]],dtype=torch.bool,device=device)
    old=masks.clone(); result=run(masks,image(pixel_size_um=.5))
    assert torch.equal(masks,old)
    a,b,c=result['objects']
    assert a==dict(object_id=1,area_px=3,centroid_yx=[1/3,2/3],
        bbox_yxyx=[0,0,2,2],perimeter_px=8,area_um2=.75)
    assert b==dict(object_id=2,area_px=2,centroid_yx=[0.,1.5],
        bbox_yxyx=[0,1,1,3],perimeter_px=6,area_um2=.5)
    assert c==dict(a,object_id=4)
    assert result['empty_slot_ids']==[3] and result['slots']==4
    assert result['edges']==[
        dict(source=1,target=2,relation='pixel_membership_overlap',overlap_px=1),
        dict(source=1,target=4,relation='pixel_membership_overlap',overlap_px=3),
        dict(source=2,target=4,relation='pixel_membership_overlap',overlap_px=1)]
    assert result['artifact'].representation=='overlapping-slot-object-geometry-v1'
    assert result['artifact'].links==(('objects_of','image'),)
    assert result['artifact'].evidence=='model_predicted'
    assert result['artifact'].uncertainty=='uncalibrated'
    assert not result['biological_validation']


def test_all_empty_missing_scale_and_generated_provenance():
    result=run(torch.zeros(2,2,3,dtype=torch.bool),
        image(evidence='model_generated',checkpoint='generator:v1'))
    assert result['objects']==[] and result['edges']==[]
    assert result['empty_slot_ids']==[1,2]
    assert result['image_evidence']=='model_generated'
    assert 'model_consistency_only' in result['artifact'].quality_flags
    result=run(torch.ones(1,2,3,dtype=torch.bool))
    assert result['objects'][0]['area_um2'] is None
    assert result['objects'][0]['perimeter_px']==10


@pytest.mark.parametrize('case',['float','rank','shape','zero_slots','too_many_slots',
    'target','evaluation','desired','asset','budget','bool_budget','frame'])
def test_refuses_invalid_or_unbound_sets(case):
    masks=torch.zeros(2,2,3,dtype=torch.bool);im=image();kw={}
    if case=='float':masks=masks.float()
    if case=='rank':masks=masks[0]
    if case=='shape':masks=masks[:,:,:2]
    if case=='zero_slots':masks=masks[:0]
    if case=='too_many_slots':masks=torch.zeros(513,2,3,dtype=torch.bool)
    if case in ('target','evaluation'):im=replace(im,role=case)
    if case=='desired':im=replace(im,role='desired',evidence='desired')
    if case=='asset':im=replace(im,asset='new-prediction.pt')
    if case=='frame':im=replace(im,reference_frame='unknown')
    with pytest.raises(ValueError):
        if case in ('budget','bool_budget'):
            measurements.measure_predicted_mask_set(masks,im,checkpoint='s',mask_asset='p',
                max_mask_bytes=1 if case=='budget' else True)
        else:run(masks,im,**kw)
