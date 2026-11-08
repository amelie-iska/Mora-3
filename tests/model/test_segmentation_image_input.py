from dataclasses import replace
import hashlib
import numpy as np
import pytest
import torch
from test_segmentation_inference import setup
from mora3.model import segmentation_inference as bridge


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('origin',['source_observed','model_generated','model_predicted'])
def test_image_only_input_matches_raster_without_fabricated_masks(tmp_path,device,origin):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('no CUDA')
    t,n,r=setup(tmp_path,device)
    a=replace(r.artifacts[0],evidence=origin,checkpoint=None if origin=='source_observed' else 'generator-v1')
    r=replace(r,artifacts=(a,))
    image=bridge.SegmentationImageInput(a,t.image_dtype,bytes.fromhex(t.image_hex))
    assert not hasattr(image,'labels_hex')
    with torch.inference_mode(): expected=n.model(t)
    with torch.inference_mode(): assert torch.equal(n.model(image),expected)
    output=n.infer(r,{'image':image},prediction_assets={'image':'pred.pt'},expected_pixel_sha256={'image':t.image_sha256})['image']
    assert output['status']=='predicted'
    assert not output['measurement']['biological_validation']
    assert output['measurement']['artifact'].evidence=='model_predicted'
    assert output['measurement']['artifact'].kind=='objects'
    with pytest.raises(ValueError,match='inference'): n.model(image,for_training=True)


@pytest.mark.parametrize('change',['pixels','artifact','dtype','size'])
def test_neutral_binding_rejects_changes(tmp_path,change):
    t,n,r=setup(tmp_path)
    a=r.artifacts[0]; raw=bytes.fromhex(t.image_hex); dtype=t.image_dtype
    if change=='pixels': raw=bytes(len(raw))
    if change=='artifact': a=replace(a,parent_id='wrong')
    if change=='dtype': dtype='<f2'
    if change=='size': raw=raw[:-1]
    with pytest.raises(ValueError):
        image=bridge.SegmentationImageInput(a,dtype,raw)
        n.infer(r,{'image':image},prediction_assets={'image':'p.pt'},expected_pixel_sha256={'image':t.image_sha256})


def test_hidden_neutral_image_never_resolved(tmp_path):
    t,n,r=setup(tmp_path)
    r=replace(r,artifacts=(replace(r.artifacts[0],role='target'),))
    assert n.infer(r,{},prediction_assets={},expected_pixel_sha256={})['image']['status']=='unavailable:held_out_target'
