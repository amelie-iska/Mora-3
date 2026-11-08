"""Source wrapper honors exact model extent without caller policy overrides."""
import copy
import hashlib
from dataclasses import replace
import numpy as np
import pytest
import torch
from test_color_semantic_inference import fixture
from mora3.model.color_semantic import ColorNuclearSemantic
from mora3.model.color_conditioning import ConditionedColorSemantic
from mora3.model.color_semantic_inference import infer_nuclear_semantic, model_digest, image_condition_graph

@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('conditioned',[False,True])
def test_bound_native_extent_matches_direct_equation(device,conditioned):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(21901)
    pixels=np.random.default_rng(21901).integers(0,256,(600,3,3),dtype=np.uint8)
    image,_=fixture();image=replace(image,spatial_shape=(600,3))
    base=ColorNuclearSemantic(4,alpha_policy='require-opaque',input_policy='native-max1000-v1')
    model=(ConditionedColorSemantic(base,id_width=1) if conditioned else base).to(device=device,dtype=torch.float64)
    model.train();before=model_digest(model);modes=[m.training for m in model.modules()]
    sha=hashlib.sha256(pixels.tobytes()).hexdigest()
    out=infer_nuclear_semantic(model,image,pixels,expected_pixel_sha256=sha,expected_model_sha256=before,mask_asset='new.npy',seed=21901)
    ref=copy.deepcopy(model).eval()
    with torch.inference_mode():
        if conditioned:
            graph,task=image_condition_graph(image,pixel_sha256=sha)
            logits=ref(pixels,graph,task)
        else:logits=ref(pixels)
    assert np.array_equal(out['mask'],(logits>=0).cpu().numpy())
    assert out['mask'].shape==(600,3) and not out['mask'].flags.writeable
    assert out['artifact'].parent_id==image.parent_id
    assert out['instance_count'] is None and out['quality_accepted'] is None
    assert model_digest(model)==before and modes==[m.training for m in model.modules()]
    old=ColorNuclearSemantic(4,alpha_policy='require-opaque')
    if conditioned:old=ConditionedColorSemantic(old,id_width=1)
    with pytest.raises(ValueError):
        infer_nuclear_semantic(old,image,pixels,expected_pixel_sha256=sha,expected_model_sha256=model_digest(old),mask_asset='new.npy',seed=21901)

@pytest.mark.parametrize('case',['overbound','alpha','pixel','model'])
def test_native_extent_retains_refusals(case):
    pixels=np.zeros((600,2,4 if case=='alpha' else 3),dtype=np.uint8)
    if case=='overbound':pixels=np.zeros((1001,2,3),dtype=np.uint8)
    image,_=fixture();image=replace(image,spatial_shape=pixels.shape[:2],channels=('red','green','blue','alpha') if case=='alpha' else ('red','green','blue'))
    model=ColorNuclearSemantic(4,alpha_policy='require-opaque',input_policy='native-max1000-v1')
    kw=dict(expected_pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),expected_model_sha256=model_digest(model),mask_asset='new.npy',seed=21901)
    if case in ('pixel','model'):kw['expected_'+case+'_sha256']='0'*64
    with pytest.raises(ValueError):infer_nuclear_semantic(model,image,pixels,**kw)
