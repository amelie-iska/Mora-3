import copy
from dataclasses import replace
import pytest
import torch
from test_segmentation_inference import setup
from mora3.data.segmentation_image import SegmentationImageInput
from mora3.model.cooperative_training import CooperativeModel
from mora3.model.geometry_pixel_fusion import GeometryPixelFusion

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_fusion_equation_and_gradient_ownership(tmp_path,device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.manual_seed(73)
    t,n,r=setup(tmp_path,device)
    image=SegmentationImageInput(r.artifacts[0],t.image_dtype,bytes.fromhex(t.image_hex))
    m=CooperativeModel(('A','B'),width=8).to(device)
    f=GeometryPixelFusion(m.image,id_width=16).to(device)
    y,audit=f(image,r,n,prediction_asset='prediction.pt')
    assert y.shape==(3,3,4)
    g,task=audit['input_graph'],audit['task']
    context=f.geometry(g,task)
    expected=m.image._logits(image,external_context=context)
    torch.testing.assert_close(y,expected,atol=0,rtol=0)
    base=m.image(image)
    zero=m.image._logits(image,external_context=torch.zeros_like(context))
    assert torch.equal(base,zero)
    # Exact additive context equation through the existing 1x1 pixel head.
    delta=(m.image.head.weight[:,8:,0,0] @ context[0])[:,None,None]
    torch.testing.assert_close(y-base,delta.expand_as(y),atol=2e-6,rtol=2e-5)
    assert not torch.equal(y,m.image(image))
    y.square().mean().backward()
    for p in [m.image.head.weight,m.image.pixels.enc0[0].weight,
              m.response.backbone.layers[0].attention.qkv.weight,
              f.geometry.graph.incidence.weight,f.geometry.payload.embedding.weight]:
        assert p.grad is not None and p.grad.abs().sum()>0
    assert m.response.response_head.weight.grad is None
    assert all(p.grad is None for p in n.model.parameters())
    assert audit['task'].available_targets(g)==()
    assert not audit['biological_validation']
    hidden=replace(r.artifacts[0],artifact_id='hidden',role='target')
    z,_=f(image,replace(r,artifacts=r.artifacts+(hidden,)),n,prediction_asset='prediction.pt')
    assert torch.equal(y,z)
    # No training admission via inference transport.
    with pytest.raises(ValueError): image.require_training()

@pytest.mark.parametrize('change',['artifact','checkpoint','second_image','head'])
def test_mismatched_or_ambiguous_inputs_rejected(tmp_path,change):
    t,n,r=setup(tmp_path)
    image=SegmentationImageInput(r.artifacts[0],t.image_dtype,bytes.fromhex(t.image_hex))
    m=CooperativeModel(('A','B'),width=8)
    f=GeometryPixelFusion(m.image,id_width=16)
    if change=='artifact': image=replace(image,artifact=replace(image.artifact,parent_id='other'))
    if change=='checkpoint': r=replace(r,head_checkpoint='wrong')
    if change=='second_image': r=replace(r,artifacts=r.artifacts+(replace(r.artifacts[0],artifact_id='other'),))
    if change=='head': r=replace(r,head_checkpoint=None)
    with pytest.raises(ValueError): f(image,r,n,prediction_asset='prediction.pt')


def test_checkpoint_reconstruction_and_context_shape(tmp_path):
    t,n,r=setup(tmp_path)
    image=SegmentationImageInput(r.artifacts[0],t.image_dtype,bytes.fromhex(t.image_hex))
    m=CooperativeModel(('A','B'),width=8)
    f=GeometryPixelFusion(m.image,id_width=16)
    y,_=f(image,r,n,prediction_asset='prediction.pt')
    state=copy.deepcopy(f.state_dict())
    b=GeometryPixelFusion(CooperativeModel(('A','B'),width=8).image,id_width=16)
    b.load_state_dict(state)
    assert b.geometry.graph.layers is b.image.backbone.layers
    assert torch.equal(y,b(image,r,n,prediction_asset='prediction.pt')[0])
    with pytest.raises(ValueError):
        m.image._logits(image,external_context=torch.zeros(2,8))
    with pytest.raises(ValueError):
        GeometryPixelFusion(CooperativeModel(('A','B'),width=8).image,id_width=17).load_state_dict(state)
