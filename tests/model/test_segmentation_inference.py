from dataclasses import replace
import hashlib
import pytest
import torch
from mora3.data.cooperative_segmentation import SegmentationArtifact, SegmentationRequest
from mora3.model import segmentation_inference as bridge
from mora3.model.morphology import MorphologyInstance
from mora3.data.raster_morphology import RasterMorphologyTask
import numpy as np


def task(tmp_path):
    pixels=np.arange(12,dtype=np.uint16).tobytes()
    return RasterMorphologyTask(3,4,'<u2','<u2',pixels.hex(),'0000'*12,
        'fixture','v1','field.tif','a'*64,hashlib.sha256(pixels).hexdigest(),
        'b'*64,'CC-BY-4.0','parent','train','bacterium','strain','brightfield','fixture',0.08)


def setup(tmp_path, device='cpu'):
    t=task(tmp_path)
    m=MorphologyInstance(8,normalization='image-minmax')
    p=tmp_path/'native.pt'
    torch.save({'model':m.state_dict(),'config':{'variant':'instance','width':8,'normalization':'image-minmax'}},p)
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    native=bridge.NativeRasterSegmentation(p,expected_sha256=digest,device=device)
    a=SegmentationArtifact('image','image','source_observed','condition',t.source+':'+t.revision,
        t.parent_group,(('species',t.species),('strain',t.strain),('perturbation',None),('expression',None)),
        'archive#'+t.source_id,'source-yx-no-transform-v1',channels=('brightfield',),
        spatial_shape=(t.height,t.width),pixel_size_um=t.pixel_size_um)
    r=SegmentationRequest((a,),(('brightfield',),),('whole_cell',),'whole_cell',native.identity)
    return t,native,r


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_native_request_matches_direct_head_and_excludes_targets(tmp_path,device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('no CUDA')
    t,n,r=setup(tmp_path,device)
    result=n.infer(r,{'image':t},prediction_assets={'image':'immutable/prediction.pt'},expected_pixel_sha256={'image':t.image_sha256})
    from mora3.model.instance_decoding import decode_instances
    with torch.inference_mode(): expected=decode_instances(n.model(t))
    assert torch.equal(result['image']['labels'],expected)
    assert result['image']['measurement']['artifact'].checkpoint==n.identity
    changed=replace(t,labels_hex='0000'*12,mask_sha256='a'*64)
    assert torch.equal(result['image']['labels'],n.infer(r,{'image':changed},prediction_assets={'image':'other.pt'},expected_pixel_sha256={'image':t.image_sha256})['image']['labels'])
    assert not result['image']['measurement']['biological_validation']
    assert all(p.grad is None for p in n.model.parameters())


@pytest.mark.parametrize('change',['checkpoint','parent','context','pixels','direction','shape','channels'])
def test_incompatible_binding_rejected(tmp_path,change):
    t,n,r=setup(tmp_path)
    if change=='checkpoint': r=replace(r,head_checkpoint='wrong')
    if change=='parent': t=replace(t,parent_group='different')
    if change=='context': t=replace(t,strain='different')
    if change=='pixels': t=replace(t,image_hex='0000'*12)
    if change=='direction': t=replace(t,direction='mask_to_image')
    if change=='shape': r=replace(r,artifacts=(replace(r.artifacts[0],spatial_shape=(4,3)),))
    if change=='channels': r=replace(r,artifacts=(replace(r.artifacts[0],channels=('unknown',)),),supported_channels=(('unknown',),))
    with pytest.raises(ValueError): n.infer(r,{'image':t},prediction_assets={'image':'p.pt'},expected_pixel_sha256={'image':t.image_sha256})


def test_hidden_image_not_resolved_and_missing_head_explicit(tmp_path):
    t,n,r=setup(tmp_path)
    hidden=replace(r,artifacts=(replace(r.artifacts[0],role='target'),))
    assert n.infer(hidden,{},prediction_assets={},expected_pixel_sha256={})=={'image':{'status':'unavailable:held_out_target'}}
    assert n.infer(replace(r,head_checkpoint=None),{},prediction_assets={},expected_pixel_sha256={})=={'image':{'status':'unavailable:missing_native_head'}}


def test_checkpoint_hash_rejected_before_load(tmp_path):
    t,n,r=setup(tmp_path)
    with pytest.raises(ValueError,match='hash'): bridge.NativeRasterSegmentation(tmp_path/'native.pt',expected_sha256='0'*64)


def test_executed_constraint_bridge_matches_direct_assessment(tmp_path):
    from mora3.data.segmentation_constraints import GeometryConstraint, assess_geometry
    t,n,r=setup(tmp_path)
    x=r.artifacts[0]
    d=replace(x,artifact_id='desired',kind='objects',evidence='desired',role='desired',
              channels=(),links=(('objects_of',x.artifact_id),),asset='desired-request')
    k=GeometryConstraint(d,'whole_cell','total_area','px2',0,12)
    args=dict(prediction_assets={'image':'p.pt'},expected_pixel_sha256={'image':t.image_sha256})
    direct=n.infer(r,{'image':t},**args)['image']
    actual=n.infer_and_assess(r,{'image':t},constraints={'image':(k,)},**args)['image']
    expected=assess_geometry(direct['labels'],x,k,object_type='whole_cell',checkpoint=n.identity,mask_asset='p.pt')
    assert actual['assessments']==(expected,)
    assert torch.equal(actual['labels'],direct['labels'])
    changed=replace(t,labels_hex='ffff'*12,mask_sha256='c'*64)
    assert n.infer_and_assess(r,{'image':changed},constraints={'image':(k,)},**args)['image']['assessments']==(expected,)
    rejected=n.infer_and_assess(r,{'image':t},constraints={'image':(k,)},quality_accepted=False,**args)
    assert rejected['image']['assessments'][0]['status']=='unevaluated:quality_not_accepted'
    with pytest.raises(ValueError,match='binding'):
        n.infer_and_assess(r,{'image':t},constraints={'image':(replace(k,desired=replace(d,parent_id='wrong')), )},**args)
    with pytest.raises(ValueError,match='unknown'):
        n.infer_and_assess(r,{},constraints={'absent':(k,)},**args)


@pytest.mark.parametrize('route',['held_out_target','missing_native_head','unsupported_channels'])
def test_unavailable_constraint_routes_never_pass_or_load_pixels(tmp_path,route):
    from mora3.data.segmentation_constraints import GeometryConstraint
    t,n,r=setup(tmp_path); x=r.artifacts[0]
    d=replace(x,artifact_id='desired',kind='objects',evidence='desired',role='desired',channels=(),links=(('objects_of','image'),),asset='request')
    k=GeometryConstraint(d,'whole_cell','object_count','count',0,None)
    if route=='held_out_target': r=replace(r,artifacts=(replace(x,role='target'),))
    elif route=='missing_native_head': r=replace(r,head_checkpoint=None)
    else: r=replace(r,supported_channels=(('other',),))
    result=n.infer_and_assess(r,{},constraints={'image':(k,)},prediction_assets={},expected_pixel_sha256={})['image']
    assert result['status']=='unavailable:'+route
    assert result['assessments']==({'status':'unevaluated:'+route,'satisfied':None,'biological_validation':False},)
