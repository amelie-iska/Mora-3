"""Pixel geometry oracle tests; fixtures are not training coverage."""
from dataclasses import replace
import pytest
import torch
from mora3.data.cooperative_segmentation import SegmentationArtifact
from mora3.data.segmentation_measurements import measure_predicted_instances


def image(**kw):
    d=dict(artifact_id='image',kind='image',evidence='source_observed',role='condition',
           source_id='s:v1',parent_id='p',context=(('expression',None),),asset='immutable.tif',
           representation='yx',channels=('brightfield',),spatial_shape=(2,3))
    d.update(kw)
    return SegmentationArtifact(**d)


def test_exact_pixel_geometry_adjacency_and_missing_scale():
    r=measure_predicted_instances(torch.tensor([[1,1,2],[0,1,2]]),image(),checkpoint='native:sha',mask_asset='pred.pt')
    a,b=r['objects']
    assert a==dict(object_id=1,area_px=3,centroid_yx=[1/3,2/3],bbox_yxyx=[0,0,2,2],perimeter_px=8,area_um2=None)
    assert b['area_px']==2 and b['perimeter_px']==6
    assert r['edges']==[dict(source=1,target=2,relation='pixel_boundary_adjacency',shared_boundary_px=2)]
    assert r['artifact'].evidence=='model_predicted'
    assert r['artifact'].checkpoint=='native:sha'
    assert r['artifact'].links==( ('objects_of','image'), )
    assert r['biological_validation'] is False


def test_scale_empty_and_arbitrary_labels():
    r=measure_predicted_instances(torch.tensor([[0,500,500],[0,0,0]]),image(pixel_size_um=.5),checkpoint='c',mask_asset='m')
    assert r['objects'][0]['object_id']==500
    assert r['objects'][0]['area_um2']==.5
    assert r['edges']==[]
    assert measure_predicted_instances(torch.zeros((2,3),dtype=torch.long),image(),checkpoint='c',mask_asset='m')['objects']==[]


@pytest.mark.parametrize('role',['target','evaluation','unused'])
def test_excludes_hidden_images(role):
    with pytest.raises(ValueError,match='condition'):
        measure_predicted_instances(torch.ones((2,3),dtype=torch.long),image(role=role),checkpoint='c',mask_asset='m')


@pytest.mark.parametrize('labels',[torch.ones(2,3),torch.full((2,3),-1),torch.ones((3,2),dtype=torch.long)])
def test_rejects_invalid_geometry(labels):
    with pytest.raises(ValueError):
        measure_predicted_instances(labels,image(),checkpoint='c',mask_asset='m')


def test_generated_origin_preserved_and_no_mutation():
    x=image(evidence='model_generated',checkpoint='generator:v1')
    labels=torch.tensor([[1,1,0],[0,0,0]])
    old=labels.clone()
    r=measure_predicted_instances(labels,x,checkpoint='segmenter:v2',mask_asset='m')
    assert torch.equal(labels,old)
    assert r['image_evidence']=='model_generated'
    assert r['artifact'].quality_flags==('model_consistency_only',)
    assert r['artifact'].context==x.context


def test_cpu_cuda_geometry_equal():
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    x=torch.tensor([[1,1,2],[0,1,2]])
    a=measure_predicted_instances(x,image(),checkpoint='c',mask_asset='m')
    b=measure_predicted_instances(x.cuda(),image(),checkpoint='c',mask_asset='m')
    assert a==b
