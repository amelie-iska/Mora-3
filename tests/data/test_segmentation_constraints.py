"""Discrete constraint oracles, never evidence of biological efficacy."""
from dataclasses import replace
import pytest
import torch
from mora3.data.cooperative_segmentation import SegmentationArtifact
from mora3.data import segmentation_constraints as c


def image(**kw):
    d=dict(artifact_id='i',kind='image',evidence='source_observed',role='condition',
           source_id='s',parent_id='p',context=(('assay','a'),),asset='image',
           representation='yx',channels=('brightfield',),spatial_shape=(2,3))
    d.update(kw)
    return SegmentationArtifact(**d)


def desired(x):
    return replace(x,artifact_id='d',kind='objects',evidence='desired',role='desired',
                   channels=(),links=(('objects_of',x.artifact_id),),asset='request')


def assess(x=None,metric='object_count',unit='count',lower=2,upper=2,device='cpu',**kw):
    x=x or image()
    constraint=c.GeometryConstraint(desired(x), 'whole_cell',metric,unit,lower,upper)
    return c.assess_geometry(torch.tensor([[1,1,0],[0,2,2]],device=device),x,
            constraint,object_type='whole_cell',checkpoint='segmenter:v1',mask_asset='mask',**kw)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_exact_inclusive_count_area_reference_and_provenance(device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    r=assess(device=device)
    assert (r['value'],r['satisfied'],r['status'])==(2,True,'evaluated')
    assert r['biological_validation'] is False and r['differentiable'] is False
    assert r['measurement']['artifact'].checkpoint=='segmenter:v1'
    assert assess(metric='total_area',unit='px2',lower=4,upper=4)['value']==4
    assert assess(metric='mean_area',unit='px2',lower=2,upper=2)['satisfied']
    assert not assess(lower=3,upper=None)['satisfied']
    assert assess(x=image(pixel_size_um=.5),metric='total_area',unit='um2',lower=1,upper=1)['value']==1


def test_missing_scale_empty_mean_quality_and_generated_consistency():
    assert assess(metric='total_area',unit='um2')['status']=='unevaluated:missing_scale'
    assert assess(quality_accepted=False)['status']=='unevaluated:quality_not_accepted'
    x=image(evidence='model_generated',checkpoint='generator:v1')
    assert assess(x=x)['evidence_scope']=='model_consistency_only'
    k=c.GeometryConstraint(desired(x),'whole_cell','mean_area','px2',0,None)
    r=c.assess_geometry(torch.zeros(2,3,dtype=torch.long),x,k,object_type='whole_cell',checkpoint='s',mask_asset='m')
    assert r['status']=='unevaluated:no_objects' and r['satisfied'] is None


@pytest.mark.parametrize('field,value',[('metric','volume'),('unit','mm2'),('lower',float('nan')),('lower',True),('upper',-1),('object_type','')])
def test_invalid_constraints(field,value):
    d=dict(desired=desired(image()),object_type='whole_cell',metric='object_count',unit='count',lower=0,upper=2)
    d[field]=value
    with pytest.raises(ValueError): c.GeometryConstraint(**d)


@pytest.mark.parametrize('field,value',[('parent_id','other'),('context',(('assay','other'),)),('reference_frame','other'),('pixel_size_um',.5),('spatial_shape',(3,2)),('links',(('objects_of','other'),))])
def test_exact_context_and_frame_binding(field,value):
    x=image(); k=c.GeometryConstraint(replace(desired(x),**{field:value}),'whole_cell','object_count','count',0,None)
    with pytest.raises(ValueError,match='binding'):
        c.assess_geometry(torch.zeros(2,3,dtype=torch.long),x,k,object_type='whole_cell',checkpoint='s',mask_asset='m')


def test_object_type_target_exclusion_and_no_mutation():
    x=image(); k=c.GeometryConstraint(desired(x),'nucleus','object_count','count',0,None)
    with pytest.raises(ValueError,match='object type'):
        c.assess_geometry(torch.zeros(2,3,dtype=torch.long),x,k,object_type='whole_cell',checkpoint='s',mask_asset='m')
    with pytest.raises(ValueError): assess(x=image(role='target'))
    with pytest.raises(ValueError): c.GeometryConstraint(replace(desired(x),evidence='model_predicted',role='condition',checkpoint='s'),'whole_cell','object_count','count',0,None)
    a=torch.tensor([[1,1,0],[0,2,2]]); old=a.clone()
    k=replace(k,object_type='whole_cell')
    c.assess_geometry(a,x,k,object_type='whole_cell',checkpoint='s',mask_asset='m')
    assert torch.equal(a,old)
