import copy
from dataclasses import replace
import pytest
import torch
from test_slot_shared_graph import case
from test_bound_instance_inference import setup_case
from mora3.model.color_instance_inference import infer_nuclear_instances, instance_model_digest
from mora3.data import slot_geometry_constraints as c


def constraint(image,metric='object_count',unit='count',lower=2,upper=2):
    desired=replace(image,artifact_id='desired',kind='objects',evidence='desired',
        role='desired',channels=(),links=(('objects_of',image.artifact_id),),asset='desired.json')
    return c.SlotGeometryConstraint(desired,'nucleus',metric,unit,lower,upper)


def assess(image,r,k,**kw):
    return c.assess_bound_slot_geometry(image,r,k,max_mask_bytes=189,**kw)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_analytic_duplicate_plane_count_area_pairwise_overlap(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('no CUDA')
    # The bound result is device-independent; inference fixture has two identical
    # full planes and one empty plane: area sum=126, pair sum=63, count=2.
    m,image,pixels,kw=setup_case(device)
    with torch.no_grad():
        m.color.head.projection.weight.zero_();m.color.head.projection.bias.zero_()
        m.offset.weight.zero_();m.offset.bias.zero_();m.color.head.projection.bias[2]=-1
    kw['expected_model_sha256']=instance_model_digest(m)
    r=infer_nuclear_instances(m,image,pixels,**kw)
    for metric,unit,value in [('object_count','count',2),('total_area','px2',126),
            ('mean_area','px2',63),('pairwise_overlap_area','px2',63)]:
        k=constraint(image,metric,unit,value,value)
        out=assess(image,r,k,quality_accepted=True)
        assert (out['value'],out['satisfied'],out['status'])==(value,True,'evaluated')
        assert out['measurement']['empty_slot_ids']==[3]
        assert out['binding']==r['binding'] and out['mask_sha256']==r['mask_sha256']
        assert out['quality_accepted'] is True and not out['biological_validation']
        assert not out['training_admission'] and not out['differentiable']
        assert out['evidence_scope']=='predicted_segmentation_only'
    assert assess(image,r,constraint(image,lower=3,upper=None),quality_accepted=True)['satisfied'] is False


def test_quality_unknown_rejected_and_missing_scale_never_pass():
    image,r,_=case()
    for quality,status in [(None,'quality_unevaluated'),(False,'quality_not_accepted')]:
        out=assess(image,r,constraint(image),quality_accepted=quality)
        assert out['status']=='unevaluated:'+status and out['satisfied'] is None and out['value'] is None
    out=assess(image,r,constraint(image,'total_area','um2',0,None),quality_accepted=True)
    assert out['status']=='unevaluated:missing_scale' and out['satisfied'] is None


@pytest.mark.parametrize('change',['mask','area','checkpoint','binding','image','desired','budget','quality'])
def test_binding_and_policy_refusal_even_without_quality(change):
    image,r,_=case();r=copy.deepcopy(r);k=constraint(image);kw={}
    if change=='mask':r['masks'][0,0,0]=False
    if change=='area':r['measurements']['objects'][0]['area_px']=1
    if change=='checkpoint':r['measurements']['artifact']=replace(r['measurements']['artifact'],checkpoint='wrong')
    if change=='binding':r['binding']['task_sha256']='0'*64
    if change=='image':image=replace(image,parent_id='other')
    if change=='desired':k=replace(k,desired=replace(k.desired,context=(('assay','other'),)))
    if change=='budget':kw['max_mask_bytes']=188
    if change=='quality':kw['quality_accepted']=1
    with pytest.raises(ValueError):c.assess_bound_slot_geometry(image,r,k,max_mask_bytes=kw.pop('max_mask_bytes',189),**kw)


@pytest.mark.parametrize('field,value',[('metric','union_area'),('unit','mm2'),('lower',True),('lower',float('nan')),('upper',-1),('object_type','whole_cell')])
def test_explicit_nuclear_slot_semantics_and_bounds(field,value):
    image,_,_=case();k=constraint(image)
    with pytest.raises(ValueError):replace(k,**{field:value})


def test_result_isolation_and_inclusive_bounds():
    image,r,_=case();k=constraint(image)
    out=assess(image,r,k,quality_accepted=True)
    out['binding']['pixel_sha256']='wrong'
    out['measurement']['objects'][0]['area_px']=0
    assert r['binding']['pixel_sha256']!='wrong' and r['measurements']['objects'][0]['area_px']==63


@pytest.mark.parametrize('empty',[False,True])
def test_physical_scale_triple_overlap_and_empty_mean(empty):
    from mora3.model.color_semantic_inference import image_artifact_digest
    m,image,pixels,kw=setup_case()
    image=replace(image,pixel_size_um=.5)
    kw['expected_image_sha256']=image_artifact_digest(image)
    with torch.no_grad():
        m.color.head.projection.weight.zero_();m.color.head.projection.bias.fill_(-1 if empty else 1)
        m.offset.weight.zero_();m.offset.bias.zero_()
    kw['expected_model_sha256']=instance_model_digest(m)
    r=infer_nuclear_instances(m,image,pixels,**kw)
    # Three identical planes: sum areas = sum unordered pair overlaps = 189px2.
    # This deliberately differs from union overlap, which is only 63px2.
    for metric in ('total_area','pairwise_overlap_area'):
        value=0 if empty else 189*.25
        out=assess(image,r,constraint(image,metric,'um2',value,value),quality_accepted=True)
        assert out['satisfied'] is True and out['value']==value
    k=constraint(image,'mean_area','um2',0,None)
    out=assess(image,r,k,quality_accepted=True)
    if empty:
        assert out['status']=='unevaluated:no_objects' and out['satisfied'] is None
        assert assess(image,r,constraint(image,lower=0,upper=0),quality_accepted=True)['satisfied']
    else:assert out['value']==63*.25
