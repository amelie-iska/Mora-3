from dataclasses import replace
import hashlib
import numpy as np
import pytest
from test_color_semantic_inference import fixture


def setup_case(scale=None):
    from mora3.model.color_semantic_assessment import SemanticAreaConstraint
    from mora3.model.color_semantic_inference import image_artifact_digest
    image,_=fixture();image=replace(image,pixel_size_um=scale)
    c=SemanticAreaConstraint(image_artifact_digest(image),'request','hypothetical', 'px2',2,2)
    mask=np.zeros(image.spatial_shape,dtype=bool);mask[0,:2]=True
    artifact=replace(image,artifact_id='prediction',kind='mask',evidence='predicted_mask',role='evaluation',
        channels=(),asset='prediction.npy',links=(('mask_of',image.artifact_id),),
        representation='nuclear-semantic-logit-ge0-v1',checkpoint='native-state-sha256:'+'a'*64,
        uncertainty='uncalibrated',quality_flags=('quality_unevaluated','semantic_not_instance'))
    return image,c,mask,artifact


def test_area_bounds_and_scale():
    from mora3.model.color_semantic_assessment import assess_semantic_area
    image,c,mask,a=setup_case(.5)
    for unit,value in [('px2',2),('um2',.5)]:
        result=assess_semantic_area(mask,a,image,replace(c,unit=unit,lower=value,upper=value),quality_accepted=True)
        assert result['value']==value and result['satisfied'] is True
        assert result['biological_validation'] is False and result['instance_count'] is None
        assert result['mask_sha256']==hashlib.sha256(mask.tobytes()).hexdigest()
    assert assess_semantic_area(mask,a,image,replace(c,lower=3,upper=4),quality_accepted=True)['satisfied'] is False

@pytest.mark.parametrize('quality,status',[(None,'quality_unassessed'),(False,'quality_not_accepted')])
def test_quality_not_silently_accepted(quality,status):
    from mora3.model.color_semantic_assessment import assess_semantic_area
    image,c,mask,a=setup_case()
    r=assess_semantic_area(mask,a,image,c,quality_accepted=quality)
    assert r['status']=='unevaluated:'+status and r['satisfied'] is None and r['value'] is None


def test_missing_scale_and_empty_foreground():
    from mora3.model.color_semantic_assessment import assess_semantic_area
    image,c,mask,a=setup_case()
    assert assess_semantic_area(mask,a,image,replace(c,unit='um2'),quality_accepted=True)['status']=='unevaluated:missing_scale'
    mask[:]=False
    assert assess_semantic_area(mask,a,image,replace(c,lower=0,upper=0),quality_accepted=True)['satisfied'] is True

@pytest.mark.parametrize('change',[{'unit':'count'},{'lower':True},{'lower':float('nan')},{'lower':3},{'lower':None,'upper':None},{'origin':''},{'image_sha256':'x'}])
def test_invalid_request(change):
    image,c,mask,a=setup_case()
    with pytest.raises(ValueError):replace(c,**change)

@pytest.mark.parametrize('change',[{'parent_id':'other'},{'links':(('mask_of','other'),)}, {'checkpoint':'other'}, {'representation':'instance-labels'}, {'role':'target'},{'context':(('object_type','cell'),)}])
def test_prediction_binding_rejected(change):
    from mora3.model.color_semantic_assessment import assess_semantic_area
    image,c,mask,a=setup_case()
    with pytest.raises(ValueError):assess_semantic_area(mask,replace(a,**change),image,c,quality_accepted=True)


def test_mask_and_image_binding():
    from mora3.model.color_semantic_assessment import assess_semantic_area
    image,c,mask,a=setup_case()
    for bad in [mask.astype('uint8'),mask[:1],np.ones((7,9,1),dtype=bool)]:
        with pytest.raises(ValueError):assess_semantic_area(bad,a,image,c,quality_accepted=True)
    with pytest.raises(ValueError):assess_semantic_area(mask,a,replace(image,asset='different'),c,quality_accepted=True)
    with pytest.raises(ValueError):assess_semantic_area(mask,a,image,c,quality_accepted=1)
