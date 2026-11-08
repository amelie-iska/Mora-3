import copy
import hashlib
from dataclasses import replace
import numpy as np
import pytest
import torch
from test_color_selected_context import sample
from mora3.model import color_semantic_assessment as api
from mora3.model.color_semantic_inference import model_digest,image_artifact_digest
from mora3.model.color_semantic import ColorNuclearSemantic


def case(device='cpu'):
    torch.manual_seed(18701)
    model=ColorNuclearSemantic(4,alpha_policy='require-opaque').to(device=device,dtype=torch.float64)
    image,pixels=sample()
    constraint=api.SemanticAreaConstraint(image_artifact_digest(image),'area-request','hypothetical','px2',0,63)
    kw=dict(expected_pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),expected_model_sha256=model_digest(model),mask_asset='new.npy',seed=18701)
    return model,image,pixels,constraint,kw


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('quality',[True,False,None])
def test_executed_assessment_oracle_and_bindings(device,quality):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('no CUDA')
    m,image,pixels,c,kw=case(device);rng=torch.get_rng_state().clone();modes=[x.training for x in m.modules()]
    out=api.infer_and_assess_semantic_area(m,image,pixels,c,quality_accepted=quality,**kw)
    a=out['area_assessment'];mask=out['mask']
    with torch.inference_mode():oracle=(copy.deepcopy(m).eval()(pixels)>=0).cpu().numpy()
    assert np.array_equal(mask,oracle)
    assert a['mask_sha256']==hashlib.sha256(mask.tobytes()).hexdigest()
    assert a['execution']['pixel_sha256']==kw['expected_pixel_sha256']
    assert a['execution']['model_sha256']==kw['expected_model_sha256']
    assert a['execution']['seed']==18701
    assert a['value']==(sum(int(v) for v in oracle.flat) if quality else None)
    assert a['satisfied'] is (True if quality else None)
    assert a['training_admission'] is False and a['biological_validation'] is False
    assert modes==[x.training for x in m.modules()]
    assert model_digest(m)==kw['expected_model_sha256'] and torch.equal(rng,torch.get_rng_state())


@pytest.mark.parametrize('bad',['constraint','quality','pixels','state'])
def test_bad_bindings_fail_before_model_execution(bad):
    m,image,pixels,c,kw=case();quality=True
    if bad=='constraint':c=replace(c,image_sha256='a'*64)
    if bad=='quality':quality=1
    if bad=='pixels':kw['expected_pixel_sha256']='a'*64
    if bad=='state':kw['expected_model_sha256']='a'*64
    def forbidden(*args,**kwargs):raise AssertionError('model executed before binding validation')
    m.forward=forbidden
    with pytest.raises(ValueError):api.infer_and_assess_semantic_area(m,image,pixels,c,quality_accepted=quality,**kw)


def test_assessment_bounds_do_not_condition_or_change_prediction():
    from mora3.model.color_conditioning import ConditionedColorSemantic
    from mora3.model.color_semantic_inference import desired_color_context_request
    m,image,pixels,c,kw=case()
    m=ConditionedColorSemantic(m,id_width=1).double();kw['expected_model_sha256']=model_digest(m)
    desired=desired_color_context_request(image,expected_image_sha256=c.image_sha256,request_id='context',origin='test',values=(('stain','hypothetical'),))
    kw.update(context_fields=('tissue',),expected_image_sha256=c.image_sha256,desired_context=desired)
    a=api.infer_and_assess_semantic_area(m,image,pixels,c,quality_accepted=True,**kw)
    b=api.infer_and_assess_semantic_area(m,image,pixels,replace(c,request_id='other',lower=10000,upper=None),quality_accepted=True,**kw)
    assert np.array_equal(a['mask'],b['mask']) and a['conditioning']==b['conditioning']
    assert a['area_assessment']['satisfied'] is True and b['area_assessment']['satisfied'] is False
    assert a['area_assessment']['request_sha256']!=b['area_assessment']['request_sha256']
    assert a['area_assessment']['execution']['conditioning']['desired_request_sha256']==hashlib.sha256(desired.canonical_bytes()).hexdigest()
