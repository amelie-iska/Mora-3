import copy
import hashlib
from dataclasses import replace
import numpy as np
import pytest
import torch
from test_color_semantic_inference import fixture
from mora3.model.color_conditioning import ConditionedColorInstance
from mora3.model.color_semantic import ColorInstancePredictor


def setup_case(device='cpu'):
    from mora3.model.color_instance_inference import instance_model_digest
    torch.manual_seed(23801)
    m=ConditionedColorInstance(ColorInstancePredictor(4,slots=3,max_output_bytes=4096,
        alpha_policy='require-opaque'),id_width=1).to(device=device,dtype=torch.float64)
    image,pixels=fixture()
    return m,image,pixels,dict(expected_pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
        expected_model_sha256=instance_model_digest(m),
        expected_image_sha256=__import__('mora3.model.color_semantic_inference',fromlist=['image_artifact_digest']).image_artifact_digest(image),
        mask_asset='new-planes.npy',seed=23801,max_mask_bytes=189)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_bound_planes_geometry_and_caller_isolation(device):
    from mora3.model.color_instance_inference import infer_nuclear_instances,instance_model_digest
    from mora3.model.color_semantic_inference import image_condition_graph
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('no CUDA')
    m,image,pixels,kw=setup_case(device)
    with torch.no_grad():
        m.color.head.projection.weight.zero_();m.color.head.projection.bias.zero_()
        m.offset.weight.zero_();m.offset.bias.zero_()
        m.color.head.projection.bias[2]=-1
    kw['expected_model_sha256']=instance_model_digest(m)
    m.train();m.color.pixels.eval()
    for p in m.parameters():p.grad=torch.ones_like(p)
    state=copy.deepcopy(m.state_dict());grads=[p.grad.clone() for p in m.parameters()]
    modes=[x.training for x in m.modules()];rng=torch.get_rng_state().clone()
    out=infer_nuclear_instances(m,image,pixels,**kw)
    g,t=image_condition_graph(image,pixel_sha256=kw['expected_pixel_sha256'])
    with torch.inference_mode():ref=copy.deepcopy(m).eval()(pixels,g,t)>=0
    assert np.array_equal(out['masks'],ref.cpu().numpy())
    assert out['masks'][:2].all() and not out['masks'][2].any()
    assert out['measurements']['empty_slot_ids']==[3]
    assert out['measurements']['edges']==[dict(source=1,target=2,relation='pixel_membership_overlap',overlap_px=63)]
    assert not out['masks'].flags.writeable
    assert out['mask_sha256']==hashlib.sha256(out['masks'].tobytes()).hexdigest()
    assert out['measurements']['artifact'].checkpoint.endswith(kw['expected_model_sha256'])
    assert not out['training_admission'] and out['quality_accepted'] is None
    assert out['binding']['image_sha256']==kw['expected_image_sha256']
    assert modes==[x.training for x in m.modules()] and torch.equal(rng,torch.get_rng_state())
    assert instance_model_digest(m)==kw['expected_model_sha256']
    for p,grad in zip(m.parameters(),grads):assert torch.equal(p.grad,grad)
    for k,v in state.items():
        assert torch.equal(v,m.state_dict()[k]) if torch.is_tensor(v) else v==m.state_dict()[k]


@pytest.mark.parametrize('change',['pixel','model','image','parent','role','budget','asset','seed'])
def test_refusal_before_model_copy(change,monkeypatch):
    from mora3.model.color_instance_inference import infer_nuclear_instances
    m,image,pixels,kw=setup_case()
    if change in ('pixel','model','image'):kw['expected_'+change+'_sha256']='0'*64
    if change=='parent':image=replace(image,parent_id='changed')
    if change=='role':image=replace(image,role='target')
    if change=='budget':kw['max_mask_bytes']=188
    if change=='asset':kw['mask_asset']=image.asset
    if change=='seed':kw['seed']=True
    original=copy.deepcopy
    def guarded(obj,*args):
        if obj is m:pytest.fail('model copy before refusal')
        return original(obj,*args)
    monkeypatch.setattr(copy,'deepcopy',guarded)
    with pytest.raises(ValueError):infer_nuclear_instances(m,image,pixels,**kw)
