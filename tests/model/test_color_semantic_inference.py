import copy
import hashlib
from dataclasses import replace
import numpy as np
import pytest
import torch
from mora3.data.cooperative_segmentation import SegmentationArtifact
from mora3.model.color_semantic import ColorNuclearSemantic


def fixture():
    a=SegmentationArtifact('image','image','source_observed','condition','source','parent',
        (('object_type','nucleus'),),'source.png','source-yx-no-transform-v1',
        channels=('red','green','blue'),spatial_shape=(7,9))
    p=np.random.default_rng(16701).integers(0,256,(7,9,3),dtype=np.uint8)
    return a,p

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_inference_equation_provenance_and_state(device):
    from mora3.model.color_semantic_inference import infer_nuclear_semantic, model_digest
    torch.manual_seed(16701)
    m=ColorNuclearSemantic(4,alpha_policy='require-opaque').to(device=device,dtype=torch.float64)
    m.train();m.pixels.eval()
    a,p=fixture();before=copy.deepcopy(m.state_dict());modes=[x.training for x in m.modules()]
    rng=torch.get_rng_state().clone();crng=torch.cuda.get_rng_state().clone()
    digest=model_digest(m)
    result=infer_nuclear_semantic(m,a,p,expected_pixel_sha256=hashlib.sha256(p.tobytes()).hexdigest(),
        expected_model_sha256=digest,mask_asset='new-mask.npy',seed=16701)
    ref=copy.deepcopy(m).eval()
    with torch.no_grad(): expected=ref(p)>=0
    assert np.array_equal(result['mask'],expected.cpu().numpy())
    out=result['artifact']
    assert out.evidence=='predicted_mask' and out.role=='evaluation'
    assert out.links==(('mask_of','image'),) and out.context==a.context
    assert out.pixel_size_um is None and result['instance_count'] is None
    assert result['quality_accepted'] is None and not result['biological_validation']
    assert out.checkpoint.endswith(digest) and not result['mask'].flags.writeable
    assert model_digest(m)==digest and [x.training for x in m.modules()]==modes
    assert torch.equal(rng,torch.get_rng_state()) and torch.equal(crng,torch.cuda.get_rng_state())
    for k,v in before.items():
        if torch.is_tensor(v): assert torch.equal(v,m.state_dict()[k])

@pytest.mark.parametrize('case',['pixels','model','channels','frame','role','object','asset'])
def test_reject_unbound_inputs(case):
    from mora3.model.color_semantic_inference import infer_nuclear_semantic,model_digest
    m=ColorNuclearSemantic(4,alpha_policy='require-opaque');a,p=fixture()
    kw=dict(expected_pixel_sha256=hashlib.sha256(p.tobytes()).hexdigest(),expected_model_sha256=model_digest(m),mask_asset='mask.npy',seed=1)
    if case in ('pixels','model'):kw['expected_'+('pixel' if case=='pixels' else 'model')+'_sha256']='0'*64
    if case=='channels':a=replace(a,channels=('blue','green','red'))
    if case=='frame':a=replace(a,reference_frame='other')
    if case=='role':a=replace(a,role='target')
    if case=='object':a=replace(a,context=(('object_type','whole_cell'),))
    if case=='asset':kw['mask_asset']=a.asset
    with pytest.raises(ValueError):infer_nuclear_semantic(m,a,p,**kw)
