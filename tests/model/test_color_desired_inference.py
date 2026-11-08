import copy
import hashlib
from dataclasses import replace
import numpy as np
import pytest
import torch
from test_color_selected_context import sample
from mora3.model import color_semantic_inference as api
from mora3.model.color_conditioning import ConditionedColorSemantic
from mora3.model.color_semantic import ColorNuclearSemantic


def request(image):
    return api.desired_color_context_request(image,expected_image_sha256=api.image_artifact_digest(image),request_id='test-request',origin='hypothetical-test',values=(('tissue','desired-tissue'),))


def test_desired_graph_roles_and_identity_exclusion():
    image,pixels=sample();r=request(image)
    kw=dict(pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),context_fields=('tissue','organism'))
    g,t=api.desired_image_condition_graph(image,desired_context=r,**kw)
    ms={m.name:m for m in g.nodes[0].measurements}
    assert ms['source_context:tissue'].value=='breast'
    assert ms['desired_context:tissue'].value=='desired-tissue'
    assert dict(t.roles)[('node',0,'desired_context:tissue')]=='desired'
    assert not t.available_targets(g)
    g2,t2=api.desired_image_condition_graph(image,desired_context=replace(r,origin='other',request_id='other'),**kw)
    assert g.nodes==g2.nodes and t==t2
    with pytest.raises(ValueError,match='binding'):
        api.desired_image_condition_graph(replace(image,parent_id='other'),desired_context=r,**kw)
    with pytest.raises(ValueError):api.desired_image_condition_graph(image,desired_context={},**kw)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_desired_inference_reference_and_state(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('no CUDA')
    torch.manual_seed(18501)
    m=ConditionedColorSemantic(ColorNuclearSemantic(4,alpha_policy='require-opaque'),id_width=1).to(device=device,dtype=torch.float64)
    image,pixels=sample();r=request(image);sha=hashlib.sha256(pixels.tobytes()).hexdigest()
    digest=api.model_digest(m);modes=[x.training for x in m.modules()];rng=torch.get_rng_state().clone()
    g,t=api.desired_image_condition_graph(image,pixel_sha256=sha,desired_context=r,context_fields=('tissue',))
    out=api.infer_nuclear_semantic(m,image,pixels,expected_pixel_sha256=sha,expected_model_sha256=digest,mask_asset='desired.npy',seed=18501,context_fields=('tissue',),expected_image_sha256=r.image_sha256,desired_context=r)
    ref=copy.deepcopy(m).eval()
    with torch.inference_mode():
        x=ref.context.payload.forward_task(g,task=t,identifiers=torch.eye(1,device=device,dtype=torch.float64))
        latent=ref.context.graph([x])[0][:,0]
        logits=ref.color(pixels)+(latent@ref.offset.weight.T+ref.offset.bias).reshape(())
    assert np.array_equal(out['mask'],(logits>=0).cpu().numpy())
    assert out['conditioning']['desired_request_sha256']==hashlib.sha256(r.canonical_bytes()).hexdigest()
    assert out['conditioning']['policy']=='source-image-explicit-desired-context-v1'
    assert 'desired-context' in out['artifact'].representation
    assert out['artifact'].context==image.context
    assert not out['biological_validation'] and out['quality_accepted'] is None
    assert digest==api.model_digest(m) and modes==[x.training for x in m.modules()]
    assert torch.equal(rng,torch.get_rng_state())
