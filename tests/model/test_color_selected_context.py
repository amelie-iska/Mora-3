"""Only selected source artifact context enters color inference; no target proxy API."""
import copy
import hashlib
from dataclasses import replace
import numpy as np
import pytest
import torch
from test_color_semantic_inference import fixture
from mora3.model import color_semantic_inference as api
from mora3.model.color_conditioning import ConditionedColorSemantic
from mora3.model.color_semantic import ColorNuclearSemantic


def sample():
    image,pixels=fixture()
    return replace(image,context=image.context+(('tissue','breast'),('organism',None),('nuclear_area','secret-target'))),pixels


def test_selected_graph_missingness_and_target_exclusion():
    image,pixels=sample();sha=hashlib.sha256(pixels.tobytes()).hexdigest()
    graph,task=api.selected_image_condition_graph(image,pixel_sha256=sha,
        context_fields=('tissue','organism'),expected_image_sha256=api.image_artifact_digest(image))
    ms={m.name:m for m in graph.nodes[0].measurements}
    assert ms['source_context:tissue'].value=='breast'
    assert ms['source_context:organism'].status=='missing'
    assert ('node',0,'source_context:organism') not in task.input_fields(graph)
    assert not task.available_targets(graph)
    assert all('secret-target' not in str(m) for m in ms.values())
    other=replace(image,context=tuple((k,'changed-target' if k=='nuclear_area' else v) for k,v in image.context))
    g2,t2=api.selected_image_condition_graph(other,pixel_sha256=sha,context_fields=('tissue','organism'),expected_image_sha256=api.image_artifact_digest(other))
    assert graph.nodes==g2.nodes and graph.edges==g2.edges and task==t2


@pytest.mark.parametrize('change',['source_id','parent_id','artifact_id','context'])
def test_stale_artifact_binding_refuses(change):
    image,_=sample();digest=api.image_artifact_digest(image)
    changed=replace(image,**{change: (('object_type','nucleus'),) if change=='context' else 'different'})
    with pytest.raises(ValueError,match='artifact binding'):
        api.selected_image_condition_graph(changed,pixel_sha256='a'*64,context_fields=('tissue',),expected_image_sha256=digest)


@pytest.mark.parametrize('fields',[('nuclear_area',),('mask',),('response',),('tissue','tissue'),'tissue',(),('desired:tissue',)])
def test_selection_rejects_unsupported_or_ambiguous_fields(fields):
    image,_=sample()
    with pytest.raises(ValueError):
        api.selected_image_condition_graph(image,pixel_sha256='a'*64,context_fields=fields,expected_image_sha256=api.image_artifact_digest(image))


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_selected_inference_explicit_reference_and_state(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('no CUDA')
    torch.manual_seed(17401)
    model=ConditionedColorSemantic(ColorNuclearSemantic(4,alpha_policy='require-opaque'),id_width=1).to(device=device,dtype=torch.float64)
    image,pixels=sample();sha=hashlib.sha256(pixels.tobytes()).hexdigest()
    digest=api.model_digest(model);rng=torch.get_rng_state().clone()
    graph,task=api.selected_image_condition_graph(image,pixel_sha256=sha,context_fields=('tissue','organism'),expected_image_sha256=api.image_artifact_digest(image))
    result=api.infer_nuclear_semantic(model,image,pixels,expected_pixel_sha256=sha,expected_model_sha256=digest,mask_asset='new.npy',seed=17401,context_fields=('tissue','organism'),expected_image_sha256=api.image_artifact_digest(image))
    ref=copy.deepcopy(model).eval()
    with torch.inference_mode():
        x=ref.context.payload.forward_task(graph,task=task,identifiers=torch.eye(1,device=device,dtype=torch.float64))
        latent=ref.context.graph([x])[0][:,0]
        logits=ref.color(pixels)+(latent@ref.offset.weight.T+ref.offset.bias).reshape(())
    assert np.array_equal(result['mask'],(logits>=0).cpu().numpy())
    assert result['conditioning']['policy']=='source-observed-selected-context-v1'
    assert result['conditioning']['image_artifact_sha256']==api.image_artifact_digest(image)
    assert not result['conditioning']['biological_pairing']
    assert api.model_digest(model)==digest and torch.equal(rng,torch.get_rng_state())
    assert 'selected-context' in result['artifact'].representation


def test_selected_context_requires_conditioned_model_and_complete_binding():
    image,pixels=sample()
    common=dict(expected_pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),mask_asset='new.npy',seed=17401)
    plain=ColorNuclearSemantic(4,alpha_policy='require-opaque')
    with pytest.raises(ValueError,match='conditioned'):
        api.infer_nuclear_semantic(plain,image,pixels,expected_model_sha256=api.model_digest(plain),context_fields=('tissue',),expected_image_sha256=api.image_artifact_digest(image),**common)
    model=ConditionedColorSemantic(plain,id_width=1)
    for fields,binding in [(('tissue',),None),(None,api.image_artifact_digest(image))]:
        with pytest.raises(ValueError,match='binding'):
            api.infer_nuclear_semantic(model,image,pixels,expected_model_sha256=api.model_digest(model),context_fields=fields,expected_image_sha256=binding,**common)


def test_selection_order_is_canonical_and_observed_selection_changes_features():
    image,pixels=sample();sha=hashlib.sha256(pixels.tobytes()).hexdigest()
    kw=dict(pixel_sha256=sha,expected_image_sha256=api.image_artifact_digest(image))
    a,ta=api.selected_image_condition_graph(image,context_fields=('tissue','organism'),**kw)
    b,tb=api.selected_image_condition_graph(image,context_fields=('organism','tissue'),**kw)
    assert a==b and ta==tb
    c,tc=api.selected_image_condition_graph(image,context_fields=('organism',),**kw)
    assert ('node',0,'source_context:tissue') in ta.input_fields(a)
    assert ('node',0,'source_context:tissue') not in tc.input_fields(c)
