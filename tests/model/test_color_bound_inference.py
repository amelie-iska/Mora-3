"""Image-only context must be source-bound and cannot accept a target graph."""
import copy
import hashlib
from dataclasses import replace
import numpy as np
import pytest
import torch
from test_color_semantic_inference import fixture
from mora3.model.color_conditioning import ConditionedColorSemantic
from mora3.model.color_semantic import ColorNuclearSemantic

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_bound_inference_reference_and_no_state_changes(device):
    from mora3.model.color_semantic_inference import infer_nuclear_semantic,model_digest,image_condition_graph
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('no CUDA')
    torch.manual_seed(17201)
    m=ConditionedColorSemantic(ColorNuclearSemantic(4,alpha_policy='require-opaque'),id_width=1).to(device=device,dtype=torch.float64)
    m.train();m.color.pixels.eval()
    image,pixels=fixture();sha=hashlib.sha256(pixels.tobytes()).hexdigest()
    graph,task=image_condition_graph(image,pixel_sha256=sha)
    ms={v.name:v for v in graph.nodes[0].measurements}
    assert ms['height'].value==7 and ms['width'].value==9
    assert not ms['pixel_size_um'].available and task.available_targets(graph)==()
    assert graph.provenance[0].sha256==sha
    assert graph.leakage_keys[0].value==image.parent_id
    assert len(graph.nodes)==1 and not graph.edges
    snapshot=copy.deepcopy(m.state_dict());modes=[x.training for x in m.modules()]
    rng=torch.get_rng_state().clone();crng=torch.cuda.get_rng_state().clone()
    digest=model_digest(m)
    out=infer_nuclear_semantic(m,image,pixels,expected_pixel_sha256=sha,expected_model_sha256=digest,mask_asset='new.npy',seed=17201)
    ref=copy.deepcopy(m).eval()
    with torch.inference_mode():
        tensor=ref.context.payload.forward_task(graph,task=task,identifiers=torch.eye(1,device=device,dtype=torch.float64))
        latent=ref.context.graph([tensor])[0][:,0]
        logits=ref.color(pixels)+(latent@ref.offset.weight.T+ref.offset.bias).reshape(())
    assert np.array_equal(out['mask'],(logits>=0).cpu().numpy())
    assert out['conditioning']['graph_sha256']==hashlib.sha256(graph.canonical_bytes()).hexdigest()
    assert out['conditioning']['pixel_sha256']==sha and not out['conditioning']['training_admission']
    assert 'image-only' in out['artifact'].representation
    assert out['artifact'].checkpoint.endswith(digest)
    assert out['artifact'].links==(('mask_of',image.artifact_id),)
    assert not out['mask'].flags.writeable and out['instance_count'] is None
    assert not out['biological_validation'] and out['quality_accepted'] is None
    assert model_digest(m)==digest and modes==[x.training for x in m.modules()]
    assert torch.equal(rng,torch.get_rng_state()) and torch.equal(crng,torch.cuda.get_rng_state())
    for k,v in snapshot.items():
        assert torch.equal(v,m.state_dict()[k]) if torch.is_tensor(v) else v==m.state_dict()[k]
    # Source/parent identity changes audit binding, not network features.
    other,_=image_condition_graph(replace(image,parent_id='other'),pixel_sha256=sha)
    assert other.canonical_bytes()!=graph.canonical_bytes()
    assert other.nodes==graph.nodes
    with pytest.raises(TypeError):infer_nuclear_semantic(m,image,pixels,record=graph,task=task)

@pytest.mark.parametrize('change',['role','object','channels','frame','hash'])
def test_context_refuses_inapplicable_images(change):
    from mora3.model.color_semantic_inference import image_condition_graph
    image,_=fixture();sha='a'*64
    if change=='role':image=replace(image,role='target')
    if change=='object':image=replace(image,context=(('object_type','whole_cell'),))
    if change=='channels':image=replace(image,channels=('blue','green','red'))
    if change=='frame':image=replace(image,reference_frame='other')
    if change=='hash':sha='invalid'
    with pytest.raises(ValueError):image_condition_graph(image,pixel_sha256=sha)

@pytest.mark.parametrize('binding',['pixel','model'])
def test_conditioned_inference_rejects_wrong_actual_binding(binding):
    from mora3.model.color_semantic_inference import infer_nuclear_semantic,model_digest
    image,pixels=fixture()
    m=ConditionedColorSemantic(ColorNuclearSemantic(4,alpha_policy='require-opaque'),id_width=1)
    kw=dict(expected_pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
            expected_model_sha256=model_digest(m),mask_asset='new.npy',seed=17201)
    kw['expected_'+binding+'_sha256']='0'*64
    with pytest.raises(ValueError,match='binding mismatch'):
        infer_nuclear_semantic(m,image,pixels,**kw)


def test_physical_scale_is_observed_only_when_supplied():
    from mora3.model.color_semantic_inference import image_condition_graph
    image,_=fixture()
    a,ta=image_condition_graph(image,pixel_sha256='a'*64)
    b,tb=image_condition_graph(replace(image,pixel_size_um=0.5),pixel_sha256='a'*64)
    ref=('node',0,'pixel_size_um')
    assert ref not in ta.input_fields(a) and ref in tb.input_fields(b)
    scale=next(m for m in b.nodes[0].measurements if m.name=='pixel_size_um')
    assert scale.value==0.5 and scale.units=='um/pixel' and scale.status=='observed'
