import copy
import hashlib
import numpy as np
import pytest
import torch
from test_color_supervised_task import source
from mora3.model.color_supervised_task import ColorSemanticTask
from mora3.model.color_conditioning import ConditionedColorSemantic
from mora3.model.color_semantic import ColorNuclearSemantic
from mora3.model.color_semantic_inference import image_condition_graph, selected_image_condition_graph, image_artifact_digest

@pytest.mark.parametrize('context_fields',[None,('tissue','organism')])
@pytest.mark.parametrize('device',['cpu','cuda'])
def test_conditioned_objective_explicit_reference_and_all_gradients(device,context_fields):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('no CUDA')
    torch.manual_seed(18901)
    s=source();t=ColorSemanticTask(s,s.binding_sha256)
    m=ConditionedColorSemantic(ColorNuclearSemantic(4,alpha_policy='require-opaque'),id_width=1).to(device=device,dtype=torch.float64)
    ref=copy.deepcopy(m)
    kw={} if context_fields is None else dict(context_fields=context_fields,expected_image_sha256=image_artifact_digest(s.image))
    result=t.evaluate_conditioned(m,**kw)
    pixels=s.image_array()
    g,task=image_condition_graph(s.image,pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest())
    if context_fields is not None:
        g,task=selected_image_condition_graph(s.image,pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),**kw)
    x=ref.context.payload.forward_task(g,task=task,identifiers=torch.eye(1,device=device,dtype=torch.float64))
    latent=ref.context.graph([x])[0][:,0]
    logits=ref.color(pixels)+(latent@ref.offset.weight.T+ref.offset.bias).reshape(())
    target=torch.tensor(s.target_arrays()[0],device=device)
    oracle=(torch.nn.functional.softplus(logits)-target*logits).mean()
    torch.testing.assert_close(result['loss'],oracle)
    for a,b in zip(torch.autograd.grad(result['loss'],tuple(m.parameters())),torch.autograd.grad(oracle,tuple(ref.parameters()))):
        torch.testing.assert_close(a,b)
    assert not result['training_admission']
    assert not task.available_targets(g)
    assert result['source_sha256']==s.binding_sha256

def test_conditioned_rejects_unconditioned_model():
    s=source();t=ColorSemanticTask(s,s.binding_sha256)
    with pytest.raises(ValueError,match='conditioned'):t.evaluate_conditioned(ColorNuclearSemantic(4,alpha_policy='require-opaque'))

def test_changed_mask_changes_only_supervision_and_preserves_model_modes():
    from dataclasses import replace
    import io
    from PIL import Image
    s=source();mask=s.target_arrays()[0]
    b=io.BytesIO();Image.fromarray((~mask).astype('uint8')*255).save(b,format='PNG')
    raw=b.getvalue()
    changed=replace(s,masks_encoded=(raw,),mask_sha256=(hashlib.sha256(raw).hexdigest(),))
    torch.manual_seed(18901)
    m=ConditionedColorSemantic(ColorNuclearSemantic(4,alpha_policy='require-opaque'),id_width=1).double()
    modes=[x.training for x in m.modules()];rng=torch.get_rng_state().clone()
    first=ColorSemanticTask(s,s.binding_sha256);second=ColorSemanticTask(changed,changed.binding_sha256)
    assert np.array_equal(first.input_pixels(),second.input_pixels())
    pixels=first.input_pixels()
    g,t=image_condition_graph(s.image,pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest())
    logits=m(pixels,g,t)
    a=first.evaluate_conditioned(m)['loss'];b=second.evaluate_conditioned(m)['loss']
    torch.testing.assert_close(a-b,((1-2*torch.tensor(mask,dtype=logits.dtype))*logits).mean())
    assert modes==[x.training for x in m.modules()] and torch.equal(rng,torch.get_rng_state())
    assert all(p.grad is None for p in m.parameters())
    with pytest.raises(ValueError,match='held-out'):second.require_training()


def test_selected_context_binding_refuses():
    s=source();t=ColorSemanticTask(s,s.binding_sha256)
    m=ConditionedColorSemantic(ColorNuclearSemantic(4,alpha_policy='require-opaque'),id_width=1)
    with pytest.raises(ValueError,match='binding'):
        t.evaluate_conditioned(m,context_fields=('tissue',),expected_image_sha256='0'*64)
