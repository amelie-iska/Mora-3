"""Source-bound complete ordered masks stay exclusively on the loss side."""
import copy
import hashlib
import itertools
from dataclasses import replace
import pytest
import torch
from test_color_supervised_task import source
from mora3.model.color_conditioning import ConditionedColorInstance
from mora3.model.color_semantic import ColorInstancePredictor
from mora3.model.color_semantic_inference import image_condition_graph, image_artifact_digest


def model(device='cpu'):
    torch.manual_seed(22401)
    return ConditionedColorInstance(ColorInstancePredictor(4,slots=3,
        max_output_bytes=4096,alpha_policy='require-opaque'),id_width=1).to(device=device,dtype=torch.float64)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_complete_mask_assignment_loss_and_all_parameter_gradients(device):
    from mora3.model.color_instance_task import ColorInstanceTask
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('no CUDA')
    s=source('ordered_binary_instances')
    # Duplicate overlapping memberships remain two distinct source instances.
    s=replace(s,masks_encoded=s.masks_encoded*2,mask_sha256=s.mask_sha256*2)
    m=model(device);ref=copy.deepcopy(m)
    result=ColorInstanceTask(s,s.binding_sha256).evaluate_conditioned(m)
    pixels=s.image_array()
    g,t=image_condition_graph(s.image,pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest())
    z=ref(pixels,g,t)
    targets=torch.zeros_like(z)
    targets[:2]=torch.tensor(s.target_arrays()[0],device=device)
    losses=torch.stack([(torch.nn.functional.softplus(z)-targets[list(p)]*z).mean()
                        for p in itertools.permutations(range(3))])
    expected=losses.min()
    torch.testing.assert_close(result['loss'],expected,rtol=1e-12,atol=1e-12)
    for a,b in zip(torch.autograd.grad(result['loss'],tuple(m.parameters())),
                   torch.autograd.grad(expected,tuple(ref.parameters()))):
        torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-12)
    assert result['instances']==2 and result['source_sha256']==s.binding_sha256
    assert not result['training_admission'] and not t.available_targets(g)


def test_mask_order_changes_binding_but_not_input_or_objective():
    from mora3.model.color_instance_task import ColorInstanceTask
    a=source('ordered_binary_instances');b=source('ordered_binary_instances')
    import io
    import numpy as np
    from PIL import Image
    buf=io.BytesIO();Image.fromarray((~b.target_arrays()[0]).astype(np.uint8)*255).save(buf,format='PNG')
    raw=buf.getvalue()
    s=replace(a,masks_encoded=a.masks_encoded+(raw,),mask_sha256=a.mask_sha256+(hashlib.sha256(raw).hexdigest(),))
    r=replace(s,masks_encoded=s.masks_encoded[::-1],mask_sha256=s.mask_sha256[::-1])
    m=model();m.train();m.color.pixels.eval()
    state=copy.deepcopy(m.state_dict());modes=[x.training for x in m.modules()];rng=torch.get_rng_state().clone()
    x=ColorInstanceTask(s,s.binding_sha256);y=ColorInstanceTask(r,r.binding_sha256)
    assert s.binding_sha256!=r.binding_sha256
    assert np.array_equal(x.input_pixels(),y.input_pixels())
    torch.testing.assert_close(x.evaluate_conditioned(m)['loss'],y.evaluate_conditioned(m)['loss'],rtol=0,atol=0)
    assert modes==[x.training for x in m.modules()] and torch.equal(rng,torch.get_rng_state())
    for k,v in m.state_dict().items():
        if isinstance(v,torch.Tensor):assert torch.equal(v,state[k])
        else:assert v==state[k]
    assert all(p.grad is None for p in m.parameters())


@pytest.mark.parametrize('case',['binding','encoding','type','capacity','context'])
def test_refusals_before_forward(case):
    from mora3.model.color_instance_task import ColorInstanceTask
    s=source('semantic_binary' if case=='encoding' else 'ordered_binary_instances')
    if case=='capacity':s=replace(s,masks_encoded=s.masks_encoded*4,mask_sha256=s.mask_sha256*4)
    m=model();calls=[]
    hook=m.register_forward_pre_hook(lambda *args:calls.append(True))
    with pytest.raises(ValueError):
        task=ColorInstanceTask(s,'0'*64 if case=='binding' else s.binding_sha256)
        task.evaluate_conditioned(torch.nn.Linear(1,1) if case=='type' else m,
            **(dict(context_fields=('tissue',),expected_image_sha256='0'*64) if case=='context' else {}))
    hook.remove();assert not calls


@pytest.mark.parametrize('split',['train','test','validation','evaluation'])
def test_training_stays_closed(split):
    from mora3.model.color_instance_task import ColorInstanceTask
    s=source('ordered_binary_instances',split=split)
    with pytest.raises(ValueError,match='admission|held-out'):ColorInstanceTask(s,s.binding_sha256).require_training()
