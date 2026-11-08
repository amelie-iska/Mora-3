"""Source task forwards explicit extended policy; complete targets never truncate."""
import copy
import hashlib
from dataclasses import replace
import pytest
import numpy as np
import torch
from test_color_supervised_task import source
from mora3.model.color_instance_task import ColorInstanceTask
from mora3.model.color_conditioning import ConditionedColorInstance
from mora3.model.color_semantic import ColorInstancePredictor
from mora3.model.color_semantic_inference import image_condition_graph
from mora3.model.instance_affinity import overlapping_set_objective


def make(device, slots, budget=1000000):
    torch.manual_seed(23301)
    return ConditionedColorInstance(ColorInstancePredictor(4, slots=slots,
        max_output_bytes=budget, alpha_policy='require-opaque',
        slot_policy='extended-max512-v2'), id_width=1).to(device=device,dtype=torch.float64)

@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('slots',[65,512])
@pytest.mark.parametrize('block',[None,3])
def test_extended_task_loss_and_all_gradients(device,slots,block):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('no CUDA')
    s=source('ordered_binary_instances')
    s=replace(s,masks_encoded=s.masks_encoded*65,mask_sha256=s.mask_sha256*65)
    m=make(device,slots);ref=copy.deepcopy(m)
    actual=ColorInstanceTask(s,s.binding_sha256).evaluate_conditioned(m,pixel_block_size=block)
    pixels=s.image_array()
    g,t=image_condition_graph(s.image,pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest())
    z=ref(pixels,g,t)
    target=torch.from_numpy(np.stack(s.target_arrays())).to(device)
    expected=overlapping_set_objective(z,target,slot_policy='extended-max512-v2',pixel_block_size=block)
    torch.testing.assert_close(actual['loss'],expected,rtol=1e-12,atol=1e-12)
    for a,b in zip(torch.autograd.grad(actual['loss'],tuple(m.parameters())),torch.autograd.grad(expected,tuple(ref.parameters()))):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-12)
    assert actual['instances']==65 and actual['slots']==slots and not actual['training_admission']
    with pytest.raises(ValueError,match='admission|held-out'):ColorInstanceTask(s,s.binding_sha256).require_training()

@pytest.mark.parametrize('case',['budget','capacity'])
def test_extended_task_refuses_before_decode_or_any_forward(monkeypatch,case):
    s=source('ordered_binary_instances')
    if case=='capacity':s=replace(s,masks_encoded=s.masks_encoded*66,mask_sha256=s.mask_sha256*66)
    m=make('cpu',65,budget=1 if case=='budget' else 100000)
    def forbidden(*args):raise AssertionError('decode before refusal')
    monkeypatch.setattr(ColorInstanceTask,'input_pixels',forbidden)
    calls=[];hook=m.register_forward_pre_hook(lambda *a:calls.append(True))
    with pytest.raises(ValueError,match='output-byte' if case=='budget' else 'complete source'):
        ColorInstanceTask(s,s.binding_sha256).evaluate_conditioned(m)
    hook.remove();assert not calls
