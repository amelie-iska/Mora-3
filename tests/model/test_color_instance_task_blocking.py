"""Explicit task loss policy preserves dense equations and source boundaries."""
import copy
from dataclasses import replace
import pytest
import torch
from test_color_instance_task import model
from test_color_supervised_task import source
from mora3.model.color_instance_task import ColorInstanceTask

@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('block',[1,7,1000])
@pytest.mark.parametrize('selected',[False,True])
def test_blocked_task_loss_all_gradients_and_policy(device,block,selected):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('no CUDA')
    s=source('ordered_binary_instances')
    s=replace(s,masks_encoded=s.masks_encoded*2,mask_sha256=s.mask_sha256*2)
    t=ColorInstanceTask(s,s.binding_sha256)
    a=model(device);b=copy.deepcopy(a)
    kwargs={}
    if selected:
        from mora3.model.color_semantic_inference import image_artifact_digest
        kwargs=dict(context_fields=('tissue',),expected_image_sha256=image_artifact_digest(s.image))
    dense=t.evaluate_conditioned(a,**kwargs)
    blocked=t.evaluate_conditioned(b,pixel_block_size=block,**kwargs)
    torch.testing.assert_close(dense['loss'],blocked['loss'],rtol=1e-12,atol=1e-12)
    for x,y in zip(torch.autograd.grad(dense['loss'],tuple(a.parameters())),torch.autograd.grad(blocked['loss'],tuple(b.parameters()))):
        torch.testing.assert_close(x,y,rtol=1e-10,atol=1e-12)
    assert dense['pixel_block_size'] is None and blocked['pixel_block_size']==block
    assert not blocked['training_admission']
    for k in dense:
        if k not in ('loss','pixel_block_size'):assert dense[k]==blocked[k]

@pytest.mark.parametrize('block',[0,-1,True,1.5,'2'])
def test_invalid_policy_refuses_before_neural_or_decode(monkeypatch,block):
    s=source('ordered_binary_instances');t=ColorInstanceTask(s,s.binding_sha256)
    def forbidden(*args):raise AssertionError('decode before refusal')
    monkeypatch.setattr(ColorInstanceTask,'input_pixels',forbidden)
    with pytest.raises(ValueError,match='pixel block'):t.evaluate_conditioned(model(),pixel_block_size=block)

def test_complete_capacity_refuses_before_decode_with_blocking(monkeypatch):
    s=source('ordered_binary_instances');s=replace(s,masks_encoded=s.masks_encoded*4,mask_sha256=s.mask_sha256*4)
    def forbidden(*args):raise AssertionError('decode before refusal')
    monkeypatch.setattr(ColorInstanceTask,'input_pixels',forbidden)
    with pytest.raises(ValueError,match='complete source'):ColorInstanceTask(s,s.binding_sha256).evaluate_conditioned(model(),pixel_block_size=1)

def test_task_forwards_explicit_policy_to_real_objective(monkeypatch):
    import mora3.model.color_instance_task as module
    actual=module.overlapping_set_objective;seen=[]
    def observe(logits,targets,*,pixel_block_size=None):
        seen.append(pixel_block_size)
        return actual(logits,targets,pixel_block_size=pixel_block_size)
    monkeypatch.setattr(module,'overlapping_set_objective',observe)
    s=source('ordered_binary_instances');t=ColorInstanceTask(s,s.binding_sha256)
    t.evaluate_conditioned(model(),pixel_block_size=2)
    t.evaluate_conditioned(model())
    assert seen==[2,None]
