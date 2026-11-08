"""Blocked matching storage must scale with block pixels, not image pixels."""
import pytest
import torch
from mora3.model.instance_resources import instance_resource_preflight

@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('block',[1,7,100])
def test_blocked_storage_matches_tensor_extents(device,block):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('no CUDA')
    r=instance_resource_preflight(instances=2,slots=3,height=4,width=5,
        element_bytes=4,max_output_bytes=1000,pixel_block_size=block)
    b=min(block,20)
    shapes={'matching_logits_fp64_bytes':(3,b),
            'matching_targets_fp64_bytes':(2,b),
            'matching_softplus_fp64_bytes':(3,b),
            'matching_cost_fp64_bytes':(3,3)}
    for key,shape in shapes.items():
        t=torch.empty(shape,dtype=torch.float64,device=device)
        assert r[key]==t.numel()*t.element_size()
    assert r['logits_bytes']==240 and r['assigned_target_bytes']==240
    assert r['target_bool_bytes']==40
    assert r['version']=='ordered-instance-blocked-storage-preflight-v1'
    assert r['pixel_block_size']==block
    assert not r['peak_memory_estimate'] and not r['training_admission']

@pytest.mark.parametrize('block',[0,-1,True,1.5,'10'])
def test_bad_block_refuses(block):
    with pytest.raises(ValueError,match='block'):
        instance_resource_preflight(instances=1,slots=1,height=1,width=1,
            element_bytes=4,max_output_bytes=4,pixel_block_size=block)

def test_source_blocked_gate_no_decode(monkeypatch):
    from test_color_supervised_task import source
    from mora3.model.color_instance_task import ColorInstanceTask
    from mora3.data.public_image_masks import PublicImageMasks
    s=source('ordered_binary_instances')
    def forbidden(*a):raise AssertionError('decoded')
    monkeypatch.setattr(PublicImageMasks,'image_array',forbidden)
    monkeypatch.setattr(PublicImageMasks,'target_arrays',forbidden)
    r=ColorInstanceTask(s,s.binding_sha256).resource_preflight(slots=65,
        element_bytes=4,max_output_bytes=10**9,slot_policy='extended-max512-v2',pixel_block_size=3)
    assert r['source_sha256']==s.binding_sha256
    assert r['slot_policy']=='extended-max512-v2'
    assert r['matching_logits_fp64_bytes']==65*3*8
