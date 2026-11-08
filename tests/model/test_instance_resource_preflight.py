"""Byte equations are checked against actual tiny CPU/CUDA tensor storage."""
import pytest
import torch
from test_color_supervised_task import source

@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_storage_equations(device,dtype):
    from mora3.model.instance_resources import instance_resource_preflight
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('no CUDA')
    q,n,h,w=3,2,4,5
    r=instance_resource_preflight(instances=n,slots=q,height=h,width=w,
        element_bytes=torch.empty((),dtype=dtype).element_size(),max_output_bytes=10000)
    tensors=dict(logits_bytes=torch.empty(q,h,w,device=device,dtype=dtype),
        target_bool_bytes=torch.empty(n,h,w,device=device,dtype=torch.bool),
        matching_logits_fp64_bytes=torch.empty(q,h*w,device=device,dtype=torch.float64),
        matching_targets_fp64_bytes=torch.empty(n,h*w,device=device,dtype=torch.float64),
        matching_cost_fp64_bytes=torch.empty(q,q,device=device,dtype=torch.float64),
        assigned_target_bytes=torch.empty(q,h,w,device=device,dtype=dtype))
    for k,t in tensors.items():assert r[k]==t.numel()*t.element_size()
    assert r['current_reference_supported'] and r['refusals']==()
    assert r['pairwise_pixel_workspace_bytes']==0
    assert r['naive_pairwise_pixel_fp64_bytes']==q*n*h*w*8
    assert not r['training_admission'] and not r['peak_memory_estimate']

@pytest.mark.parametrize('n,q,budget,reason',[(465,465,10**12,'slot_policy'),(424,64,10**12,'instance_capacity'),(2,3,1,'output_budget')])
def test_complete_source_policy_refusals(n,q,budget,reason):
    from mora3.model.instance_resources import instance_resource_preflight
    r=instance_resource_preflight(instances=n,slots=q,height=1000,width=1000,
        element_bytes=8,max_output_bytes=budget)
    assert reason in r['refusals'] and not r['current_reference_supported']
    assert r['target_bool_bytes']==n*1000000

@pytest.mark.parametrize('key,value',[('instances',-1),('slots',0),('height',0),('width',True),('element_bytes',3),('max_output_bytes',0),('instances',1.0)])
def test_invalid_metadata(key,value):
    from mora3.model.instance_resources import instance_resource_preflight
    args=dict(instances=0,slots=1,height=2,width=2,element_bytes=4,max_output_bytes=100)
    args[key]=value
    with pytest.raises(ValueError):instance_resource_preflight(**args)

def test_source_bound_preflight_does_not_decode(monkeypatch):
    from mora3.model.color_instance_task import ColorInstanceTask
    from mora3.data.public_image_masks import PublicImageMasks
    s=source('ordered_binary_instances');task=ColorInstanceTask(s,s.binding_sha256)
    def forbidden(*args):raise AssertionError('preflight must not decode')
    monkeypatch.setattr(PublicImageMasks,'image_array',forbidden)
    monkeypatch.setattr(PublicImageMasks,'target_arrays',forbidden)
    r=task.resource_preflight(slots=3,element_bytes=8,max_output_bytes=10000)
    assert r['instances']==len(s.masks_encoded)
    assert r['source_sha256']==task.expected_source_sha256
    assert r['height']==s.image.spatial_shape[0]
