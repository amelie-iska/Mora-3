"""Explicit policy resource gate, without allocating full source tensors."""
import pytest
from mora3.model.instance_resources import instance_resource_preflight

@pytest.mark.parametrize('n',[65,424,465,512])
@pytest.mark.parametrize('size',[4,8])
def test_extended_complete_source_storage(n,size):
    r=instance_resource_preflight(instances=n,slots=n,height=1000,width=1000,
        element_bytes=size,max_output_bytes=n*1000000*size,
        slot_policy='extended-max512-v2')
    assert r['refusals']==() and r['current_reference_supported']
    assert r['slot_policy']=='extended-max512-v2'
    assert r['logits_bytes']==n*1000000*size
    assert r['matching_cost_fp64_bytes']==n*n*8
    assert not r['peak_memory_estimate'] and not r['training_admission']

@pytest.mark.parametrize('n,q,budget,reason',[(513,513,10**12,'slot_policy'),(466,465,10**12,'instance_capacity'),(465,465,1,'output_budget')])
def test_extended_refusals(n,q,budget,reason):
    r=instance_resource_preflight(instances=n,slots=q,height=1000,width=1000,
        element_bytes=4,max_output_bytes=budget,slot_policy='extended-max512-v2')
    assert reason in r['refusals']

@pytest.mark.parametrize('policy',['unknown',None,512,True])
def test_unknown_policy_refuses(policy):
    with pytest.raises(ValueError,match='slot policy'):
        instance_resource_preflight(instances=1,slots=1,height=1,width=1,
            element_bytes=4,max_output_bytes=4,slot_policy=policy)

def test_source_preflight_explicit_policy_without_decoding(monkeypatch):
    from test_color_supervised_task import source
    from mora3.model.color_instance_task import ColorInstanceTask
    from mora3.data.public_image_masks import PublicImageMasks
    s=source('ordered_binary_instances')
    def forbidden(*args):
        raise AssertionError('metadata preflight decoded source')
    monkeypatch.setattr(PublicImageMasks,'image_array',forbidden)
    monkeypatch.setattr(PublicImageMasks,'target_arrays',forbidden)
    r=ColorInstanceTask(s,s.binding_sha256).resource_preflight(slots=65,
        element_bytes=8,max_output_bytes=10**9,slot_policy='extended-max512-v2')
    assert not r['refusals'] and r['source_sha256']==s.binding_sha256
