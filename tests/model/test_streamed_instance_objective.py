"""Full-source slot/pixel equation and all gradients, tiny synthetic rasters."""
import copy
import pytest,torch
from mora3.model.instance_affinity import OverlappingInstanceHead,overlapping_set_objective


def envelope(n=5):
    from mora3.model.raster_source_envelope import RasterSourceEnvelope
    return RasterSourceEnvelope('a'*64,7,9,n)


def head(n=5):
    return OverlappingInstanceHead(4,slots=n,max_output_bytes=1_000_000,slot_policy='source-envelope-v3',
        source_envelope=envelope(n),parameterization_policy='mup-mask-slots-v1',base_channels=2,output_mult=1.5).double()


@pytest.mark.parametrize('slot_block,pixel_block,recompute',[(1,1,False),(2,13,True),(5,1000,True)])
def test_every_slot_pixel_matching_loss_and_all_gradients_equal_dense(slot_block,pixel_block,recompute):
    from mora3.model.streamed_instance_objective import streamed_affine_mask_set_objective
    torch.manual_seed(20261106141);h=head();x=torch.randn(1,4,7,9,dtype=torch.float64,requires_grad=True)
    offset=torch.randn(5,dtype=torch.float64,requires_grad=True);target=torch.zeros(3,7,9,dtype=torch.bool)
    target[0,1:5,2:6]=True;target[1,2:6,4:8]=True;target[2,0:3,1:5]=True
    dense=h(x)+offset[:,None,None];expected=overlapping_set_objective(dense,target)
    actual=streamed_affine_mask_set_objective(h,x,target,offset=offset,slot_block_size=slot_block,pixel_block_size=pixel_block,recompute=recompute)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    ps=(x,h.projection.weight,h.projection.bias,offset)
    a=torch.autograd.grad(actual,ps);b=torch.autograd.grad(expected,ps)
    for v,w in zip(a,b):torch.testing.assert_close(v,w,rtol=1e-11,atol=1e-12)


def test_fullsource1863slots_are_explicit_and_legacy_caps_unchanged():
    from mora3.model.raster_source_envelope import RasterSourceEnvelope
    for policy in('bounded-max64-v1','extended-max512-v2'):
        with pytest.raises(ValueError):OverlappingInstanceHead(4,slots=1863,max_output_bytes=10_000_000,slot_policy=policy)
    h=OverlappingInstanceHead(4,slots=1863,max_output_bytes=10_000_000,slot_policy='source-envelope-v3',
        source_envelope=RasterSourceEnvelope('b'*64,1000,1000,1863),parameterization_policy='mup-mask-slots-v1',base_channels=2)
    state=h.get_extra_state();assert state['version']==4 and state['source_envelope']['max_instances']==1863
    assert h(torch.zeros(1,4,2,3)).shape==(1863,2,3)
    bad=copy.deepcopy(h.state_dict());bad['_extra_state']['source_envelope']['max_instances']=100
    with pytest.raises(ValueError):h.load_state_dict(bad)


def test_streaming_resource_bound_and_no_target_crop_or_truncation():
    from mora3.model.streamed_instance_objective import streamed_affine_mask_set_objective
    h=head();x=torch.zeros(1,4,7,9,dtype=torch.float64)
    too_many=torch.ones(6,7,9,dtype=torch.bool)
    with pytest.raises(ValueError):streamed_affine_mask_set_objective(h,x,too_many)
    with pytest.raises(ValueError):streamed_affine_mask_set_objective(h,x,torch.ones(1,7,9,dtype=torch.bool),max_workspace_bytes=1)


def test_literal_empty_source_layer_new_mode_preserved_legacy_guard_unchanged():
    from mora3.model.streamed_instance_objective import streamed_affine_mask_set_objective
    from torch.nn import functional as F
    h=head();x=torch.randn(1,4,7,9,dtype=torch.float64,requires_grad=True)
    supplied_empty=torch.zeros(1,7,9,dtype=torch.bool)
    with pytest.raises(ValueError):overlapping_set_objective(h(x),supplied_empty)
    actual=streamed_affine_mask_set_objective(h,x,supplied_empty,slot_block_size=2,pixel_block_size=13)
    logits=h(x);expected=F.binary_cross_entropy_with_logits(logits,torch.zeros_like(logits))
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    for a,b in zip(torch.autograd.grad(actual,(x,h.projection.weight,h.projection.bias)),
                   torch.autograd.grad(expected,(x,h.projection.weight,h.projection.bias))):
        torch.testing.assert_close(a,b,rtol=1e-11,atol=1e-12)
