"""Ordered native mask layers/count; complete tiny source observations only."""
import numpy as np
import pytest,torch
from torch.nn import functional as F
from mora3.model.color_semantic import ColorNuclearSemantic,ColorInstancePredictor
from mora3.model.color_mup import color_mup_width_family,color_mup_parameter_roles
from mora3.model.raster_source_envelope import RasterSourceEnvelope


def make(width=16):
    e=RasterSourceEnvelope('d'*64,512,512,5)
    native=ColorNuclearSemantic(width,alpha_policy='require-opaque',input_policy='source-envelope-v2',source_envelope=e,
        parameterization_policy='mup-color-v1',base_width=8,output_mult=1.5)
    return ColorInstancePredictor(width,slots=5,max_output_bytes=1_000_000,alpha_policy='require-opaque',input_policy='source-envelope-v2',
        source_envelope=e,slot_policy='source-layer-list-v4',parameterization_policy='mup-color-v1',base_width=8,output_mult=1.5,native_color=native).double().eval()


def test_ordered_layer_count_loss_complete_empty_layer_and_padding_gradient_exclusion():
    m=make();pixels=np.arange(7*9*3,dtype=np.uint8).reshape(7,9,3)
    supplied=torch.zeros(3,7,9,dtype=torch.bool);supplied[0,1:5,2:6]=True;supplied[2,3:6,1:8]=True # plane1 observedempty
    result=m.layer_list_objective(pixels,supplied,slot_block_size=2,pixel_block_size=13,return_components=True)
    dense=m.forward_layer_list(pixels)
    count=F.cross_entropy(dense['count_logits'],torch.tensor([3]))
    mask=F.binary_cross_entropy_with_logits(dense['mask_logits'][:3].double(),supplied.double())
    torch.testing.assert_close(result['loss'],count+mask,rtol=1e-12,atol=1e-12)
    ps=tuple(m.parameters());a=torch.autograd.grad(result['loss'],ps,allow_unused=True);b=torch.autograd.grad(count+mask,ps,allow_unused=True)
    for x,y in zip(a,b):
        assert(x is None)==(y is None)
        if x is not None:torch.testing.assert_close(x,y,rtol=1e-10,atol=1e-11)
    g={id(p):v for p,v in zip(ps,a)}
    assert g[id(m.count.weight)].abs().sum()>0
    assert torch.equal(g[id(m.head.projection.weight)][3:],torch.zeros_like(g[id(m.head.projection.weight)][3:]))
    assert g[id(m.head.projection.weight)][1].abs().sum()>0 # observedempty is supervised, notpadding


def test_count_and_first_k_preserve_empty_planes_and_zero_layer_list():
    m=make();pixels=np.zeros((7,9,3),dtype=np.uint8)
    with torch.no_grad():
        m.count.weight.zero_();m.count.bias.fill_(-100);m.count.bias[3]=100
        m.head.projection.weight.zero_();m.head.projection.bias.fill_(-100)
    result=m.generate_layer_list(pixels)
    assert result['source_layer_count']==3 and result['ordered_masks'].shape==(3,7,9)
    assert not result['ordered_masks'].any()
    with torch.no_grad():m.count.bias.fill_(-100);m.count.bias[0]=100
    assert m.generate_layer_list(pixels)['ordered_masks'].shape==(0,7,9)
    empty=torch.zeros(0,7,9,dtype=torch.bool);parts=m.layer_list_objective(pixels,empty,return_components=True)
    assert parts['mask_loss']==0 and torch.isfinite(parts['count_loss'])


def test_new_count_axis_source_width_rules_and_old_set_mode_distinct():
    ms=tuple(make(w)for w in(8,12,16));family=color_mup_width_family(*ms);roles=color_mup_parameter_roles(ms[-1])
    assert roles['count.weight'].role=='output' and roles['count.bias'].role=='vector'
    assert family.total_numel[-1]==sum(p.numel()for p in ms[-1].parameters())
    state=ms[-1].get_extra_state();assert state['version']==5
    assert state['mup']['source_layer_list']['mask_padding']=='not-observed;excluded-from-loss'


def test_supplied_order_is_neural_target_and_count_sampling_uses_private_rng():
    torch.manual_seed(20261106144);m=make();pixels=np.arange(7*9*3,dtype=np.uint8).reshape(7,9,3)
    planes=torch.zeros(2,7,9,dtype=torch.bool);planes[0,1:4,1:5]=True;planes[1,3:7,4:9]=True
    a=m.layer_list_objective(pixels,planes,return_components=True);b=m.layer_list_objective(pixels,planes.flip(0),return_components=True)
    torch.testing.assert_close(a['count_loss'],b['count_loss'],rtol=0,atol=0)
    assert not torch.equal(a['mask_loss'],b['mask_loss'])
    state=torch.get_rng_state().clone();g=torch.Generator().manual_seed(20261106145)
    result=m.generate_layer_list(pixels,count_rule='sample',generator=g)
    assert torch.equal(state,torch.get_rng_state())
    assert result['ordered_masks'].shape[0]==result['source_layer_count']


def test_source_list_conditioned_count_and_masks_share_full_context_gradients():
    from mora3.model.color_conditioning import ConditionedColorInstance
    from mora3.model.color_mup import conditioned_color_mup_parameter_roles
    from mora3.data.schema import GraphRecord,Node,Measurement
    from mora3.data.tasks import ConditionalTask
    from mora3.data.provenance import Provenance
    m=ConditionedColorInstance(make(),id_width=2,parameterization_policy='mup-color-context-v1').double().eval()
    pixels=np.arange(7*9*3,dtype=np.uint8).reshape(7,9,3)
    record=GraphRecord('synthetic-context',(Node('image',(Measurement('height',7,'pixel','observed',True),Measurement('target',None,None,'missing',False))),),(),
        (Provenance('synthetic-unit','v1','context','a'*64,'unit-only-no-source-admission'),))
    task=ConditionalTask(((('node',0,'height'),'condition'),(('node',0,'target'),'target')),topology_known=True)
    planes=torch.zeros(2,7,9,dtype=torch.bool);planes[0,1:4,1:5]=True
    actual=m.layer_list_objective(pixels,record,task,planes,slot_block_size=1,pixel_block_size=13)
    dense=m(pixels,record,task)
    expected=F.cross_entropy(dense['count_logits'],torch.tensor([2]))+F.binary_cross_entropy_with_logits(dense['mask_logits'][:2],planes.double())
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    ps=tuple(m.parameters());a=torch.autograd.grad(actual,ps,allow_unused=True);b=torch.autograd.grad(expected,ps,allow_unused=True)
    for x,y in zip(a,b):
        assert(x is None)==(y is None)
        if x is not None:torch.testing.assert_close(x,y,rtol=1e-10,atol=1e-11)
    roles=conditioned_color_mup_parameter_roles(m,identifier_axis=('context:0','context:1'))
    assert roles['color.count.weight'].role=='output'
