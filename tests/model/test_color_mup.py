"""Explicit RGB semantic/slot/context μP, tiny source-free image tensors."""
import copy
import math
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from mora3.model.color_semantic import ColorNuclearSemantic,ColorInstancePredictor
from mora3.model.color_conditioning import ConditionedColorSemantic,ConditionedColorInstance
from mora3.model.color_mup import color_mup_parameter_roles,color_mup_width_family,conditioned_color_mup_parameter_roles,conditioned_color_mup_width_family
from mora3.model.instance_affinity import OverlappingInstanceHead
from test_sciplex_mup import make as native


@pytest.fixture(autouse=True)
def seed():torch.manual_seed(2026110667)


def make(w=8,instance=False,shared=None):
    kw=dict(width=w,alpha_policy='require-opaque',parameterization_policy='mup-color-v1',base_width=8,output_mult=1.5)
    if instance:return ColorInstancePredictor(**kw,slots=3,max_output_bytes=100000,native_color=shared)
    return ColorNuclearSemantic(**kw,backbone=shared.backbone if shared is not None else None)


def conditioned(w=8,instance=False):
    return (ConditionedColorInstance if instance else ConditionedColorSemantic)(make(w,instance),id_width=4,
        parameterization_policy='mup-color-context-v1')


@pytest.mark.parametrize('instance',[False,True])
def test_complete_rgb_and_conditioner_families_qualify_actual_local_operators(instance):
    ms=[make(w,instance) for w in (8,12,16)]
    family=color_mup_width_family(*ms)
    assert family.total_numel==tuple(sum(p.numel() for p in m.parameters()) for m in ms)
    assert not any(r.role=='custom' for r in color_mup_parameter_roles(ms[-1]).values())
    cs=[conditioned(w,instance) for w in (8,12,16)]
    cf=conditioned_color_mup_width_family(*cs,identifier_axis=('g0','g1','g2','g3'))
    assert cf.total_numel==tuple(sum(p.numel() for p in m.parameters()) for m in cs)
    assert not any(r.role=='custom' for r in conditioned_color_mup_parameter_roles(cs[-1],identifier_axis=('g0','g1','g2','g3')).values())


@pytest.mark.parametrize('instance',[False,True])
def test_native_rgb_readout_equation_and_all_parameter_gradients(instance):
    m=make(16,instance).double();pixels=np.arange(90,dtype=np.uint8).reshape(5,6,3)
    x=torch.tensor(pixels,dtype=torch.float64).permute(2,0,1)[None]/255
    local=m.pixels(x)
    graph=dict(features=local.mean((2,3)),endpoints=torch.zeros((1,2),dtype=torch.long),
               roles=torch.zeros(1,dtype=torch.long),identifiers=torch.ones(1,1,dtype=torch.float64))
    latent=m.backbone([graph])[0][:,0];context=latent[:,:,None,None].expand(-1,-1,5,6)
    conv=m.head.projection if instance else m.head
    expected=F.conv2d(torch.cat((local,context),1)*(1.5/(16/8)),conv.weight,conv.bias)[0]
    if not instance:expected=expected[0]
    actual=m(pixels)
    torch.testing.assert_close(actual,expected,rtol=1e-11,atol=1e-11)
    params=tuple(m.parameters())
    for a,b in zip(torch.autograd.grad(actual.sum(),params,retain_graph=True),torch.autograd.grad(expected.sum(),params)):
        torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-10)


def test_new_slot_readout_fresh_scaling_rng_budget_and_no_double_division():
    torch.manual_seed(2026110668);old=OverlappingInstanceHead(24,slots=3,max_output_bytes=10000)
    rng=torch.get_rng_state().clone()
    torch.manual_seed(2026110668);new=OverlappingInstanceHead(24,slots=3,max_output_bytes=10000,
        parameterization_policy='mup-mask-slots-v1',base_channels=16,output_mult=1.5)
    assert torch.equal(rng,torch.get_rng_state())
    for n,p in old.named_parameters():assert torch.equal(dict(new.named_parameters())[n],p*math.sqrt(24/16))
    x=torch.linspace(-1,1,144).reshape(1,24,2,3)
    assert torch.equal(new(x),F.conv2d(x*(1.5/(24/16)),new.projection.weight,new.projection.bias)[0])
    new.max_output_bytes=1
    with pytest.raises(ValueError):new(x)


def test_shared_native_transforms_and_pixel_reuse_never_reinitialized():
    source=native(12);before={n:p.clone() for n,p in source.backbone.named_parameters()}
    semantic=make(12,False,source)
    for name in ('feature','role','graph_token','layers','final_norm'):
        assert getattr(semantic.backbone,name) is getattr(source.backbone,name)
    assert semantic.backbone.incidence is not source.backbone.incidence
    instance=make(12,True,semantic)
    assert instance.backbone is semantic.backbone and instance.pixels is semantic.pixels
    context=ConditionedColorSemantic(semantic,id_width=4,parameterization_policy='mup-color-context-v1')
    other=ConditionedColorInstance(instance,id_width=4,parameterization_policy='mup-color-context-v1',context=context.context)
    assert other.context is context.context
    for n,p in source.backbone.named_parameters():assert torch.equal(p,before[n])


def test_checkpoint_refusal_precedes_child_tensor_copy_and_never_reinitializes():
    m=conditioned(12,True);state=copy.deepcopy(m.state_dict());rng=torch.get_rng_state().clone()
    m.load_state_dict(state);assert torch.equal(rng,torch.get_rng_state())
    state['color.head._extra_state']['mup']['base_channels']=8
    state['color.pixels.enc0.0.weight'].fill_(99)
    before={n:p.clone() for n,p in m.named_parameters()}
    with pytest.raises(ValueError,match='configuration|policy'):m.load_state_dict(state)
    for n,p in m.named_parameters():assert torch.equal(p,before[n])


@pytest.mark.parametrize('instance',[False,True])
def test_legacy_state_and_rng_exact(instance):
    cls=ColorInstancePredictor if instance else ColorNuclearSemantic
    kw=dict(alpha_policy='require-opaque')
    if instance:kw.update(slots=3,max_output_bytes=100000)
    torch.manual_seed(2026110669);a=cls(8,**kw);rng=torch.get_rng_state().clone()
    torch.manual_seed(2026110669);b=cls(8,**kw,parameterization_policy='legacy')
    assert torch.equal(rng,torch.get_rng_state()) and a.get_extra_state()==b.get_extra_state()
    assert set(a.state_dict())==set(b.state_dict())
    for k,v in a.state_dict().items():
        if isinstance(v,torch.Tensor):assert torch.equal(v,b.state_dict()[k])


@pytest.mark.parametrize('instance',[False,True])
def test_complete_context_offset_equation_gradients_target_exclusion_and_fresh_scale(instance):
    color=make(16,instance).double()
    cls=ConditionedColorInstance if instance else ConditionedColorSemantic
    m=cls(color,id_width=4,parameterization_policy='mup-color-context-v1').double()
    torch.manual_seed(2026110670);linear=torch.nn.Linear(16,3 if instance else 1);rng=torch.get_rng_state().clone()
    torch.manual_seed(2026110670);other=cls(color,id_width=4,parameterization_policy='mup-color-context-v1',context=m.context)
    assert torch.equal(rng,torch.get_rng_state())
    assert torch.equal(other.offset.weight,linear.weight*math.sqrt(16/8))
    assert torch.equal(other.offset.bias,linear.bias*math.sqrt(16/8))
    from mora3.data.schema import GraphRecord,Node,Measurement
    from mora3.data.provenance import Provenance
    from mora3.data.tasks import ConditionalTask
    record=GraphRecord('request',(Node('object',(Measurement('x',.5,'um','observed',True),
        Measurement('target',99.,'um','observed',True))),),(),
        (Provenance('synthetic-unit','v1','request','a'*64,'no-rights-inferred'),))
    query=ConditionalTask(((('node',0,'x'),'condition'),(('node',0,'target'),'target')),topology_known=True)
    pixels=np.arange(90,dtype=np.uint8).reshape(5,6,3)
    context=m.context(record,query)
    offset=F.linear(context*(1.5/(16/8)),m.offset.weight,m.offset.bias)
    expected=m.color(pixels)+offset.reshape(3,1,1) if instance else m.color(pixels)+offset.reshape(())
    actual=m(pixels,record,query)
    torch.testing.assert_close(actual,expected,rtol=1e-11,atol=1e-11)
    params=tuple(m.parameters())
    for a,b in zip(torch.autograd.grad(actual.sum(),params,retain_graph=True),torch.autograd.grad(expected.sum(),params)):
        torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-10)
    from dataclasses import replace
    changed=replace(record,nodes=(replace(record.nodes[0],measurements=(record.nodes[0].measurements[0],
        replace(record.nodes[0].measurements[1],value=-999.))),))
    assert torch.equal(actual,m(pixels,changed,query))


@pytest.mark.parametrize('change',['channel','pixel','readout','source','base','slots'])
def test_source_operator_and_axis_drift_refuses(change):
    ms=[make(w,instance=True) for w in (8,12,16)];m=ms[-1]
    if change=='channel':m.pixels.enc0[0].in_channels=1
    elif change=='pixel':m.pixels.dec1[0].padding=(0,0)
    elif change=='readout':m.head.projection.groups=2
    elif change=='source':m.alpha_policy='ignore-alpha'
    elif change=='base':m.base_width=4
    else:m.head.slots=4
    with pytest.raises(ValueError):color_mup_width_family(*ms)
