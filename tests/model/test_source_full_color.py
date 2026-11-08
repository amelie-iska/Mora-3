"""Opt-in full-source RGB/context capacity; tiny synthetic complete rasters."""
import numpy as np
import pytest,torch
from mora3.model.color_semantic import ColorNuclearSemantic,ColorInstancePredictor
from mora3.model.color_mup import color_mup_width_family,color_mup_parameter_roles
from mora3.model.instance_affinity import overlapping_set_objective


def make(width=8):
    from mora3.model.raster_source_envelope import RasterSourceEnvelope
    e=RasterSourceEnvelope('c'*64,1024,1024,5)
    native=ColorNuclearSemantic(width,alpha_policy='require-opaque',input_policy='source-envelope-v2',
        source_envelope=e,parameterization_policy='mup-color-v1',base_width=8,output_mult=1.5)
    return ColorInstancePredictor(width,slots=5,max_output_bytes=100_000_000,alpha_policy='require-opaque',input_policy='source-envelope-v2',
        source_envelope=e,slot_policy='source-envelope-v3',parameterization_policy='mup-color-v1',base_width=8,output_mult=1.5,native_color=native).double()


def test_full_source_rgb_streamed_loss_all_native_encoder_and_readout_gradients():
    torch.manual_seed(20261106142);m=make(16).eval();pixels=np.arange(7*9*3,dtype=np.uint8).reshape(7,9,3)
    target=torch.zeros(3,7,9,dtype=torch.bool);target[0,1:5,1:4]=True;target[1,3:6,2:7]=True;target[2,1:4,4:8]=True
    actual=m.streamed_objective(pixels,target,slot_block_size=2,pixel_block_size=13,recompute_pixels=True)
    expected=overlapping_set_objective(m(pixels),target)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    ps=tuple(m.parameters());a=torch.autograd.grad(actual,ps,allow_unused=True);b=torch.autograd.grad(expected,ps,allow_unused=True)
    for x,y in zip(a,b):
        assert(x is None)==(y is None)
        if x is not None:torch.testing.assert_close(x,y,rtol=1e-10,atol=1e-11)


def test_complete_source_slot_and_raster_axis_family_is_explicit_finite():
    ms=tuple(make(w)for w in(8,12,16));family=color_mup_width_family(*ms);roles=color_mup_parameter_roles(ms[-1])
    assert family.total_numel[-1]==sum(p.numel()for p in ms[-1].parameters())
    assert roles['head.projection.weight'].role=='output'
    assert ms[-1].get_extra_state()['version']==4
    assert ms[-1].head.get_extra_state()['source_envelope']['max_height']==1024


def test_full_source_conditioned_context_offset_all_gradients_equal_dense():
    from mora3.model.color_conditioning import ConditionedColorInstance
    from mora3.data.schema import GraphRecord,Node,Measurement
    from mora3.data.tasks import ConditionalTask
    from mora3.data.provenance import Provenance
    torch.manual_seed(20261106143);m=ConditionedColorInstance(make(16),id_width=2,parameterization_policy='mup-color-context-v1').double().eval()
    pixels=np.arange(7*9*3,dtype=np.uint8).reshape(7,9,3)
    record=GraphRecord('synthetic-context',(Node('image',(
        Measurement('height',7,'pixel','observed',True),Measurement('target',None,None,'missing',False))),),(),
        (Provenance('synthetic-unit','v1','knowncontext','a'*64,'unit-only-no-source-admission'),))
    task=ConditionalTask(((('node',0,'height'),'condition'),(('node',0,'target'),'target')),topology_known=True)
    masks=torch.zeros(2,7,9,dtype=torch.bool);masks[0,1:6,1:4]=True;masks[1,2:6,3:8]=True
    actual=m.streamed_objective(pixels,record,task,masks,slot_block_size=2,pixel_block_size=13,recompute_pixels=True)
    expected=overlapping_set_objective(m(pixels,record,task),masks)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    ps=tuple(m.parameters());a=torch.autograd.grad(actual,ps,allow_unused=True);b=torch.autograd.grad(expected,ps,allow_unused=True)
    for x,y in zip(a,b):
        assert(x is None)==(y is None)
        if x is not None:torch.testing.assert_close(x,y,rtol=1e-10,atol=1e-11)


def test_source_semantic_checkpoint_metadata_is_primitive_and_legacy1000_schema_exact():
    from mora3.model.raster_source_envelope import RasterSourceEnvelope
    def primitive(x):
        if type(x)in(str,int,float,bool,type(None)):return True
        if type(x)in(tuple,list):return all(primitive(v)for v in x)
        return type(x)is dict and all(type(k)is str and primitive(v)for k,v in x.items())
    legacy=ColorNuclearSemantic(8,alpha_policy='require-opaque',input_policy='native-max1000-v1')
    assert 'source_envelope'not in legacy.get_extra_state()
    assert legacy.get_extra_state()==dict(version=2,architecture='rgb-two-level-padded-bilinear-v1',width=8,
        alpha_policy='require-opaque',normalization='uint8-div255',outputs=['nuclear-semantic'],attention='vanilla',input_policy='native-max1000-v1')
    env=RasterSourceEnvelope('a'*64,512,512,297)
    new=ColorNuclearSemantic(8,alpha_policy='require-opaque',input_policy='source-envelope-v2',source_envelope=env,
        parameterization_policy='mup-color-v1',base_width=8)
    assert primitive(new.get_extra_state())
