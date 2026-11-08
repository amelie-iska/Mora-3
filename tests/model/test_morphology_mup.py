"""Fixed-kernel native image μP, tiny synthetic raster/graph unit equations."""
import copy
import math
import pytest
import torch
from torch.nn import functional as F
from mora3.model.morphology import MultiscalePixels,MorphologyInstance
from mora3.model.morphology_mup import (pixel_mup_parameter_roles,morphology_mup_parameter_roles,
    morphology_mup_width_family,cooperative_mup_parameter_roles,cooperative_mup_width_family)
from mora3.model.cooperative_training import CooperativeModel,GeometryCooperativeModel
from mora3.model.mu_optimizer import MuAdamW
from test_morphology import task


@pytest.fixture(autouse=True)
def seed():torch.manual_seed(2026110665)


def image(w=8):return MorphologyInstance(w,parameterization_policy='mup-image-v1',base_width=8,output_mult=1.5)
def cooperative(w=8,geometry=False):
    return (GeometryCooperativeModel if geometry else CooperativeModel)(('a','b'),width=w,
        parameterization_policy='mup-cooperative-v1',base_width=8,readout_mult=1.5,
        **(dict(id_width=4) if geometry else {}))


def test_complete_fixed_kernel_initialization_and_rng():
    torch.manual_seed(2026110665);old=MultiscalePixels(12);rng=torch.get_rng_state().clone()
    torch.manual_seed(2026110665);new=MultiscalePixels(12,parameterization_policy='mup-image-v1',base_width=8)
    assert torch.equal(rng,torch.get_rng_state())
    for n,p in old.named_parameters():
        factor=math.sqrt(12/8) if n.endswith('bias') and n!='enc0.0.bias' else 1
        assert torch.equal(dict(new.named_parameters())[n],p*factor)
    roles=pixel_mup_parameter_roles(new,channel_axis=('intensity',))
    assert roles['enc0.0.weight'].role=='input'
    assert all(r.role!='custom' for r in roles.values())
    assert roles['dec1.0.weight'].role=='hidden'
    assert tuple(a.kind for a in roles['dec1.0.weight'].axes)==('width','width','finite','finite')


def test_full_resolution_readout_equation_all_gradients_and_target_exclusion():
    m=image(16).double();t=task()
    x=torch.arange(30,dtype=torch.float64).reshape(1,1,5,6)/255
    local=m.pixels(x)
    graph=dict(features=local.mean((2,3)),endpoints=torch.zeros((1,2),dtype=torch.long),
        roles=torch.zeros(1,dtype=torch.long),identifiers=torch.ones(1,1,dtype=torch.float64))
    latent=m.backbone([graph])[0][:,0]
    context=latent[:,:,None,None].expand(-1,-1,5,6)
    expected=F.conv2d(torch.cat((local,context),1)*(1.5/(16/8)),m.head.weight,m.head.bias)[0]
    actual=m(t)
    torch.testing.assert_close(actual,expected,rtol=1e-11,atol=1e-11)
    owners=tuple(m.parameters())
    for a,b in zip(torch.autograd.grad(actual.sum(),owners,retain_graph=True),torch.autograd.grad(expected.sum(),owners)):
        torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-10)
    from dataclasses import replace
    from mora3.data.livecell_graph import LiveCellGraphTask
    changed=LiveCellGraphTask(replace(t.example,polygons=()),t.direction)
    assert torch.equal(actual,m(changed))


@pytest.mark.parametrize('geometry',[False,True])
def test_complete_native_cooperative_family_aliases_and_optimizer_no_updates(geometry):
    ms=[cooperative(w,geometry) for w in (8,12,16)]
    kwargs=dict(geometry_identifier_axis=('g0','g1','g2','g3')) if geometry else {}
    family=cooperative_mup_width_family(*ms,**kwargs)
    for member,m in zip(('base','delta','target'),ms):
        roles=cooperative_mup_parameter_roles(m,**kwargs)
        assert not any(r.role=='custom' for r in roles.values())
        opt=MuAdamW(m,family,roles,base_lr=.001,weight_decay=0,decay_aliases=(),member=member)
        assert not opt.state
        assert m.image.backbone is m.response.backbone
        assert m.get_extra_state()['version']==3
    assert family.total_numel==tuple(sum(p.numel() for p in m.parameters()) for m in ms)


def test_standalone_image_family_and_conditional_pixel_checkpoint_policy():
    ms=[image(w) for w in (8,12,16)]
    family=morphology_mup_width_family(*ms,identifier_axis=('image',))
    assert not any(r.role=='custom' for r in morphology_mup_parameter_roles(ms[-1],identifier_axis=('image',)).values())
    assert family.total_numel==tuple(sum(p.numel() for p in m.parameters()) for m in ms)
    assert 'pixels._pixel_parameterization' in ms[-1].state_dict()
    assert '_pixel_parameterization' not in MultiscalePixels(8).state_dict()
    model=ms[-1];state=copy.deepcopy(model.state_dict());rng=torch.get_rng_state().clone()
    model.load_state_dict(state);assert torch.equal(rng,torch.get_rng_state())
    state['pixels._pixel_parameterization']['base_width']=4
    state['head.weight'].fill_(99)
    before={n:p.clone() for n,p in model.named_parameters()}
    with pytest.raises(ValueError,match='configuration|policy'):model.load_state_dict(state)
    for n,p in model.named_parameters():assert torch.equal(p,before[n])


def test_new_geometry_native_graph_configuration_forward_and_split_payload():
    m=cooperative(12,True);g=m.fusion.geometry
    assert g.graph.layers is m.response.backbone.layers
    assert g.graph.configuration()['heads']==2 and g.graph.configuration()['layers']==2
    assert g.config['version']==2
    assert g.payload.projection.__class__.__name__=='MixedPayloadProjection'
    # Provided input graph only; no frozen checkpoint, source masks or teacher job.
    from mora3.data.schema import GraphRecord,Node,Measurement
    from mora3.data.provenance import Provenance
    from mora3.data.tasks import ConditionalTask
    record=GraphRecord('request', (Node('object',(Measurement('x',.5,'um','observed',True),
        Measurement('target',99.,'um','observed',True))),),(),
        (Provenance('synthetic-unit','v1','request','a'*64,'no-rights-inferred'),))
    query=ConditionalTask(((('node',0,'x'),'condition'),(('node',0,'target'),'target')),topology_known=True)
    out=g(record,query)
    assert out.shape==(1,12) and torch.isfinite(out).all()
    out.sum().backward()
    assert g.payload.projection.length_weight.grad is not None
    assert m.response.backbone.layers[0].attention.qkv.weight.grad is not None


def test_legacy_instance_state_and_rng_exact():
    torch.manual_seed(2026110666);a=MorphologyInstance(8);rng=torch.get_rng_state().clone()
    torch.manual_seed(2026110666);b=MorphologyInstance(8,parameterization_policy='legacy')
    assert torch.equal(rng,torch.get_rng_state())
    assert a.get_extra_state()==b.get_extra_state()
    assert set(a.state_dict())==set(b.state_dict())
    for k,v in a.state_dict().items():
        if isinstance(v,torch.Tensor):assert torch.equal(v,b.state_dict()[k])


@pytest.mark.parametrize('change',['kernel','concat','groups','base','head','native','axis'])
def test_complete_family_refuses_operator_or_finite_axis_drift(change):
    ms=[cooperative(w) for w in (8,12,16)];m=ms[-1]
    if change=='kernel':m.image.pixels.dec1[0].dilation=(2,2)
    elif change=='concat':m.image.pixels.dec0[0].in_channels=2*m.image.width
    elif change=='groups':m.image.pixels.enc0[2].groups=2
    elif change=='base':m.image.pixels.base_width=4
    elif change=='head':m.image.head.out_channels=4
    elif change=='native':m.image.backbone=copy.deepcopy(m.response.backbone)
    else:m.response.axis_ids=('b','a')
    with pytest.raises(ValueError):cooperative_mup_width_family(*ms)


def test_runtime_image_operator_drift_refuses_before_raster_forward():
    m=image(12);m.pixels.enc0[2].dilation=(2,2)
    with pytest.raises(ValueError,match='configuration|convolution'):
        m(task())


def test_nested_pixel_policy_checkpoint_refuses_before_shared_tensors():
    m=cooperative(12,True);state=copy.deepcopy(m.state_dict())
    state['fusion.image.pixels._pixel_parameterization']['base_width']=4
    state['response.edge'].fill_(99)
    before={n:p.clone() for n,p in m.named_parameters()}
    with pytest.raises(ValueError,match='configuration|policy'):m.load_state_dict(state)
    for n,p in m.named_parameters():assert torch.equal(p,before[n])
