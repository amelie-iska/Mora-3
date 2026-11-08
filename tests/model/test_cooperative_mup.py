"""Explicit native cooperative policy propagation, no image rule relabeling."""
import copy
import pytest
import torch
from mora3.model.cooperative_training import CooperativeModel,GeometryCooperativeModel
from mora3.model.flow_unified import FlowUnifiedModel
from mora3.model.annotation_multihead import SharedAnnotationModel
from test_annotation_mup import vocab


def make(w=8,geometry=False):
    cls=GeometryCooperativeModel if geometry else CooperativeModel
    extra=dict(id_width=4) if geometry else {}
    return cls(('gene-a','gene-b'),width=w,parameterization_policy='mup-cooperative-v1',
        base_width=8,readout_mult=1.5,attn_mult=.5,encoder_var=.7,**extra)

def annotation(native):
    return SharedAnnotationModel(vocab(),('GO:0000001',),('1.1.1.1',),width=native.response.config['width'],
        heads=2,layers=2,max_tokens=512,parameterization_policy='mup-annotation-v1',base_width=8,
        readout_mult=1.5,attn_mult=.5,encoder_var=.7,backbone=native.response.backbone)


@pytest.mark.parametrize('geometry',[False,True])
def test_explicit_native_initialization_readout_policies_and_exact_sharing(geometry):
    m=make(12,geometry);cfg=m.get_extra_state()
    assert cfg['version']==3 and cfg['mup']['policy']=='mup-cooperative-v1'
    assert cfg['response']['version']==4
    assert m.image.backbone is m.response.backbone
    assert cfg['mup']['image_operator_qualification']=='applied-mup-image-v1'
    assert cfg['response']['mup']['base_width']==8
    assert cfg['response']['native_backbone']['initialization_policy']=='mup-transformer-v1'
    before={n:p.clone() for n,p in m.response.backbone.named_parameters()}
    a=annotation(m)
    assert a.backbone is m.response.backbone
    for n,p in m.response.backbone.named_parameters():assert torch.equal(p,before[n])


@pytest.mark.parametrize('version',[1,2])
def test_new_flow_owner_schema_policy_aliases_and_no_reinitialization(version):
    native=make(12);a=annotation(native)
    before={n:p.clone() for n,p in native.response.named_parameters()}
    model=FlowUnifiedModel(native,a,flow_version=version,sigma_min=.1,
                          parameterization_policy='mup-cooperative-v1')
    assert model.flow.get_extra_state()['version']==version+2
    assert model.get_extra_state()['version']==2
    for n,p in native.response.named_parameters():assert torch.equal(p,before[n])
    assert model.flow.native is model.native.response
    snap=copy.deepcopy(model.state_dict());rng=torch.get_rng_state().clone()
    model.load_state_dict(snap)
    assert torch.equal(rng,torch.get_rng_state())
    # A nested wrong readout recipe must refuse before ANY shared tensor copies.
    bad=copy.deepcopy(snap);bad['flow._extra_state']['mup']['base_width']=4
    bad['native.response.edge'].fill_(999)
    tensors={k:v.clone() for k,v in model.state_dict().items() if isinstance(v,torch.Tensor)}
    with pytest.raises(ValueError,match='configuration|policy'):model.load_state_dict(bad)
    for k,v in tensors.items():assert torch.equal(v,model.state_dict()[k])


def test_legacy_arithmetic_configuration_and_rng_are_untouched():
    torch.manual_seed(2026110664);a=CooperativeModel(('gene-a','gene-b'),width=8);rng=torch.get_rng_state().clone()
    torch.manual_seed(2026110664);b=CooperativeModel(('gene-a','gene-b'),width=8,parameterization_policy='legacy')
    assert torch.equal(rng,torch.get_rng_state())
    assert a.get_extra_state()==b.get_extra_state() and a.get_extra_state()['version']==1
    assert set(a.state_dict())==set(b.state_dict())
    for (n,p),(name,z) in zip(a.named_parameters(),b.named_parameters()):
        assert n==name and torch.equal(p,z)


@pytest.mark.parametrize('change',['policy','base','readout','annotation'])
def test_policy_mismatch_refuses_before_adding_random_flow_parameters(change):
    native=make(12);a=annotation(native)
    if change=='policy':kwargs={}
    else:
        kwargs=dict(parameterization_policy='mup-cooperative-v1')
        if change=='base':a._configuration['mup']['base_width']=4
        elif change=='readout':a._configuration['mup']['output_mult']=2.
        else:a._configuration['version']=1
    rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):FlowUnifiedModel(native,a,**kwargs)
    assert torch.equal(rng,torch.get_rng_state())
