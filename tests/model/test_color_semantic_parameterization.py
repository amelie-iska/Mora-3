"""Native semantic inventory is opt-in metadata, never scaling/admission."""
import copy
import hashlib
import json
import numpy as np
import pytest
import torch
from torch import nn
from mora3.model import parameterization as p
from mora3.model.color_semantic import ColorNuclearSemantic
from mora3.model.color_conditioning import ConditionedColorSemantic
from test_segmentation_shared import fixture
from mora3.data.segmentation_shared import project_native_geometry


def make(w=8, conditioned=False):
    m=ColorNuclearSemantic(w,alpha_policy='require-opaque')
    return ConditionedColorSemantic(m,id_width=4) if conditioned else m


def inventory(m,conditioned=False):
    name='conditioned_semantic_parameter_ledger' if conditioned else 'semantic_parameter_ledger'
    assert hasattr(p,name), 'missing native semantic inventory: '+name
    return getattr(p,name)(m,channel_axis=('R','G','B'),identifier_axis=('a','b','c','d') if conditioned else ('image',))


@pytest.mark.parametrize('conditioned',[False,True])
@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('w',[8,12])
def test_complete_counts_axes_and_numerical_nonmutation(tmp_path,conditioned,device,dtype,w):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(36301)
    m=make(w,conditioned).to(device=device,dtype=dtype).eval()
    r,pr,prov=fixture(tmp_path)
    x=project_native_geometry(r,pr,provenance=prov,requested_targets=('response',))
    pixels=np.random.default_rng(36301).integers(0,256,(7,9,3),dtype=np.uint8)
    def forward():return m(pixels,x['input_graph'],x['task']) if conditioned else m(pixels)
    out=forward();params=tuple(m.parameters())
    grads=torch.autograd.grad(out.square().mean(),params,allow_unused=True)
    state=copy.deepcopy(m.state_dict());modes=[a.training for a in m.modules()]
    rng=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone() if device=='cuda' else None
    ledger=inventory(m,conditioned)
    expected=9*(3*w+51*w*w)+20*w+w*(w+7)+6*w*w+10*w+2*w+1
    if conditioned:expected+=8*w+517*w+w*(w+1)+w+w+1
    assert ledger.total_numel==expected==sum(t.numel() for t in params)
    rows={a:e for e in ledger.parameters for a in e.aliases}
    assert set(rows)==set(dict(m.named_parameters(remove_duplicate=False)))
    prefix='color.' if conditioned else ''
    axis=rows[prefix+'head.weight'].axes[0]
    digest=hashlib.sha256(json.dumps(['semantic.output.v1',('nuclear-foreground-logit',)],separators=(',',':')).encode()).hexdigest()
    assert axis==p.Axis('finite',digest)
    assert rows[prefix+'head.bias'].axes==(axis,)
    assert rows[prefix+'head.weight'].axes[1]==p.Axis('width','image.concat-local-context')
    if conditioned:
        assert rows['offset.weight'].axes==(axis,p.Axis('width','hidden'))
        assert rows['color.backbone.feature.weight'] is rows['context.graph.feature.weight']
    assert torch.equal(rng,torch.get_rng_state())
    if gpu is not None:assert torch.equal(gpu,torch.cuda.get_rng_state())
    after=forward();torch.testing.assert_close(out,after,rtol=0,atol=0)
    g2=torch.autograd.grad(after.square().mean(),params,allow_unused=True)
    for a,b in zip(grads,g2):
        assert (a is None)==(b is None)
        if a is not None:
            # Baseline CUDA bilinear/conv backward is not bitwise repeatable;
            # keep CPU exact and compare CUDA with explicit floating-point tolerances.
            rtol,atol=(0,0) if device=='cpu' else ((1e-5,1e-8) if dtype==torch.float32 else (1e-10,1e-12))
            torch.testing.assert_close(a,b,rtol=rtol,atol=atol)
    for k,v in m.state_dict().items():
        if isinstance(v,torch.Tensor):assert torch.equal(v,state[k])
        else:assert v==state[k]
    assert modes==[a.training for a in m.modules()]
    assert all(t.grad is None for t in params)
    with pytest.raises(ValueError,match='custom'):p.width_family(ledger,ledger,ledger)


@pytest.mark.parametrize('conditioned',[False,True])
@pytest.mark.parametrize('drift',['width','alpha','policy','extra','root-param','root-buffer','head-size',
    'head-shape','head-bias','head-child','head-buffer','head-subclass','root-subclass','tie','encoder'])
def test_native_ownership_drift_refuses(conditioned,drift):
    m=make(conditioned=conditioned);color=m.color if conditioned else m
    if drift=='width':color.width=10
    elif drift=='alpha':color.alpha_policy='unknown'
    elif drift=='policy':color.input_policy='unknown'
    elif drift=='extra':m.extra=nn.Identity()
    elif drift=='root-param':m.extra=nn.Parameter(torch.ones(1))
    elif drift=='root-buffer':m.register_buffer('extra',torch.ones(1))
    elif drift=='head-size':color.head.out_channels=2
    elif drift=='head-shape':color.head.weight=nn.Parameter(torch.zeros(1,15,1,1))
    elif drift=='head-bias':color.head.bias=None
    elif drift=='head-child':color.head.extra=nn.Identity()
    elif drift=='head-buffer':color.head.register_buffer('extra',torch.ones(1))
    elif drift=='head-subclass':
        class Other(nn.Conv2d):pass
        color.head.__class__=Other
    elif drift=='root-subclass':
        class Other(type(m)):pass
        m.__class__=Other
    elif drift=='tie':color.head.bias=color.backbone.graph_token
    else:color.pixels.enc0[0].in_channels=1
    name='conditioned_semantic_parameter_ledger' if conditioned else 'semantic_parameter_ledger'
    assert hasattr(p,name),'missing native semantic inventory'
    with pytest.raises(ValueError):inventory(m,conditioned)


@pytest.mark.parametrize('drift',['shared','config','context-config','payload','offset','rank','context-buffer'])
def test_conditioned_configuration_refuses(drift):
    m=make(conditioned=True)
    if drift=='shared':m.context.graph.feature=nn.Linear(8,8)
    elif drift=='config':m.config['training_admission']=True
    elif drift=='context-config':m.context.config['width']=12
    elif drift=='payload':m.context.payload.pooling='bag'
    elif drift=='offset':m.offset=nn.Linear(8,2)
    elif drift=='rank':m.config['id_width']=2
    else:m.context.register_buffer('extra',torch.ones(1))
    assert hasattr(p,'conditioned_semantic_parameter_ledger'),'missing native semantic inventory'
    with pytest.raises(ValueError):inventory(m,True)


def test_compatible_cross_owner_aliases_and_input_policies():
    a=make();b=make();b.pixels=a.pixels;b.backbone=a.backbone
    b.input_policy='native-max1000-v1'
    root=nn.ModuleDict({'a':a,'b':b})
    la=inventory(a);lb=inventory(b)
    result=p.compose_owner_parameter_ledgers(root,{'a':la,'b':lb})
    assert result.total_numel==la.total_numel+17
    assert next(e for e in result.parameters if 'a.pixels.enc0.0.weight' in e.aliases).aliases==('a.pixels.enc0.0.weight','b.pixels.enc0.0.weight')


def test_rgb_and_single_image_identifier_are_explicit():
    m=make();inventory(m)
    for channels,ids in [(('R','G'),('image',)),(('R','G','B'),('a','b'))]:
        with pytest.raises(ValueError):p.semantic_parameter_ledger(m,channel_axis=channels,identifier_axis=ids)
