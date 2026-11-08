"""Versioned opt-in owner, independently checked tiny equations; no training."""
import copy
import math

import pytest
import torch
from torch.nn import functional as F

from mora3.model.sciplex import SciPlexSharedModel
from mora3.model.tokengt import TokenGT
from mora3.model.response_flow import ResponseFlowHead, ResponseFlowDenoiserV2
from test_sciplex_shared import task


def native(**kw):
    return SciPlexSharedModel(('A','B'),width=8,scaling_policy='mup-v1',attn_mult=.5,**kw)


def test_versioned_configuration_roundtrip_and_legacy_tensor_initialization():
    torch.manual_seed(2026110601);legacy=SciPlexSharedModel(('A','B'),width=8)
    torch.manual_seed(2026110601);m=native()
    assert legacy.config['version']==1 and 'native_backbone' not in legacy.config
    assert m.config['version']==2
    assert m.config['native_backbone']==m.backbone.configuration()
    assert m.config['native_backbone']['scaling_policy']=='mup-v1'
    assert m.config['native_backbone']['attn_mult']==.5
    a,b=legacy.state_dict(),m.state_dict();assert a.keys()==b.keys()
    for k,v in a.items():
        if isinstance(v,torch.Tensor):assert torch.equal(v,b[k])
    for original in (legacy,m):
        fresh=SciPlexSharedModel.from_configuration(original.get_extra_state())
        fresh.load_state_dict(original.state_dict(),strict=True)
        assert torch.equal(original([task()]),fresh([task()]))
        assert fresh.config==original.config
    exported=m.get_extra_state();exported['native_backbone']['attn_mult']=3.
    assert m.config['native_backbone']['attn_mult']==.5


@pytest.mark.parametrize('kw',[
    {'scaling_policy':'unknown'},{'scaling_policy':'mup-v1','attn_mult':float('nan')},
    {'scaling_policy':'mup-v1','attn_mult':True},{'scaling_policy':'legacy','attn_mult':2.},
    {'scaling_policy':'mup-v1','attention':'tropical-code'},
    {'scaling_policy':'mup-v1','attention':'tropical-blockwise'},
    {'scaling_policy':'mup-v1','width':7},{'scaling_policy':'mup-v1','width':False},
])
def test_invalid_constructor_refuses_before_initialization(kw):
    kwargs=dict(width=8);kwargs.update(kw)
    rng=torch.random.get_rng_state().clone()
    with pytest.raises(ValueError):SciPlexSharedModel(('A','B'),**kwargs)
    assert torch.equal(rng,torch.random.get_rng_state())


@pytest.mark.parametrize('drift',['version-bool','version','missing-native','native-extra',
    'native-heads','native-width','native-policy','native-mult-bool','native-layer',
    'native-id','native-ffn','axis-list','policy','extra'])
def test_invalid_reconstruction_precedes_any_parameter_initialization(drift):
    m=native();config=m.get_extra_state()
    if drift=='version-bool':config['version']=True
    if drift=='version':config['version']=1
    if drift=='missing-native':del config['native_backbone']
    if drift=='native-extra':config['native_backbone']['unexpected']=1
    if drift=='native-heads':config['native_backbone']['heads']=4
    if drift=='native-width':config['native_backbone']['width']=12
    if drift=='native-policy':config['native_backbone']['scaling_policy']='legacy'
    if drift=='native-mult-bool':config['native_backbone']['attn_mult']=True
    if drift=='native-layer':config['native_backbone']['layers']=3
    if drift=='native-id':config['native_backbone']['id_width']=4
    if drift=='native-ffn':config['native_backbone']['ffn_width']=16
    if drift=='axis-list':config['axis_ids']=list(config['axis_ids'])
    if drift=='policy':config['policy']='different-source-policy'
    if drift=='extra':config['unexpected']=0
    rng=torch.random.get_rng_state().clone()
    with pytest.raises(ValueError):SciPlexSharedModel.from_configuration(config)
    assert torch.equal(rng,torch.random.get_rng_state())


@pytest.mark.parametrize('drift',['owner-policy','native-config','legacy-backbone','layer-policy'])
def test_opt_in_runtime_refuses_drift_without_tensor_or_rng_mutation(drift):
    m=native()
    if drift=='owner-policy':m.config['attention']='tropical-code'
    if drift=='native-config':m.config['native_backbone']['attn_mult']=2.
    if drift=='legacy-backbone':m.backbone=TokenGT(8,2,width=8,heads=2,layers=2)
    if drift=='layer-policy':m.backbone.layers[0].attention.attn_mult=2.
    before=[p.detach().clone() for p in m.parameters()];rng=torch.random.get_rng_state().clone()
    with pytest.raises(ValueError):m([task()])
    assert torch.equal(rng,torch.random.get_rng_state())
    assert all(torch.equal(a,b) and b.grad is None for a,b in zip(before,m.parameters()))


@pytest.mark.parametrize('cls',[ResponseFlowHead,ResponseFlowDenoiserV2])
def test_legacy_flow_schema_refuses_new_native_owner_before_initialization(cls):
    m=native();rng=torch.random.get_rng_state().clone()
    with pytest.raises(ValueError,match='version|legacy|schema'):cls(m)
    assert torch.equal(rng,torch.random.get_rng_state())


@pytest.mark.parametrize('cls',[ResponseFlowHead,ResponseFlowDenoiserV2])
def test_existing_flow_refuses_grafted_opt_in_native_without_rng_mutation(cls):
    flow=cls(SciPlexSharedModel(('A','B'),width=8));m=native()
    flow.native=m
    # Even coherent metadata grafting cannot opt a legacy flow schema in.
    flow.config['native']=m.get_extra_state()
    flow._owner_configuration=copy.deepcopy(flow.config)
    rng=torch.random.get_rng_state().clone()
    with pytest.raises(ValueError,match='version|legacy|schema'):flow.get_extra_state()
    assert torch.equal(rng,torch.random.get_rng_state())


def equation(model,t):
    """Independent complete dense computation using the same graph input contract."""
    graph=model.graph(t);backbone=model.backbone;w=model.edge.numel()
    x=(F.linear(graph['features'],backbone.feature.weight)
       +F.linear(graph['identifiers'][graph['endpoints']].flatten(1),backbone.incidence.weight)
       +F.embedding(graph['roles'],backbone.role.weight))
    x=torch.cat((backbone.graph_token[None],x))[None]
    for layer in backbone.layers:
        z=F.layer_norm(x,(w,),layer.norm1.weight,layer.norm1.bias,layer.norm1.eps)
        a=layer.attention;head=w//a.heads
        q,k,v=F.linear(z,a.qkv.weight,a.qkv.bias).reshape(1,4,3,a.heads,head).unbind(2)
        q,k,v=(value.transpose(1,2) for value in (q,k,v))
        scores=torch.einsum('bhqd,bhkd->bhqk',q,k)*math.sqrt(a.attn_mult)/head
        attended=torch.einsum('bhqk,bhkd->bhqd',scores.softmax(-1),v)
        x=x+F.linear(attended.transpose(1,2).reshape(1,4,w),a.output.weight,a.output.bias)
        z=F.layer_norm(x,(w,),layer.norm2.weight,layer.norm2.bias,layer.norm2.eps)
        x=x+F.linear(F.gelu(F.linear(z,layer.ffn[0].weight,layer.ffn[0].bias)),
                     layer.ffn[2].weight,layer.ffn[2].bias)
    latent=F.layer_norm(x,(w,),backbone.final_norm.weight,backbone.final_norm.bias,
                       backbone.final_norm.eps)[:,0]
    head=model.response_head if t.input_metadata()['task']=='response' else model.intervention_head
    return F.linear(latent,head.weight,head.bias)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('width',[8,12])
def test_independent_opt_in_equations_all_active_gradients_state_and_rng(device,dtype,width):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(2026110601)
    m=SciPlexSharedModel(('A','B'),width=width,scaling_policy='mup-v1',attn_mult=.5).to(
        device=device,dtype=dtype)
    assert sum(p.numel() for p in m.parameters())<3000
    t=task('response' if width==8 else 'intervention')
    before=[p.detach().clone() for p in m.parameters()];rng=torch.random.get_rng_state().clone()
    cuda_rng=torch.cuda.get_rng_state().clone() if device=='cuda' else None
    actual=m([t]);expected=equation(m,t)
    tol=2e-5 if dtype==torch.float32 else 1e-11
    torch.testing.assert_close(actual,expected,rtol=tol,atol=tol)
    pa=tuple(m.parameters())
    ga=torch.autograd.grad(actual.sum(),pa,allow_unused=True)
    gb=torch.autograd.grad(expected.sum(),pa,allow_unused=True)
    for a,b in zip(ga,gb):
        assert (a is None)==(b is None)
        if a is not None:torch.testing.assert_close(a,b,rtol=tol,atol=tol)
    assert torch.equal(rng,torch.random.get_rng_state())
    if cuda_rng is not None:assert torch.equal(cuda_rng,torch.cuda.get_rng_state())
    assert all(torch.equal(a,b) and b.grad is None for a,b in zip(before,m.parameters()))
