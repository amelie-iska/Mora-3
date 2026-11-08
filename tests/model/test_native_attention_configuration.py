"""Explicit native policy declarations; fixtures establish no downstream quality."""
import copy
import pytest
import torch
from mora3.model.tokengt import TokenGT, EncoderLayer, DenseAttention
from mora3.model.parameterization import tokengt_parameter_ledger, width_family


def make(**kw):
    assert hasattr(TokenGT, 'configuration'), 'missing native attention configuration'
    args = dict(width=8, heads=2, layers=2, ffn_width=12)
    args.update(kw)
    return TokenGT(3, 2, **args)


def ledger(m):
    return tokengt_parameter_ledger(m, feature_axis=('a','b','c'), identifier_axis=('x','y'))


@pytest.mark.parametrize('owner', ['native','layer'])
@pytest.mark.parametrize('bad', [dict(attention='tropical-code', scaling_policy='mup-v1'),
    dict(attention='tropical-blockwise', attn_mult=2), dict(scaling_policy='future'),
    dict(attn_mult=True),dict(attn_mult=float('nan')),dict(attn_mult=0),
    dict(heads=3),dict(heads=True),dict(width=8.0),dict(attention='unknown')])
def test_invalid_policy_precedes_initialization(owner,bad):
    assert hasattr(TokenGT,'configuration'), 'missing native attention configuration'
    rng=torch.random.get_rng_state().clone()
    with pytest.raises(ValueError):
        if owner=='native': make(**bad)
        else:
            args=dict(width=8,heads=2,ffn_width=12);args.update(bad)
            EncoderLayer(**args)
    assert torch.equal(rng,torch.random.get_rng_state())


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('width',[8,12,16])
def test_native_reconstruction_and_independent_layer_equations(device,dtype,width):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(36901)
    m=make(width=width,scaling_policy='mup-v1',attn_mult=2.25).to(device=device,dtype=dtype)
    x=torch.randn(2,4,width,device=device,dtype=dtype).requires_grad_()
    valid=torch.tensor([[1,1,0,0],[1,1,1,1]],device=device,dtype=torch.bool)
    # Independent attention projections/head split inside the native residual layer.
    layer=m.layers[0]; z=layer.norm1(x)
    q,k,v=[z@w.T+b for w,b in zip(layer.attention.qkv.weight.chunk(3),layer.attention.qkv.bias.chunk(3))]
    d=width//2; parts=[]
    for h in range(2):
        sl=slice(h*d,(h+1)*d)
        logits=(q[:,:,sl]*(1.5/d))@k[:,:,sl].transpose(1,2)
        parts.append(logits.masked_fill(~valid[:,None,:],-torch.inf).softmax(-1)@v[:,:,sl])
    residual=x+torch.cat(parts,-1)@layer.attention.output.weight.T+layer.attention.output.bias
    expected=residual+layer.ffn(layer.norm2(residual)); actual=layer(x,valid)
    tol=3e-5 if dtype==torch.float32 else 3e-12
    torch.testing.assert_close(actual,expected,atol=tol,rtol=tol)
    variables=(x,*layer.parameters());probe=torch.linspace(-.5,.5,actual.numel(),device=device,dtype=dtype).reshape_as(actual)
    ga=torch.autograd.grad((actual*probe).sum(),variables,retain_graph=True)
    ge=torch.autograd.grad((expected*probe).sum(),variables)
    for a,e in zip(ga,ge):torch.testing.assert_close(a,e,atol=tol,rtol=tol)
    cfg=m.configuration(); saved=copy.deepcopy(m.state_dict())
    rng=torch.random.get_rng_state().clone(); ledger(m)
    assert torch.equal(rng,torch.random.get_rng_state())
    restored=TokenGT.from_configuration(cfg).to(device=device,dtype=dtype)
    restored.load_state_dict(saved,strict=True)
    graph=dict(features=torch.randn(3,3,device=device,dtype=dtype),
        endpoints=torch.tensor([[0,0],[1,1],[0,1]],device=device),
        roles=torch.tensor([0,0,1],device=device),identifiers=torch.eye(2,device=device,dtype=dtype))
    a,mask=m([graph]);b,mask2=restored([graph])
    assert torch.equal(mask,mask2);torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert all(torch.equal(saved[k],v) for k,v in m.state_dict().items())
    with pytest.raises(ValueError,match='custom'):width_family(ledger(m),ledger(m),ledger(m))


@pytest.mark.parametrize('drift',['policy','multiplier','heads','root','replace'])
def test_ledger_rejects_native_policy_drift(drift):
    m=make();a=m.layers[1].attention
    if drift=='policy':a.scaling_policy='mup-v1'
    if drift=='multiplier':a.attn_mult=2
    if drift=='heads':a.heads=1;a.head_width=8
    if drift=='root':m.layers[0].attention.scaling_policy='mup-v1'
    if drift=='replace':m.layers[1].attention=DenseAttention(8,2,scaling_policy='mup-v1')
    rng=torch.random.get_rng_state().clone()
    with pytest.raises(ValueError):ledger(m)
    assert torch.equal(rng,torch.random.get_rng_state())


@pytest.mark.parametrize('attention',['vanilla','tropical-code','tropical-blockwise'])
def test_legacy_reconstruction_and_fresh_config(attention):
    m=make(attention=attention)
    cfg=m.configuration();assert cfg['scaling_policy']=='legacy' and cfg['attn_mult']==1
    n=TokenGT.from_configuration(cfg);n.load_state_dict(m.state_dict(),strict=True)
    assert n.configuration()==cfg and set(n.state_dict())==set(m.state_dict())
    cfg['heads']=999
    assert m.configuration()['heads']==2


@pytest.mark.parametrize('bad',['schema','extra','missing','layers','multiplier'])
def test_exact_config_reconstruction_rejects_before_rng(bad):
    cfg=make().configuration()
    if bad=='schema':cfg['schema']='future'
    if bad=='extra':cfg['unknown']=1
    if bad=='missing':del cfg['scaling_policy']
    if bad=='layers':cfg['layers']=True
    if bad=='multiplier':cfg['attn_mult']=2
    rng=torch.random.get_rng_state().clone()
    with pytest.raises(ValueError):TokenGT.from_configuration(cfg)
    assert torch.equal(rng,torch.random.get_rng_state())


def test_separately_owned_geometry_shared_layers_remain_valid_inventory():
    from mora3.model.geometry_conditioning import GeometryConditioner
    parent=make(layers=2)
    child=GeometryConditioner(parent,id_width=16).graph
    # The adapter has its own finite incidence rank, and shares full parent layers.
    assert child.layers is parent.layers
    before=copy.deepcopy(child.state_dict());rng=torch.random.get_rng_state().clone()
    result=tokengt_parameter_ledger(child,feature_axis=('a','b','c'),
                                  identifier_axis=tuple('id:'+str(i) for i in range(16)))
    assert result.total_numel==sum(p.numel() for p in child.parameters())
    assert torch.equal(rng,torch.random.get_rng_state())
    assert all(torch.equal(before[k],v) for k,v in child.state_dict().items())
    # Its own wrapper contract must reconstruct this composition; root config cannot.
    with pytest.raises(ValueError):child.configuration()


def test_mutated_boolean_multiplier_is_not_numeric_policy():
    m=make();m.layers[0].attention.attn_mult=True
    with pytest.raises(ValueError):ledger(m)
