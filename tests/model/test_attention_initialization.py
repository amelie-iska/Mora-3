"""Fresh-only packed attention initialization, independently pinned to upstream."""
import copy
import math

import pytest
import torch
from torch import nn

from mora3.model.tokengt import DenseAttention, TokenGT


def fresh(width=8,heads=2,var=1.):
    return DenseAttention(width,heads,scaling_policy='mup-v1',attn_mult=.5,
        initialization_policy='mup-transformer-v1',encoder_var=var)


@pytest.mark.parametrize('width,heads,var',[(8,2,1.),(12,3,.5),(16,2,2.)])
def test_exact_reference_packed_slices_values_and_rng_draw_order(width,heads,var):
    torch.manual_seed(2026110611)
    qkv,output=nn.Linear(width,3*width),nn.Linear(width,width)
    # Pinned upstream Transformer/model.py291–309, after ordinary Linear init.
    with torch.no_grad():
        nn.init.normal_(qkv.weight,mean=0.,std=math.sqrt(var/width))
        qkv.weight[:width].zero_()
        nn.init.normal_(output.weight,mean=0.,std=math.sqrt(var/width))
        qkv.bias.zero_();output.bias.zero_()
    expected_rng=torch.get_rng_state().clone()
    torch.manual_seed(2026110611);m=fresh(width,heads,var)
    assert torch.equal(expected_rng,torch.get_rng_state())
    for actual,expected in [(m.qkv.weight,qkv.weight),(m.qkv.bias,qkv.bias),
                            (m.output.weight,output.weight),(m.output.bias,output.bias)]:
        assert torch.equal(actual,expected) and actual.grad is None
    assert not torch.count_nonzero(m.qkv.weight[:width])
    assert torch.count_nonzero(m.qkv.weight[width:])==2*width*width
    cfg=m.scaling_config()
    assert cfg['schema']=='dense-attention-scaling-v2'
    assert cfg['initialization']['slices']==(('query',0,width),('key',width,2*width),
                                            ('value',2*width,3*width))


def test_legacy_default_and_existing_mup_scaling_schema_rng_and_state_exact():
    for scaling in ('legacy','mup-v1'):
        torch.manual_seed(2026110612)
        a=DenseAttention(8,2,scaling_policy=scaling)
        arng=torch.get_rng_state().clone()
        torch.manual_seed(2026110612)
        b=DenseAttention(8,2,scaling_policy=scaling,initialization_policy='legacy',encoder_var=1.)
        assert torch.equal(arng,torch.get_rng_state())
        assert a.scaling_config()==b.scaling_config()==dict(schema='dense-attention-scaling-v1',
            width=8,heads=2,scaling_policy=scaling,attn_mult=1.)
        assert set(a.state_dict())==set(b.state_dict())=={
            'qkv.weight','qkv.bias','output.weight','output.bias'}
        for p,q in zip(a.parameters(),b.parameters()):assert torch.equal(p,q)
        assert torch.count_nonzero(a.qkv.weight[:8])==64


@pytest.mark.parametrize('kwargs',[
    {'initialization_policy':'unknown'},{'encoder_var':0.},{'encoder_var':float('nan')},
    {'encoder_var':float('inf')},{'encoder_var':True},{'scaling_policy':'legacy'},
    {'initialization_policy':'legacy','encoder_var':2.},
])
def test_invalid_policy_precedes_rng_or_parameter_initialization(kwargs):
    kw=dict(scaling_policy='mup-v1',initialization_policy='mup-transformer-v1',encoder_var=1.)
    kw.update(kwargs);rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):DenseAttention(8,2,**kw)
    assert torch.equal(rng,torch.get_rng_state())


def test_reconstruct_then_load_preserves_learned_query_and_never_reinitializes_existing_weights():
    m=fresh()
    with torch.no_grad():
        m.qkv.weight.add_(3.)
        m.qkv.bias.add_(.2)
        m.output.weight.mul_(.7)
    state=copy.deepcopy(m.state_dict());cfg=m.scaling_config()
    assert '_attention_initialization' in state
    new=DenseAttention.from_scaling_config(cfg)
    assert not torch.count_nonzero(new.qkv.weight[:8])
    rng=torch.get_rng_state().clone();new.load_state_dict(state,strict=True)
    assert torch.equal(rng,torch.get_rng_state())
    assert torch.count_nonzero(new.qkv.weight[:8])==64
    for a,b in zip(m.parameters(),new.parameters()):assert torch.equal(a,b)
    assert state['_attention_initialization']==cfg['initialization']
    cfg['initialization']['encoder_var']=9.
    assert m.scaling_config()['initialization']['encoder_var']==1.


@pytest.mark.parametrize('case',['legacy-to-new','new-to-legacy','wrong-var','wrong-slices',
                               'missing','unknown-field','bool-var'])
def test_checkpoint_policy_mismatch_refuses_before_child_tensor_copy(case):
    m=fresh();checkpoint=copy.deepcopy(m.state_dict());target=fresh()
    if case=='legacy-to-new':checkpoint=DenseAttention(8,2,scaling_policy='mup-v1').state_dict()
    if case=='new-to-legacy':target=DenseAttention(8,2,scaling_policy='mup-v1')
    if case=='wrong-var':target=fresh(var=2.)
    if case=='wrong-slices':checkpoint['_attention_initialization']['slices']=(('key',0,8),)
    if case=='missing':del checkpoint['_attention_initialization']
    if case=='unknown-field':checkpoint['_attention_initialization']['extra']=0
    if case=='bool-var':checkpoint['_attention_initialization']['encoder_var']=True
    checkpoint['qkv.weight']=torch.full_like(checkpoint['qkv.weight'],7.)
    before=[p.detach().clone() for p in target.parameters()];rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):target.load_state_dict(checkpoint,strict=True)
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(a,b) and b.grad is None for a,b in zip(before,target.parameters()))


@pytest.mark.parametrize('case',['schema','var','slices','policy','extra','width'])
def test_config_reconstruction_mismatch_precedes_rng(case):
    cfg=fresh().scaling_config()
    if case=='schema':cfg['schema']='dense-attention-scaling-v1'
    if case=='var':cfg['initialization']['encoder_var']=False
    if case=='slices':cfg['initialization']['slices']=(('query',0,7),)
    if case=='policy':cfg['initialization']['policy']='unknown'
    if case=='extra':cfg['initialization']['extra']=1
    if case=='width':cfg['width']=12
    rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):DenseAttention.from_scaling_config(cfg)
    assert torch.equal(rng,torch.get_rng_state())


def test_old_native_owner_refuses_new_initializer_schema_instead_of_silent_adoption():
    native=TokenGT(8,2,width=8,heads=2,layers=2,scaling_policy='mup-v1',attn_mult=.5)
    native.layers[0].attention=fresh()
    rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):native.configuration()
    assert torch.equal(rng,torch.get_rng_state())


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_initialized_attention_output_and_all_gradients_match_explicit_equations(device,dtype):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(2026110613);m=fresh().to(device=device,dtype=dtype)
    x=torch.linspace(-1,1,24,device=device,dtype=dtype).reshape(1,3,8).requires_grad_()
    valid=torch.tensor([[True,True,False]],device=device)
    actual=m(x,valid)
    q,k,v=torch.nn.functional.linear(x,m.qkv.weight,m.qkv.bias).reshape(1,3,3,2,4).unbind(2)
    q,k,v=(t.transpose(1,2) for t in (q,k,v))
    score=(q@k.transpose(-1,-2))*math.sqrt(.5)/4
    a=score.masked_fill(~valid[:,None,None,:],-torch.inf).softmax(-1)
    expected=torch.nn.functional.linear((a@v).transpose(1,2).reshape(1,3,8),m.output.weight,m.output.bias)
    tol=1e-6 if dtype==torch.float32 else 1e-12
    torch.testing.assert_close(actual,expected,rtol=tol,atol=tol)
    owners=(x,*m.parameters())
    ga=torch.autograd.grad(actual.square().sum(),owners)
    gb=torch.autograd.grad(expected.square().sum(),owners)
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b,rtol=tol,atol=tol)
