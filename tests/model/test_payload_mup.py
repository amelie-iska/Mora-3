"""Source-specific mixed hidden/token-count μP projection, no biological data."""
import copy
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from mora3.tokenization.shortest_path import ByteTokenizer
from mora3.model.payload import PayloadEncoder
from mora3.model.mixed_payload_projection import MixedPayloadProjection
from mora3.model.payload_mup import payload_mup_parameter_roles,payload_mup_width_family
from mora3.model.mu_optimizer import MuAdamW


def tokenizer():return ByteTokenizer((),controls=('pad','node','edge'))


def make(width=8,**kw):
    args=dict(max_tokens=64,parameterization_policy='mup-payload-v1',base_width=8)
    args.update(kw)
    return PayloadEncoder(tokenizer(),width,**args)


def test_legacy_linear_default_rng_state_and_pool_values_unchanged():
    torch.manual_seed(2026110641);a=PayloadEncoder(tokenizer(),8,max_tokens=64)
    rng=torch.get_rng_state().clone()
    torch.manual_seed(2026110641);b=PayloadEncoder(tokenizer(),8,max_tokens=64,parameterization_policy='legacy')
    assert torch.equal(rng,torch.get_rng_state())
    assert type(a.projection) is nn.Linear and type(b.projection) is nn.Linear
    assert set(a.state_dict())=={'embedding.weight','projection.weight','projection.bias'}
    for p,q in zip(a.parameters(),b.parameters()):assert torch.equal(p,q)
    assert torch.equal(a.pool([65,66]),b.pool([65,66]))


@pytest.mark.parametrize('width',[8,12,16])
def test_explicit_fresh_mixed_block_variance_and_rng_recipe(width):
    torch.manual_seed(2026110642)
    embedding=nn.Embedding(517,width)
    base_bound=1/math.sqrt(9)
    hidden=torch.empty(width,width).uniform_(-base_bound*math.sqrt(8/width),base_bound*math.sqrt(8/width))
    length=torch.empty(width,1).uniform_(-base_bound,base_bound)
    bias=torch.empty(width).uniform_(-base_bound,base_bound)
    rng=torch.get_rng_state().clone()
    torch.manual_seed(2026110642);m=make(width)
    assert torch.equal(rng,torch.get_rng_state())
    assert torch.equal(m.embedding.weight,embedding.weight)
    for a,b in [(m.projection.hidden_weight,hidden),(m.projection.length_weight,length),(m.projection.bias,bias)]:
        assert torch.equal(a,b)
    assert m.projection.get_extra_state()['fixed_axis']=='log1p-complete-token-count'


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_split_projection_exact_math_and_all_input_parameter_gradients(device,dtype):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(2026110643)
    p=MixedPayloadProjection(12,base_width=8).to(device=device,dtype=dtype)
    x=torch.linspace(-1,1,26,device=device,dtype=dtype).reshape(2,13).requires_grad_()
    actual=p(x)
    expected=F.linear(x,torch.cat((p.hidden_weight,p.length_weight),dim=1),p.bias)
    tol=1e-6 if dtype==torch.float32 else 1e-12
    torch.testing.assert_close(actual,expected,rtol=tol,atol=tol)
    owners=(x,*p.parameters())
    ga=torch.autograd.grad(actual.square().sum(),owners)
    gb=torch.autograd.grad(expected.square().sum(),owners)
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b,rtol=tol,atol=tol)


def test_qualified_complete_payload_family_fixed_token_length_axes_and_optimizer_blocks():
    ms=[make(w).double() for w in (8,12,16)]
    family=payload_mup_width_family(*ms)
    assert family.total_numel==(4216,6372,8560)
    m=ms[-1];roles=payload_mup_parameter_roles(m)
    assert [a.kind for a in roles['projection.hidden_weight'].axes]==['width','width']
    assert [a.kind for a in roles['projection.length_weight'].axes]==['width','finite']
    rng=torch.get_rng_state().clone();before=[p.clone() for p in m.parameters()]
    opt=MuAdamW(m,family,roles,base_lr=.02,weight_decay=.1,
        decay_aliases=('embedding.weight','projection.hidden_weight','projection.length_weight'))
    by={g['mup_aliases'][0]:g for g in opt.param_groups}
    assert by['projection.hidden_weight']['lr']==.01
    assert by['projection.length_weight']['lr']==.02
    assert by['projection.bias']['lr']==.02
    assert not opt.state and torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(before,m.parameters()))


@pytest.mark.parametrize('failure',['tokens','controls','pooling','max-tokens','base','legacy','extra','tie'])
def test_payload_family_refuses_semantic_source_owner_and_alias_drift(failure):
    ms=[make(w).double() for w in (8,12,16)]
    if failure=='tokens':ms[1].codec.tokenizer._vocabulary=tuple(reversed(ms[1].codec.tokenizer.vocabulary))
    if failure=='controls':ms[1].codec.tokenizer._controls={'edge':256,'node':257,'pad':258}
    if failure=='pooling':ms[1].pooling='bag'
    if failure=='max-tokens':ms[1].max_tokens=63
    if failure=='base':ms[1]=make(12,base_width=6).double()
    if failure=='legacy':ms[1]=PayloadEncoder(tokenizer(),12,max_tokens=64).double()
    if failure=='extra':ms[1].extra=nn.Linear(12,12).double()
    if failure=='tie':ms[1].projection.length_weight=nn.Parameter(ms[1].projection.hidden_weight[:,:1])
    with pytest.raises(ValueError):payload_mup_width_family(*ms)


@pytest.mark.parametrize('failure',['base-bool','base-zero','base-too-large','legacy-base','unknown'])
def test_invalid_payload_policy_precedes_rng(failure):
    kw={}
    if failure=='base-bool':kw['base_width']=True
    if failure=='base-zero':kw['base_width']=0
    if failure=='base-too-large':kw['base_width']=12
    if failure=='legacy-base':kw['parameterization_policy']='legacy'
    if failure=='unknown':kw['parameterization_policy']='other'
    rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):make(**kw)
    assert torch.equal(rng,torch.get_rng_state())


@pytest.mark.parametrize('failure',['metadata','token-axis','projection','legacy-source','legacy-target'])
def test_payload_checkpoint_policy_preflight_before_embedding_copy(failure):
    m=make();state=copy.deepcopy(m.state_dict());fresh=make()
    if failure=='metadata':del state['_payload_parameterization']
    if failure=='token-axis':state['_payload_parameterization']['token_axis_sha256']='f'*64
    if failure=='projection':state['projection._extra_state']['base_width']=4
    if failure=='legacy-source':state=PayloadEncoder(tokenizer(),8,max_tokens=64).state_dict()
    if failure=='legacy-target':fresh=PayloadEncoder(tokenizer(),8,max_tokens=64)
    state['embedding.weight']=torch.full_like(state['embedding.weight'],9.)
    before=[p.clone() for p in fresh.parameters()];rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):fresh.load_state_dict(state,strict=True)
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(before,fresh.parameters()))


def test_payload_valid_strict_recovery_never_reinitializes_split_blocks():
    m=make(12).double()
    with torch.no_grad():
        for p in m.parameters():p.add_(.125)
    state=copy.deepcopy(m.state_dict());fresh=make(12).double()
    rng=torch.get_rng_state().clone();fresh.load_state_dict(state,strict=True)
    assert torch.equal(rng,torch.get_rng_state())
    for a,b in zip(m.parameters(),fresh.parameters()):assert torch.equal(a,b)
    assert torch.equal(m.pool([65,66,67]),fresh.pool([65,66,67]))
