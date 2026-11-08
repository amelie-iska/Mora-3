"""Applied native response μP family, not complete Mora3 or data admission."""
import copy
import dataclasses
import math

import pytest
import torch
from torch.nn import functional as F

from mora3.model.sciplex import SciPlexSharedModel
from mora3.model.tokengt import TokenGT
from mora3.model.parameterization import response_parameter_ledger, width_family
from mora3.model.mu_optimizer import MuAdamW
from mora3.model.sciplex_mup import sciplex_mup_parameter_roles, sciplex_mup_width_family
from test_sciplex_shared import task


def make(width=8,**kw):
    args=dict(width=width,scaling_policy='mup-v1',attn_mult=.5,
        initialization_policy='mup-transformer-v1',encoder_var=.7,
        parameterization_policy='mup-sciplex-v1',base_width=8,readout_mult=1.5)
    args.update(kw)
    return SciPlexSharedModel(('A','B'),**args)


def models():return tuple(make(w).double() for w in (8,12,16))


@pytest.mark.parametrize('width',[8,12,16])
def test_reference_readout_and_hidden_bias_fresh_rescaling_preserves_other_weights_rng(width):
    args=dict(width=width,scaling_policy='mup-v1',attn_mult=.5,
              initialization_policy='mup-transformer-v1',encoder_var=.7)
    torch.manual_seed(2026110621);plain=SciPlexSharedModel(('A','B'),**args)
    rng=torch.get_rng_state().clone()
    torch.manual_seed(2026110621);m=make(width)
    assert torch.equal(rng,torch.get_rng_state())
    assert m.config['version']==4
    assert m.backbone.configuration()['schema']=='tokengt-native-configuration-v3'
    scaled={'response_head.weight','response_head.bias','intervention_head.weight','intervention_head.bias'}
    scaled.update(f'backbone.layers.{i}.ffn.{j}.bias' for i in range(2) for j in (0,2))
    before=dict(plain.named_parameters())
    for name,p in m.named_parameters():
        expected=before[name]*(math.sqrt(width/8) if name in scaled else 1.)
        assert torch.equal(p,expected),name


@pytest.mark.parametrize('width',[8,12,16])
@pytest.mark.parametrize('direction',['response','intervention'])
def test_actual_both_readout_callsites_independent_equation_all_gradients(width,direction):
    m=make(width).double();t=task(direction)
    latent=m.backbone([m.graph(t)])[0][:,0]
    head=m.response_head if direction=='response' else m.intervention_head
    expected=F.linear(latent*(1.5/(width/8)),head.weight,head.bias)
    actual=m([t]);torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    p=tuple(m.parameters());ga=torch.autograd.grad(actual.sum(),p,allow_unused=True)
    gb=torch.autograd.grad(expected.sum(),p,allow_unused=True)
    for a,b in zip(ga,gb):
        assert (a is None)==(b is None)
        if a is not None:torch.testing.assert_close(a,b,rtol=1e-11,atol=1e-11)


def test_exact_native_qualified_family_axes_aliases_counts_and_optimizer_groups_no_updates():
    ms=models();family=sciplex_mup_width_family(*ms)
    assert family.total_numel==(1256,2504,4168)
    m=ms[-1];before=[p.detach().clone() for p in m.parameters()];rng=torch.get_rng_state().clone()
    roles=sciplex_mup_parameter_roles(m)
    assert roles['backbone.incidence.weight'].role=='input'
    assert roles['backbone.incidence.weight'].axes[1].kind=='finite'
    assert roles['backbone.layers.0.attention.qkv.weight'].role=='hidden'
    assert [a.kind for a in roles['backbone.feature.weight'].axes]==['width','width']
    decayed=tuple(n for n,r in roles.items() if len(r.axes)==2)
    opt=MuAdamW(m,family,roles,base_lr=.02,weight_decay=.1,decay_aliases=decayed)
    by={g['mup_aliases'][0]:g for g in opt.param_groups}
    for name in ('backbone.feature.weight','backbone.layers.0.attention.qkv.weight','backbone.layers.0.ffn.0.weight'):
        assert by[name]['lr']==.01 and by[name]['weight_decay']==.2
    for name in ('backbone.incidence.weight','response_head.weight','intervention_head.weight'):
        assert by[name]['lr']==.02 and by[name]['weight_decay']==.1
    assert torch.equal(rng,torch.get_rng_state()) and not opt.state
    assert all(torch.equal(a,b) and b.grad is None for a,b in zip(before,m.parameters()))
    # The existing inventory's historical custom refusals are preserved.
    old=tuple(response_parameter_ledger(m) for m in ms)
    with pytest.raises(ValueError,match='custom'):width_family(*old)


@pytest.mark.parametrize('failure',['old-policy','wrong-base','wrong-readout','gene-order','depth','custom-module','tie'])
def test_qualified_family_refuses_unsupported_owner_policy_axis_and_alias_drift(failure):
    ms=list(models())
    if failure=='old-policy':ms[1]=SciPlexSharedModel(('A','B'),width=12).double()
    if failure=='wrong-base':ms[1]=make(12,base_width=6).double()
    if failure=='wrong-readout':ms[1]=make(12,readout_mult=2.).double()
    if failure=='gene-order':ms[1].axis_ids=('B','A')
    if failure=='depth':ms[1].backbone.layers=ms[1].backbone.layers[:1]
    if failure=='custom-module':ms[1].extra=torch.nn.Linear(12,12).double()
    if failure=='tie':ms[1].backbone.layers[1].ffn[0].weight=ms[1].backbone.layers[0].ffn[0].weight
    before=[[p.detach().clone() for p in m.parameters()] for m in ms];rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):sciplex_mup_width_family(*ms)
    assert torch.equal(rng,torch.get_rng_state())
    for m,values in zip(ms,before):assert all(torch.equal(a,b) for a,b in zip(values,m.parameters()))


@pytest.mark.parametrize('kw',[
    {'base_width':True},{'base_width':0},{'base_width':10},
    {'readout_mult':True},{'readout_mult':float('nan')},{'readout_mult':0},
    {'parameterization_policy':'unknown'},{'initialization_policy':'legacy'},
    {'parameterization_policy':'legacy','base_width':8},
])
def test_invalid_native_parameterization_precedes_rng(kw):
    rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):make(8,**kw)
    assert torch.equal(rng,torch.get_rng_state())


@pytest.mark.parametrize('width,ffn,base,bffn',[(12,24,8,16),(16,48,8,24)])
def test_native_nonsquare_ffn_bias_uses_its_actual_fanin_multiplier(width,ffn,base,bffn):
    args=dict(width=width,heads=2,layers=2,ffn_width=ffn,
              scaling_policy='mup-v1',initialization_policy='mup-transformer-v1')
    torch.manual_seed(2026110622);plain=TokenGT(width,2,**args)
    torch.manual_seed(2026110622);m=TokenGT(width,2,**args,
        parameterization_policy='mup-tokengt-v1',base_width=base,base_ffn_width=bffn)
    for old,new in zip(plain.layers,m.layers):
        assert torch.equal(new.ffn[0].bias,old.ffn[0].bias*math.sqrt(width/base))
        assert torch.equal(new.ffn[2].bias,old.ffn[2].bias*math.sqrt(ffn/bffn))
        assert torch.equal(new.ffn[0].weight,old.ffn[0].weight)


def test_parameterized_config_reconstruction_strict_load_never_rescales_loaded_tensors():
    m=make(16).double()
    with torch.no_grad():
        for p in m.parameters():p.add_(.125)
    state=copy.deepcopy(m.state_dict());config=m.get_extra_state()
    fresh=SciPlexSharedModel.from_configuration(config).double()
    rng=torch.get_rng_state().clone();fresh.load_state_dict(state,strict=True)
    assert torch.equal(rng,torch.get_rng_state())
    for p,q in zip(m.parameters(),fresh.parameters()):assert torch.equal(p,q)
    assert torch.equal(m([task()]),fresh([task()]))
    old=make(16,base_width=4).double();before=[p.detach().clone() for p in old.parameters()]
    with pytest.raises(ValueError):old.load_state_dict(state,strict=True)
    assert all(torch.equal(a,b) for a,b in zip(before,old.parameters()))


@pytest.mark.parametrize('failure',['base','mult','schema','native-base','missing'])
def test_parameterized_metadata_drift_refuses_before_reconstruction_rng(failure):
    cfg=make(16).get_extra_state()
    if failure=='base':cfg['mup']['base_width']=4
    if failure=='mult':cfg['mup']['output_mult']=True
    if failure=='schema':cfg['version']=3
    if failure=='native-base':cfg['native_backbone']['parameterization']['base_width']=4
    if failure=='missing':del cfg['mup']
    rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):SciPlexSharedModel.from_configuration(cfg)
    assert torch.equal(rng,torch.get_rng_state())
