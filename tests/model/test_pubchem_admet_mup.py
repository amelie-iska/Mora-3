"""New opt-in PubChem/ADMET equations, source byte codec and guarded recovery."""
import copy
import pytest
import torch
from torch.nn import functional as F
from test_annotation_mup import make as annotation
from test_admet_model import setup
from test_pubchem_decoder import ROOT,MASTER,SHA
from mora3.model.pubchem_decoder import PubChemDecoder
from mora3.model.admet_model import SharedAdmetModel
from mora3.model.admet_mup import admet_mup_parameter_roles,admet_mup_width_family
from mora3.model.sequence_mup import sequence_mup_parameter_roles,sequence_mup_width_family,IndependentParameterGRU
from mora3.model.mu_optimizer import MuAdamW


def admet(width=8):
    _,v,c,_,_=setup('cpu','regression');_,b,d,_,_=setup('cpu','binary')
    return SharedAdmetModel(annotation(width,max_tokens=8192),(c,d),parameterization_policy='mup-admet-v1').double(),(v,b),(c,d)


def pubchem(width=8):
    return PubChemDecoder(annotation(width,max_tokens=8192),source_root=ROOT,manifest=MASTER,
        expected_manifest_sha256=SHA,parameterization_policy='mup-sequence-v1').double()


def test_admet_both_partition_outputs_and_all_gradient_equation_noncancelling():
    m,views,contracts=admet(16);context=m.condition(v.inputs() for v in views);actual=m(views,contracts)
    expected=[]
    for row,c in zip(context,contracts):
        raw=F.linear(row*.75,m.heads[c.partition_key].weight,m.heads[c.partition_key].bias)
        expected.append(dict(logit=raw[0]) if c.definition.objective_kind=='binary' else
            dict(mean=raw[0],scale=F.softplus(raw[1])+torch.finfo(raw.dtype).eps))
    for x,y in zip(actual,expected):
        for k in x:torch.testing.assert_close(x[k],y[k],rtol=0,atol=0)
    params=tuple(m.parameters());a=torch.autograd.grad(sum(t.square() for o in actual for t in o.values()),params,allow_unused=True)
    b=torch.autograd.grad(sum(t.square() for o in expected for t in o.values()),params,allow_unused=True)
    for x,y in zip(a,b):
        assert (x is None)==(y is None)
        if x is not None:torch.testing.assert_close(x,y,rtol=1e-11,atol=1e-11)
    for v,c,o in zip(views,contracts,actual):
        with pytest.raises(ValueError,match='admission'):m.observed_loss(o,v,c)


def test_admet_complete_family_optimizer_no_update_partition_axes_and_recovery():
    ms=tuple(admet(w)[0] for w in (8,12,16));f=admet_mup_width_family(*ms,identifier_axis=('protein-coordinate',));m=ms[-1]
    roles=admet_mup_parameter_roles(m,identifier_axis=('protein-coordinate',))
    opt=MuAdamW(m,f,roles,base_lr=.01,weight_decay=0.,decay_aliases=());assert not opt.state
    assert f.total_numel[-1]==sum(p.numel() for p in m.parameters())
    for n,r in roles.items():
        if n.startswith('heads.') and n.endswith('weight'):assert r.role=='output' and r.axes[0].kind=='finite'
    before=copy.deepcopy(m.state_dict());bad=copy.deepcopy(before)
    first=next(n for n,t in bad.items() if isinstance(t,torch.Tensor));bad[first]+=3
    bad['_extra_state']['mup']['output_mult']=7.
    with pytest.raises(ValueError):m.load_state_dict(bad)
    for n,t in m.state_dict().items():
        if isinstance(t,torch.Tensor):assert torch.equal(t,before[n])


def test_actual_pubchem_byte_forward_and_generation_readouts_noncancelling():
    m=pubchem(16);assert type(m.rnn)is IndependentParameterGRU
    condition=({'prior_domain':'molecular_reference_native_SMILES'},)
    prefix=torch.tensor([[m.codec.bos,67,67]],dtype=torch.long)
    context=m.condition(condition);hidden,_=m.rnn(m.embedding(prefix),context.unsqueeze(0).contiguous())
    expected=F.linear(hidden*.75,m.output.weight,m.output.bias)
    actual=m(condition,prefix);torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    params=tuple(m.parameters());a=torch.autograd.grad(actual.square().sum(),params,allow_unused=True)
    b=torch.autograd.grad(expected.square().sum(),params,allow_unused=True)
    for x,y in zip(a,b):
        assert (x is None)==(y is None)
        if x is not None:torch.testing.assert_close(x,y,rtol=1e-11,atol=1e-11)
    seen=[];original=m._readout
    def readout(h):seen.append(h.shape);return original(h)
    m._readout=readout;result=m.generate(condition,max_new_tokens=3)
    assert seen and result['caller_budget']==3 and not result['representation_validated']


def test_pubchem_complete_source_codec_width_family_and_checkpoint_no_reinitialization():
    ms=tuple(pubchem(w) for w in (8,12,16));f=sequence_mup_width_family(*ms,identifier_axis=('protein-coordinate',));m=ms[-1]
    roles=sequence_mup_parameter_roles(m,identifier_axis=('protein-coordinate',))
    opt=MuAdamW(m,f,roles,base_lr=.01,weight_decay=0.,decay_aliases=());assert not opt.state
    assert f.total_numel[-1]==sum(p.numel() for p in m.parameters())
    state=copy.deepcopy(m.state_dict());rng=torch.get_rng_state().clone();m.load_state_dict(state)
    assert torch.equal(rng,torch.get_rng_state())
    before=[p.detach().clone() for p in m.parameters()];bad=copy.deepcopy(state)
    bad['_extra_state']['mup']['base_width']=7
    with pytest.raises(ValueError):m.load_state_dict(bad)
    assert all(torch.equal(a,b) for a,b in zip(before,m.parameters()))
    assert m.codec.tokenizer.vocabulary[:256]==tuple(bytes([i]) for i in range(256))


@pytest.mark.parametrize('kind',['admet','pubchem'])
def test_legacy_new_shared_owner_missing_optin_refuses_before_rng(kind):
    shared=annotation();_,_,c,_,_=setup('cpu')
    rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match='explicit'):
        if kind=='admet':SharedAdmetModel(shared,(c,))
        else:PubChemDecoder(shared,source_root=ROOT,manifest=MASTER,expected_manifest_sha256=SHA)
    assert torch.equal(rng,torch.get_rng_state())


@pytest.mark.parametrize('kind',['admet','pubchem'])
def test_fresh_readout_and_recurrent_bias_rescale_exact_native_draws_rng(kind):
    import math
    from mora3.model.annotation_multihead import SharedAnnotationModel
    from test_annotation_mup import vocab
    new=annotation(16,max_tokens=8192)
    old=SharedAnnotationModel(vocab(),('GO:0000001','GO:0000002'),('1.1.1.1','2.2.2.2'),
        width=16,heads=2,layers=2,max_tokens=8192)
    if kind=='admet':
        _,_,c,_,_=setup('cpu','regression');_,_,d,_,_=setup('cpu','binary')
        torch.manual_seed(12817);a=SharedAdmetModel(old,(c,d));rng=torch.get_rng_state().clone()
        torch.manual_seed(12817);b=SharedAdmetModel(new,(c,d),parameterization_policy='mup-admet-v1')
        assert torch.equal(rng,torch.get_rng_state())
        for name,p in a.heads.named_parameters():
            torch.testing.assert_close(dict(b.heads.named_parameters())[name],p*math.sqrt(2),rtol=0,atol=0)
    else:
        options=dict(source_root=ROOT,manifest=MASTER,expected_manifest_sha256=SHA)
        torch.manual_seed(12818);a=PubChemDecoder(old,**options);rng=torch.get_rng_state().clone()
        torch.manual_seed(12818);b=PubChemDecoder(new,**options,parameterization_policy='mup-sequence-v1')
        assert torch.equal(rng,torch.get_rng_state())
        params=dict(b.named_parameters())
        for name,p in a.named_parameters():
            if name.startswith('shared.'):continue
            factor=math.sqrt(2) if name in ('rnn.bias_ih_l0','rnn.bias_hh_l0','output.weight','output.bias') else 1
            torch.testing.assert_close(params[name],p*factor,rtol=0,atol=0)
