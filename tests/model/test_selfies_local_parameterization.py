"""Packed reset/update/new GRU axes are custom, never ordinary hidden matrices."""
import pytest
import torch
from torch import nn
from test_profile_owner_parameterization import model as profiles
from mora3.model.profile_decoder import ProfileSelfiesDecoder, SelfiesCodec
from mora3.model import parameterization as p


def model(width=8):
    return ProfileSelfiesDecoder(profiles(width),SelfiesCodec(('CCO','CN')),max_tokens=32)


def local(m):
    roles=p.selfies_decoder_local_parameter_roles(m)
    # Local inventory deliberately excludes registered profile owner.
    return roles,p.parameter_ledger(nn.ModuleDict({'embedding':m.embedding,'rnn':m.rnn,'output':m.output}),roles)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('width',[8,12,16])
def test_local_count_gate_orientation_and_readonly(device,width):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(29101)
    m=model(width).to(device).eval()
    before={n:t.clone() for n,t in m.named_parameters()}
    rng=torch.random.get_rng_state().clone()
    gpu=torch.cuda.get_rng_state().clone() if device=='cuda' else None
    roles,ledger=local(m);v=m.codec.size;w=width
    assert ledger.total_numel==2*v*w+v+6*w*w+6*w
    h=p.Axis('width','hidden');g=p.Axis('width','selfies-gru-reset-update-new-packed')
    assert roles['embedding.weight'].axes==(roles['output.weight'].axes[0],h)
    assert roles['output.weight'].role=='output'
    for name in ('weight_ih_l0','weight_hh_l0'):
        assert roles['rnn.'+name]==p.ParameterRole('custom',(g,h))
    for name in ('bias_ih_l0','bias_hh_l0'):
        assert roles['rnn.'+name]==p.ParameterRole('custom',(g,))
    assert torch.equal(rng,torch.random.get_rng_state()) and not m.training
    if gpu is not None:assert torch.equal(gpu,torch.cuda.get_rng_state())
    assert all(torch.equal(t,before[n]) and t.grad is None for n,t in m.named_parameters())
    with pytest.raises(ValueError,match='custom'):p.width_family(ledger,ledger,ledger)


def test_vocabulary_order_is_identity_and_ties_are_checked():
    m=model();a,_=local(m)
    s=m.codec.symbols;m.codec.symbols=s[:3]+s[3:][::-1]
    m.codec.index={s:i for i,s in enumerate(m.codec.symbols)}
    b,_=local(m)
    assert a['embedding.weight'].axes!=b['embedding.weight'].axes
    m.output.weight=m.embedding.weight
    with pytest.raises(ValueError,match='conflicting'):local(m)


@pytest.mark.parametrize('drift',['root','subclass','extra-module','extra-parameter','codec-list',
    'codec-index','codec-size','codec-control','codec-duplicate','max-tokens','padding',
    'embedding-maxnorm','rnn-layers','rnn-bidirectional','rnn-batch','rnn-dropout',
    'rnn-size','rnn-bias','rnn-shape','output-shape','output-bias','owner'])
def test_configuration_and_ownership_drift_refuses(drift):
    m=model()
    if drift=='root':m=nn.Sequential(m)
    elif drift=='subclass':
        class Other(ProfileSelfiesDecoder):pass
        m.__class__=Other
    elif drift=='extra-module':m.extra=nn.Identity()
    elif drift=='extra-parameter':m.extra=nn.Parameter(torch.ones(1))
    elif drift=='codec-list':m.codec.symbols=list(m.codec.symbols)
    elif drift=='codec-index':m.codec.index={}
    elif drift=='codec-size':m.codec.size+=1
    elif drift=='codec-control':m.codec.pad=1
    elif drift=='codec-duplicate':m.codec.symbols=m.codec.symbols[:3]+('[C]',)*3
    elif drift=='max-tokens':m.max_tokens=True
    elif drift=='padding':m.embedding.padding_idx=None
    elif drift=='embedding-maxnorm':m.embedding.max_norm=1.
    elif drift=='rnn-layers':m.rnn.num_layers=2
    elif drift=='rnn-bidirectional':m.rnn.bidirectional=True
    elif drift=='rnn-batch':m.rnn.batch_first=False
    elif drift=='rnn-dropout':m.rnn.dropout=.1
    elif drift=='rnn-size':m.rnn.hidden_size=9
    elif drift=='rnn-bias':m.rnn.bias=False
    elif drift=='rnn-shape':m.rnn.weight_ih_l0=nn.Parameter(torch.ones(24,9))
    elif drift=='output-shape':m.output.weight=nn.Parameter(torch.ones(m.codec.size,9))
    elif drift=='output-bias':m.output.bias=None
    elif drift=='owner':m.profiles=nn.Identity()
    with pytest.raises(ValueError):local(m)
