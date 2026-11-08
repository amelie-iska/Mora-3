"""Latent residual axes bind declared coordinates and exact shared configuration."""
import copy
import pytest
import torch
from torch import nn
from test_selfies_local_parameterization import model as decoder
from mora3.model.profile_gfn import ProfileGFNPolicy
from mora3.model.profile_latent_decoder import ProfileLatentSelfiesDecoder
from mora3.model import parameterization as p


def make(width=8):
    d=decoder(width)
    policy=ProfileGFNPolicy(d.profiles,dimension=3,horizon=2,hidden=8,source_binding={'scope':'fixture-only'})
    return ProfileLatentSelfiesDecoder(d,policy,share_decoder=True)


def roles(m, axis=('x','y','z'), binding=None, owner=None):
    return p.latent_reconstruction_local_parameter_roles(m,latent_axis=axis,
        decoder=m.decoder if owner is None else owner,
        configuration=m.get_extra_state() if binding is None else binding)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('width',[8,12,16])
def test_local_count_identity_and_no_mutation(device,width):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(30501)
    m=make(width).to(device).eval()
    state=copy.deepcopy(m.state_dict());rng=torch.get_rng_state().clone()
    gpu=torch.cuda.get_rng_state().clone() if device=='cuda' else None
    r=roles(m);a=r['latent_projection.weight'].axes[1]
    assert r=={'latent_projection.weight':p.ParameterRole('input',(p.Axis('width','hidden'),a))}
    assert a.kind=='finite'
    assert p.parameter_ledger(m,r).total_numel==3*width
    assert roles(m,axis=('z','y','x'))!=r
    assert torch.equal(rng,torch.get_rng_state()) and not m.training
    if gpu is not None:assert torch.equal(gpu,torch.cuda.get_rng_state())
    for n,t in m.state_dict().items():
        if torch.is_tensor(t):assert torch.equal(t,state[n])
        else:assert t==state[n]
    assert all(t.grad is None for t in m.parameters())


@pytest.mark.parametrize('drift',['axis-list','axis-empty','axis-duplicate','axis-length','axis-value',
    'owner','registered-decoder','extra-module','extra-parameter','buffer','layer-buffer',
    'layer-child','layer-subclass','weight','bias','config','dimension','horizon',
    'policy-dimension','policy-width','policy-profile','policy-version','binding','source','root-subclass'])
def test_drift_refuses(drift):
    m=make();binding=m.get_extra_state();axis=('x','y','z');owner=m.decoder
    if drift=='axis-list':axis=list(axis)
    elif drift=='axis-empty':axis=()
    elif drift=='axis-duplicate':axis=('x','x','z')
    elif drift=='axis-length':axis=('x','y')
    elif drift=='axis-value':axis=('x','y',True)
    elif drift=='owner':owner=copy.deepcopy(owner)
    elif drift=='registered-decoder':
        del m.decoder;m.decoder=owner
    elif drift=='extra-module':m.extra=nn.Identity()
    elif drift=='extra-parameter':m.extra=nn.Parameter(torch.ones(1))
    elif drift=='buffer':m.register_buffer('extra',torch.ones(1))
    elif drift=='layer-buffer':m.latent_projection.register_buffer('extra',torch.ones(1))
    elif drift=='layer-child':m.latent_projection.extra=nn.Identity()
    elif drift=='layer-subclass':
        class Other(nn.Linear):pass
        m.latent_projection.__class__=Other
    elif drift=='weight':m.latent_projection.weight=nn.Parameter(torch.ones(8,4))
    elif drift=='bias':m.latent_projection.bias=nn.Parameter(torch.ones(8))
    elif drift=='config':m.latent_projection.in_features=4
    elif drift=='dimension':m.dimension=True
    elif drift=='horizon':m.horizon=3
    elif drift=='policy-dimension':m._policy['dimension']=True
    elif drift=='policy-width':m._policy['context_width']=9
    elif drift=='policy-profile':m._policy['profiles']={}
    elif drift=='policy-version':m._policy['version']=True
    elif drift=='binding':binding['generation']='other'
    elif drift=='source':m._policy['source_binding']={}
    elif drift=='root-subclass':
        class Other(ProfileLatentSelfiesDecoder):pass
        m.__class__=Other
    with pytest.raises(ValueError):roles(m,axis,binding,owner)


def test_changed_source_changes_declared_axis():
    m=make();before=roles(m)
    m._policy['source_binding']={'scope':'different-constructor-only'}
    assert roles(m)!=before


def test_finite_coordinate_identity_survives_hidden_width_change():
    axes=[roles(make(w))['latent_projection.weight'].axes[1] for w in (8,12,16)]
    assert axes[0]==axes[1]==axes[2]
