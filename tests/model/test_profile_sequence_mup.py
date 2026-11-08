"""Native profile/SELFIES readout and GRU width equations, synthetic source fixtures."""
import copy
import math
from dataclasses import replace
import pytest
import torch
from torch.nn import functional as F
from mora3.model.molecule_profile import SharedMoleculeProfileModel
from mora3.model.profile_decoder import ProfileSelfiesDecoder,SelfiesCodec
from mora3.model.sequence_mup import profile_mup_parameter_roles,profile_mup_width_family,sequence_mup_parameter_roles,sequence_mup_width_family
from test_annotation_mup import make as annotation
from test_molecule_profile_model import view


@pytest.fixture(autouse=True)
def seed():torch.manual_seed(2026110672)


def profiles(w=8):
    return SharedMoleculeProfileModel(annotation(w),{m:view(modality=m).axis for m in ('CellPainting','L1000')},
        parameterization_policy='mup-molecule-profile-v1')

def decoder(w=8):
    return ProfileSelfiesDecoder(profiles(w),SelfiesCodec(('CCO','CCN')),max_tokens=32,
        parameterization_policy='mup-sequence-v1')


def test_actual_forward_profile_readouts_and_all_gradients_plus_target_exclusion():
    m=profiles(16).double();vs=(view(),view(modality='CellPainting'))
    latent=m.condition(vs)
    expected=tuple(F.linear(c*(1.5/(16/8)),m.output[v.modality].weight,m.output[v.modality].bias) for c,v in zip(latent,vs))
    actual=m(vs)
    params=tuple(m.parameters())
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
    for a,b in zip(torch.autograd.grad(sum(z.sum() for z in actual),params,allow_unused=True,retain_graph=True),
                   torch.autograd.grad(sum(z.sum() for z in expected),params,allow_unused=True)):
        assert (a is None)==(b is None)
        if a is not None:torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-10)
    changed=replace(vs[0],target_bytes=vs[0].target_bytes.replace(b'0.25',b'9.25'),provenance_bytes=b'{}')
    assert torch.equal(actual[0],m((changed,vs[1]))[0])


def test_complete_profile_and_selfies_width_families_and_exact_axes():
    ms=[profiles(w) for w in (8,12,16)]
    family=profile_mup_width_family(*ms,identifier_axis=('protein-coordinate',))
    assert family.total_numel==tuple(sum(p.numel() for p in m.parameters()) for m in ms)
    roles=profile_mup_parameter_roles(ms[-1],identifier_axis=('protein-coordinate',))
    assert roles['profile_input.CellPainting.weight'].role=='input'
    assert roles['output.CellPainting.weight'].role=='output'
    ds=[decoder(w) for w in (8,12,16)]
    fam=sequence_mup_width_family(*ds,identifier_axis=('protein-coordinate',))
    assert fam.total_numel==tuple(sum(p.numel() for p in m.parameters()) for m in ds)
    assert not any(r.role=='custom' for r in sequence_mup_parameter_roles(ds[-1],identifier_axis=('protein-coordinate',)).values())


def test_actual_selfies_prefix_outputs_and_all_gradients_and_native_codec():
    d=decoder(16).double();v=view('profile_to_molecule');prefix=torch.tensor([[1,3,4]])
    context=d.profiles.condition((v,))
    hidden,_=d.rnn(d.embedding(prefix),context[None].contiguous())
    expected=F.linear(hidden*(1.5/(16/8)),d.output.weight,d.output.bias)
    actual=d((v,),prefix)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    params=tuple(d.parameters())
    for a,b in zip(torch.autograd.grad(actual.sum(),params,allow_unused=True,retain_graph=True),
                   torch.autograd.grad(expected.sum(),params,allow_unused=True)):
        assert (a is None)==(b is None)
        if a is not None:torch.testing.assert_close(a,b,rtol=1e-10,atol=1e-10)
    changed=replace(v,target_bytes=b'{"canonical_isomeric_smiles":"CCN"}',provenance_bytes=b'{}')
    assert torch.equal(actual,d((changed,),prefix))


def test_shared_profile_owner_never_reinitialized_and_checkpoint_config_preflight():
    p=profiles(12);before={n:w.clone() for n,w in p.named_parameters()}
    d=ProfileSelfiesDecoder(p,SelfiesCodec(('CCO','CCN')),max_tokens=32,parameterization_policy='mup-sequence-v1')
    for n,w in p.named_parameters():assert torch.equal(w,before[n])
    state=copy.deepcopy(d.state_dict());rng=torch.get_rng_state().clone();d.load_state_dict(state)
    assert torch.equal(rng,torch.get_rng_state())
    state['_extra_state']['mup']['base_width']=4
    state['profiles.shared.payload.embedding.weight'].fill_(99)
    before={n:w.clone() for n,w in d.named_parameters()}
    with pytest.raises(ValueError,match='configuration|policy'):d.load_state_dict(state)
    for n,w in d.named_parameters():assert torch.equal(w,before[n])
