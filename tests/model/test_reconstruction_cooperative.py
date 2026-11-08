import copy
from dataclasses import replace
import importlib.util
import pytest
import torch
from test_goa_cooperative import build as base_build
from test_molecule_profile_model import view
from mora3.model.profile_gfn import ProfileGFNPolicy
from mora3.model.profile_reconstruction import FixedProfileReconstruction


def build(tmp_path,device):
    assert importlib.util.find_spec('mora3.model.reconstruction_cooperative') is not None, 'shared reconstruction interface missing'
    from mora3.model.reconstruction_cooperative import ReconstructionCooperativeModel
    base,v=base_build(tmp_path,device)
    m=ReconstructionCooperativeModel(base.native,base.annotation,base.residue.head.feature_axis,
        base.profile.axes,codec=base.decoder.codec,max_tokens=64,
        annotation_registry=__import__('annotation_fixture').fixture_registry(),
        family_axis=base.rfam.family_axis,rfam_manifest_sha256='a'*64,goa_packets=(v,),
        reconstruction_dimension=2,reconstruction_horizon=3,reconstruction_hidden=5,
        reconstruction_source={'pool':'fixture'}).to(device).double()
    return m,v


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_shared_reconstruction_reference_all_gradients_ownership_and_baseline(tmp_path,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    m,v=build(tmp_path,device);r=m.reconstruction
    assert m.get_extra_state()["reconstruction"]["precision"]=="torch.float64"
    assert r.decoder is m.decoder and r.decoder.profiles is m.profile
    assert list(r.named_parameters())[0][0]=='latent_projection.weight'
    assert len(list(r.parameters()))==1
    assert not any(k.startswith('reconstruction.decoder.') for k in m.state_dict())
    copied=copy.deepcopy(m);assert copied.reconstruction.decoder is copied.decoder
    assert copied.decoder is not m.decoder
    vs=(view('profile_to_molecule'),view('profile_to_molecule','CellPainting'))
    x=torch.tensor([[.2,-.4],[.6,.1]],device=device,dtype=torch.float64)
    prefix,target=m.decoder.teacher_batch(vs)
    assert torch.equal(r(vs,prefix,torch.zeros_like(x)),m.decoder(vs,prefix))
    h=m.profile.condition(vs)+torch.nn.functional.linear(x,r.latent_projection.weight)
    outputs=[]
    for token in m.decoder.embedding(prefix).unbind(1):
        g=m.decoder.rnn
        ir,iz,inn=torch.nn.functional.linear(token,g.weight_ih_l0,g.bias_ih_l0).chunk(3,-1)
        hr,hz,hn=torch.nn.functional.linear(h,g.weight_hh_l0,g.bias_hh_l0).chunk(3,-1)
        reset=(ir+hr).sigmoid();z=(iz+hz).sigmoid();n=(inn+reset*hn).tanh();h=(1-z)*n+z*h
        outputs.append(m.decoder.output(h))
    oracle=torch.nn.functional.cross_entropy(torch.stack(outputs,1).flatten(0,1),target.flatten(),ignore_index=m.decoder.codec.pad)
    loss=r.objective(vs,x);ps=tuple(m.parameters())
    ga=torch.autograd.grad(loss,ps,allow_unused=True)
    gr=torch.autograd.grad(oracle,ps,allow_unused=True)
    for a,b in zip(ga,gr):
        assert (a is None)==(b is None)
        if a is not None:torch.testing.assert_close(a,b,atol=1e-11,rtol=1e-10)
    assert dict(zip(map(id,ps),ga))[id(m.native.response.backbone.feature.weight)] is not None
    scrub=tuple(replace(v,target_bytes=b'unreadable',provenance_bytes=b'unreadable') for v in vs)
    assert torch.equal(r(vs,prefix,x),r(scrub,prefix,x))
    assert torch.isfinite(m.goa_objective((v,)))
    state=copy.deepcopy(m.get_extra_state());state['reconstruction']['policy']['horizon']=99
    with pytest.raises(ValueError):m.set_extra_state(state)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_fixed_subset_row_identity_and_fail_closed(tmp_path,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    m,_=build(tmp_path,device)
    policy=ProfileGFNPolicy(m.profile,dimension=2,horizon=3,hidden=5,source_binding={'pool':'fixture'})
    vs=(view('profile_to_molecule'),view('profile_to_molecule','CellPainting'))
    b=FixedProfileReconstruction(m.reconstruction,policy,vs,seed=12401)
    expected=m.reconstruction.objective(vs[::-1],b.latents().flip(0))
    assert torch.equal(m.reconstruction_objective(b,indices=(1,0)),expected)
    assert torch.equal(m.reconstruction_objective(b),b.objective(m.reconstruction))
    for ix in ((),(0,0),(-1,),(2,),(True,),('0',)):
        with pytest.raises(ValueError):m.reconstruction_objective(b,indices=ix)
    b._views=b._views[::-1]
    with pytest.raises(ValueError,match='batch|order'):m.reconstruction_objective(b,indices=(0,))


def test_precision_configuration_tracks_parent_conversion(tmp_path):
    m,_=build(tmp_path,'cpu')
    m.float()
    assert m.get_extra_state()['reconstruction']['precision']=='torch.float32'
    m.set_extra_state(m.get_extra_state())
    m.double()
    m.load_state_dict(copy.deepcopy(m.state_dict()))
    assert m.reconstruction.latent_projection.weight.dtype==m.decoder.embedding.weight.dtype
