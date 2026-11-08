import copy
import importlib.util
from dataclasses import replace
import pytest
import torch
from torch.nn import functional as F
from test_molecule_profile_model import model, view
from mora3.model.profile_decoder import ProfileSelfiesDecoder, SelfiesCodec
from mora3.model.profile_gfn import ProfileGFNPolicy


def build(device):
    assert importlib.util.find_spec('mora3.model.profile_latent_decoder') is not None, 'latent decoder bridge missing'
    from mora3.model.profile_latent_decoder import ProfileLatentSelfiesDecoder
    profiles=model(device).double()
    policy=ProfileGFNPolicy(profiles,dimension=2,horizon=3,hidden=5,source_binding={'pool':'fixture'})
    decoder=ProfileSelfiesDecoder(profiles,SelfiesCodec(('CCO','CCCCO')),max_tokens=16)
    return ProfileLatentSelfiesDecoder(decoder,policy),policy


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_residual_gru_reference_every_gradient_zero_and_exclusion(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(12001);m,p=build(device)
    vs=(view('profile_to_molecule'),view('profile_to_molecule','CellPainting'))
    prefix,target=m.decoder.teacher_batch(vs)
    x=torch.tensor([[.2,-.4],[.6,.1]],device=device,dtype=torch.float64,requires_grad=True)
    assert torch.equal(m(vs,prefix,torch.zeros_like(x)),m.decoder(vs,prefix))
    ref=copy.deepcopy(m);h=ref.decoder.profiles.condition(vs)+F.linear(x,ref.latent_projection.weight)
    outputs=[]
    for token in ref.decoder.embedding(prefix).unbind(1):
        rnn=ref.decoder.rnn
        ir,iz,inn=F.linear(token,rnn.weight_ih_l0,rnn.bias_ih_l0).chunk(3,-1)
        hr,hz,hn=F.linear(h,rnn.weight_hh_l0,rnn.bias_hh_l0).chunk(3,-1)
        r=(ir+hr).sigmoid();z=(iz+hz).sigmoid();n=(inn+r*hn).tanh();h=(1-z)*n+z*h
        outputs.append(ref.decoder.output(h))
    oracle=torch.stack(outputs,1);actual=m(vs,prefix,x)
    torch.testing.assert_close(actual,oracle,atol=1e-12,rtol=1e-12)
    loss=m.objective(vs,x)
    expected=F.cross_entropy(oracle.flatten(0,1),target.flatten(),ignore_index=m.decoder.codec.pad)
    ga=torch.autograd.grad(loss,tuple(m.parameters())+(x,),allow_unused=True)
    gr=torch.autograd.grad(expected,tuple(ref.parameters())+(x,),allow_unused=True)
    for a,b in zip(ga,gr):
        assert (a is None)==(b is None)
        if a is not None:torch.testing.assert_close(a,b,atol=1e-11,rtol=1e-10)
    assert ga[-1].abs().sum()>0
    scrub=tuple(replace(v,target_bytes=b'unreadable',provenance_bytes=b'unreadable') for v in vs)
    assert torch.equal(actual,m(scrub,prefix,x))
    torch.testing.assert_close(actual.flip(0),m(vs[::-1],prefix.flip(0),x.flip(0)),atol=0,rtol=0)
    assert not torch.equal(actual,m(vs,prefix,x.flip(0)))
    for bad in (x[:1],x.float(),x*float('nan')):
        with pytest.raises(ValueError):m(vs,prefix,bad)
    with pytest.raises(ValueError):m(vs,prefix.float(),x)
    config=copy.deepcopy(m.get_extra_state());config['policy']['horizon']=99
    with pytest.raises(ValueError,match='configuration'):m.set_extra_state(config)
    assert m.get_extra_state()['precision']=='torch.float64'


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_ragged_terminal_rows_and_strict_generation(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    m,p=build(device);vs=(view('profile_to_molecule'),)*2
    states=torch.arange(12,device=device,dtype=torch.float64).reshape(2,3,2)
    states[0,1:]=0
    paths=dict(states=states,lengths=torch.tensor([1,3],device=device),
        mask=torch.tensor([[True,False,False],[True,True,True]],device=device),optional_stop=True)
    x=m.terminal_latents(paths)
    assert torch.equal(x,torch.stack((states[0,0],states[1,2])))
    assert not x.requires_grad
    with pytest.raises(ValueError):m.terminal_latents(dict(paths,lengths=paths['lengths']*0))
    with pytest.raises(ValueError):m.terminal_latents(dict(paths,mask=~paths['mask']))
    with pytest.raises(ValueError):m.terminal_latents(dict(paths,states=states.clone().requires_grad_()))
    with torch.no_grad():
        m.decoder.output.weight.zero_();m.decoder.output.bias.fill_(-100)
        m.decoder.output.bias[m.decoder.codec.eos]=100
    out=m.generate(vs,x)
    assert out['terminated']==(True,True) and out['valid']==(False,False)
    assert out['smiles']==(None,None) and out['status']==('invalid_molecule','invalid_molecule')
    with torch.no_grad():m.decoder.output.bias[m.decoder.codec.index['[C]']]=200
    out=m.generate(vs,x)
    assert out['status']==('unterminated','unterminated') and out['smiles']==(None,None)
    # Produce C then EOS by supplying an explicit recurrent reference fixture.
    accepted=m.validate_sequences(((1,m.decoder.codec.index['[C]'],2),(1,3)))
    assert accepted['valid']==(True,False) and accepted['status']==('valid','unterminated')
    assert accepted['smiles'][0]=='C'
