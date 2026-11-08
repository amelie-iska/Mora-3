import copy
from dataclasses import replace
import pytest
import torch
from test_molecule_profile_model import model, view
from mora3.model import profile_decoder as module


def test_lossless_selfies_and_length_contract():
    codec=module.SelfiesCodec(('CCO','C[C@H](N)C(=O)O'))
    for s in ('CCO','C[C@H](N)C(=O)O'):
        ids=codec.encode(s,max_tokens=128)
        assert ids[0]==codec.bos and ids[-1]==codec.eos
        assert codec.decode(ids)
    with pytest.raises(ValueError,match='length'): codec.encode('CCO',max_tokens=2)
    with pytest.raises(ValueError,match='vocabulary'): codec.encode('Cl',max_tokens=128)
    with pytest.raises(ValueError,match='EOS'): codec.decode([codec.bos,3])


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_shifted_reference_target_exclusion_and_gradients(device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.manual_seed(8301)
    v=view('profile_to_molecule')
    codec=module.SelfiesCodec((v.targets()['canonical_isomeric_smiles'],'CCCO'))
    m=module.ProfileSelfiesDecoder(model(device),codec,max_tokens=64).to(device)
    batch=m.teacher_batch((v,))
    ids=codec.encode(v.targets()['canonical_isomeric_smiles'],max_tokens=64)
    assert batch[0].tolist()==[ids[:-1]] and batch[1].tolist()==[ids[1:]]
    logits=m((v,),batch[0]);loss=m.objective((v,))
    expected=torch.nn.functional.cross_entropy(logits.reshape(-1,codec.size),batch[1].reshape(-1),ignore_index=codec.pad)
    assert torch.equal(loss,expected)
    changed=replace(v,target_bytes=b'{"canonical_isomeric_smiles":"CCCO"}',provenance_bytes=b'{}')
    assert torch.equal(logits,m((changed,),batch[0]))
    # Future prefix tokens cannot influence earlier output positions.
    prefix=batch[0].clone();prefix[0,-1]=codec.bos
    assert torch.equal(logits[:,:-1],m((v,),prefix)[:,:-1])
    loss.backward()
    for p in (m.output.weight,m.rnn.weight_ih_l0,m.profiles.profile_input['L1000'].weight,m.profiles.backbone.feature.weight):
        assert p.grad is not None and p.grad.abs().sum()>0
    assert m.profiles.output['L1000'].weight.grad is None
    with pytest.raises(ValueError,match='direction'): m.objective((view(),))
    state=copy.deepcopy(m.get_extra_state());state['max_tokens']=2
    with pytest.raises(ValueError,match='configuration'):m.set_extra_state(state)
    with torch.no_grad():
        m.output.weight.zero_();m.output.bias.fill_(-100);m.output.bias[codec.eos]=100
    result=m.generate((changed,))
    assert result['token_ids']==((codec.bos,codec.eos),) and result['terminated']==(True,)


def test_codec_requires_nonempty_molecule():
    codec=module.SelfiesCodec(('CCO',))
    with pytest.raises(ValueError,match='nonempty'): codec.encode('',max_tokens=64)


def test_padding_excluded_and_unterminated_generation_explicit():
    a=view('profile_to_molecule')
    b=replace(a,target_bytes=b'{"canonical_isomeric_smiles":"CCCCO"}')
    codec=module.SelfiesCodec(('CCO','CCCCO'))
    m=module.ProfileSelfiesDecoder(model(),codec,max_tokens=16)
    prefix,target=m.teacher_batch((a,b))
    assert (target==codec.pad).any()
    z=m((a,b),prefix)
    losses=-z.log_softmax(-1).gather(-1,target[...,None]).squeeze(-1)
    torch.testing.assert_close(m.objective((a,b)),losses[target!=codec.pad].mean())
    with torch.no_grad():
        m.output.weight.zero_();m.output.bias.fill_(-100);m.output.bias[codec.index['[C]']]=100
    result=m.generate((a,))
    assert result['terminated']==(False,) and result['smiles']==(None,)
    assert len(result['token_ids'][0])==16


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_batched_shared_context_gru_boundary(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    a=view('profile_to_molecule');codec=module.SelfiesCodec(('CCO',))
    m=module.ProfileSelfiesDecoder(model(device),codec,max_tokens=64).to(device)
    loss=m.objective((a,a));loss.backward()
    assert torch.isfinite(loss)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_gru_gate_equation_reference(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(8302);v=view('profile_to_molecule');codec=module.SelfiesCodec(('CCO',))
    m=module.ProfileSelfiesDecoder(model(device),codec,max_tokens=64).to(device).double()
    prefix,target=m.teacher_batch((v,));h=m.profiles.condition((v,));outputs=[]
    for x in m.embedding(prefix).unbind(1):
        ir,iz,inn=torch.nn.functional.linear(x,m.rnn.weight_ih_l0,m.rnn.bias_ih_l0).chunk(3,-1)
        hr,hz,hn=torch.nn.functional.linear(h,m.rnn.weight_hh_l0,m.rnn.bias_hh_l0).chunk(3,-1)
        r=(ir+hr).sigmoid();z=(iz+hz).sigmoid();n=(inn+r*hn).tanh();h=(1-z)*n+z*h
        outputs.append(m.output(h))
    oracle=torch.stack(outputs,1)
    torch.testing.assert_close(m((v,),prefix),oracle,atol=1e-12,rtol=1e-12)
    expected=torch.nn.functional.cross_entropy(oracle.reshape(-1,codec.size),target.reshape(-1))
    actual=m.objective((v,))
    params=[p for p in m.parameters() if p.requires_grad]
    a=torch.autograd.grad(actual,params,allow_unused=True,retain_graph=True)
    b=torch.autograd.grad(expected,params,allow_unused=True)
    for x,y in zip(a,b):
        assert (x is None)==(y is None)
        if x is not None:torch.testing.assert_close(x,y,atol=1e-11,rtol=1e-10)
