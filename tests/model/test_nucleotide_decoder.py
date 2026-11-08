import copy
import json
import hashlib
import pytest
import torch
from mora3.model.annotation_multihead import SharedAnnotationModel
from mora3.model.annotation_vocabulary import AnnotationVocabulary
from mora3.data.nucleotide_prior import NucleotidePriorView, ALPHABET
from mora3.model import nucleotide_decoder as module


def view(seq='AcTUNx', organism=None):
    c=json.dumps(dict(prior_domain='source_DNA_sequence',source_organism=organism),sort_keys=True).encode()
    p=json.dumps(dict(sequence_sha256=hashlib.sha256(seq.encode()).hexdigest(),condition_sha256=hashlib.sha256(c).hexdigest())).encode()
    return NucleotidePriorView(seq,c,p)


def build(device='cpu'):
    vocab=AnnotationVocabulary(json.dumps(dict(learned_hex=[],controls=['pad','node','edge'])))
    shared=SharedAnnotationModel(vocab,('GO:0000001',),('1.1.1.1',),width=4,heads=2,layers=2).to(device)
    return module.NucleotideDecoder(shared).to(device)


def test_codec_full_native_symbols_and_no_8192_limit():
    c=module.NucleotideCodec();s=ALPHABET*275
    ids=c.encode(s)
    assert c.decode(ids)==s and len(ids)==len(s)+2
    for bad in ('','Z','A C'):
        with pytest.raises(ValueError):c.encode(bad)
    with pytest.raises(ValueError):c.decode([c.bos,c.bos,c.eos])


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_manual_gru_ce_every_gradient_and_target_exclusion(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(9301);m=build(device).double();vs=(view(),view('Tu'))
    prefix,target=m.teacher_batch(vs);conditions=tuple(v.inputs() for v in vs)
    h=m.condition(conditions);outs=[]
    for x in m.embedding(prefix).unbind(1):
        ir,iz,inn=torch.nn.functional.linear(x,m.rnn.weight_ih_l0,m.rnn.bias_ih_l0).chunk(3,-1)
        hr,hz,hn=torch.nn.functional.linear(h,m.rnn.weight_hh_l0,m.rnn.bias_hh_l0).chunk(3,-1)
        r=(ir+hr).sigmoid();z=(iz+hz).sigmoid();n=(inn+r*hn).tanh();h=(1-z)*n+z*h
        outs.append(m.output(h))
    oracle=torch.stack(outs,1);z=m(conditions,prefix)
    torch.testing.assert_close(z,oracle,atol=1e-12,rtol=1e-12)
    losses=-oracle.log_softmax(-1).gather(-1,target[...,None]).squeeze(-1)
    expected=losses[target!=m.codec.pad].mean();actual=m.objective(vs)
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
    params=tuple(m.parameters());a=torch.autograd.grad(actual,params,retain_graph=True,allow_unused=True)
    b=torch.autograd.grad(expected,params,allow_unused=True)
    for x,y in zip(a,b):
        assert (x is None)==(y is None)
        if x is not None:torch.testing.assert_close(x,y,atol=1e-11,rtol=1e-10)
    assert torch.equal(m.condition((vs[0].inputs(),)),m.condition((view('a'*9000).inputs(),)))
    changed=prefix.clone();changed[:,-1]=m.codec.bos
    assert torch.equal(z[:,:-1],m(conditions,changed)[:,:-1])
    actual=m.objective(vs);actual.backward()
    assert m.shared.backbone.feature.weight.grad.abs().sum()>0
    assert m.shared.payload.embedding.weight.grad.abs().sum()>0
    assert m.shared.go.output.weight.grad is None
    assert m.rnn.weight_ih_l0.grad.abs().sum()>0


def test_conditions_generation_and_config_contract():
    m=build();c=(view().inputs(),)
    for x in ({**c[0],'length':9},{**c[0],'source_organism':''},{**c[0],'prior_domain':'other'}):
        with pytest.raises(ValueError):m.condition((x,))
    state=copy.deepcopy(m.get_extra_state());state['symbols']=('A',)
    with pytest.raises(ValueError):m.set_extra_state(state)
    with torch.no_grad():
        m.output.weight.zero_();m.output.bias.fill_(-100);m.output.bias[m.codec.index['u']]=100
    r=m.generate(c,max_new_symbols=3)
    assert r['evidence_class']=='model_produced'
    assert r['terminated']==(False,) and r['sequences']==(None,)
    assert r['partial_sequences']==('uuu',)
    with torch.no_grad():m.output.bias[m.codec.eos]=200
    r=m.generate(c,max_new_symbols=3)
    assert r['terminated']==(True,) and r['sequences']==('',)
    with pytest.raises(ValueError):m.generate(c,max_new_symbols=0)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_full_long_teacher_target_masks(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    m=build(device);v=view(ALPHABET*275);prefix,target=m.teacher_batch((v,view('N')))
    assert prefix.shape==(2,len(v.sequence)+1)
    assert target[0,-1]==m.codec.eos
    assert (target[0,:-1]-3).tolist()==v.target_tensors()[0].tolist()
    assert target[1,1]==m.codec.eos and (target[1,2:]==m.codec.pad).all()
