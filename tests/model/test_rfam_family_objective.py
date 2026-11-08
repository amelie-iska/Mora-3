import copy
import json
import pytest
import torch
from mora3.data.rfam_family import RfamFamilyView, LABEL_POLICY
from mora3.model.nucleotide_decoder import NucleotideDecoder
from test_nucleotide_decoder import build
from mora3.model import rfam_family as module

AXIS=('RF00001','RF00002','RF00003')

def view(direction='prediction', seq='AcTU', family='RF00002'):
    c={'source_native_sequence':seq} if direction=='prediction' else dict(desired_source_Rfam_family_ids=[family],desired_source_clan_ids=['CL00001'])
    t=dict(source_Rfam_family_ids=[family],source_clan_ids=['CL00001'],family_label_policy=LABEL_POLICY) if direction=='prediction' else {'source_native_sequence':seq}
    p=dict(manifest_sha256='a'*64,family_axis=AXIS,envelope={'task_name':'Rfam_source_family_'+direction+'_prior'})
    return RfamFamilyView(*(json.dumps(x).encode() for x in (c,t,p)))

def model(device='cpu'):
    return module.RfamFamilyModel(build(device),AXIS,manifest_sha256='a'*64)

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_positive_only_reference_every_gradient_and_target_exclusion(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(9901);m=model(device).double();vs=(view(),view(seq='uN',family='RF00001'))
    z=m.predict(vs);actual=m.prediction_objective(vs)
    oracle=(torch.logaddexp(z.new_zeros(()),-z[0,1])+torch.logaddexp(z.new_zeros(()),-z[1,0]))/2
    torch.testing.assert_close(actual,oracle,atol=1e-12,rtol=1e-12)
    ps=tuple(m.parameters());a=torch.autograd.grad(actual,ps,retain_graph=True,allow_unused=True);b=torch.autograd.grad(oracle,ps,allow_unused=True)
    for x,y in zip(a,b):
        assert (x is None)==(y is None)
        if x is not None:torch.testing.assert_close(x,y,atol=1e-11,rtol=1e-10)
    assert torch.equal(z,m.predict((view(family='RF00003'),vs[1])))
    actual=m.prediction_objective(vs);actual.backward()
    assert m.family_output.weight.grad[2].count_nonzero()==0
    assert m.shared.backbone.feature.weight.grad.abs().sum()>0
    assert m.shared.payload.embedding.weight.grad.abs().sum()>0

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_generation_reference_condition_target_exclusion_and_shared_ownership(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    m=model(device).double();v=view('generation');prefix=torch.tensor([[1,3,4]],device=device)
    c=(v.inputs(),);z=m(c,prefix);h=m.condition(c).unsqueeze(0).contiguous()
    expected=m.output(m.rnn(m.embedding(prefix),h)[0]);torch.testing.assert_close(z,expected)
    a=m.generation_objective((v,));a.backward()
    assert m.shared.payload.embedding.weight.grad.abs().sum()>0
    assert torch.equal(m.condition(c),m.condition((view('generation',seq='N'*9000).inputs(),)))
    assert not torch.equal(m.condition(c),m.condition((view('generation',family='RF00001').inputs(),)))
    assert len(list(m.parameters()))==len({id(p) for p in m.parameters()})
    with torch.no_grad():m.output.weight.zero_();m.output.bias.fill_(-100);m.output.bias[m.codec.eos]=100
    assert m.generate(c,max_new_symbols=2)['terminated']==(True,)

@pytest.mark.parametrize('change',['axis','manifest','direction','input'])
def test_fail_closed_binding(change):
    m=model();v=view();p=v.provenance();c=v.inputs()
    if change=='axis':p['family_axis']=list(reversed(AXIS))
    if change=='manifest':p['manifest_sha256']='b'*64
    if change=='direction':p['envelope']['task_name']='Rfam_source_family_generation_prior'
    if change=='input':c['hidden_label']='RF00001'
    bad=RfamFamilyView(json.dumps(c).encode(),v.target_bytes,json.dumps(p).encode())
    with pytest.raises(ValueError):m.prediction_objective((bad,))
    s=copy.deepcopy(m.state_dict());s['_extra_state']['manifest_sha256']='c'*64
    with pytest.raises(ValueError):m.load_state_dict(s)


def test_exact_decoder_reuse_generation_ce_gradients_and_long_target():
    base=build();m=module.RfamFamilyModel(base,AXIS,manifest_sha256='a'*64).double()
    assert m.shared is base.shared and m.rnn is base.rnn and m.embedding is base.embedding
    vs=(view('generation'),view('generation',seq='Nu'))
    prefix,target=m.teacher_batch(vs);z=m(tuple(v.inputs() for v in vs),prefix)
    # Independent ordered lexical/EOS likelihood; PAD excluded explicitly.
    losses=-z.log_softmax(-1).gather(-1,target[...,None]).squeeze(-1)
    oracle=losses[target!=0].mean();actual=m.generation_objective(vs)
    torch.testing.assert_close(actual,oracle,atol=1e-12,rtol=1e-12)
    ps=tuple(m.parameters());a=torch.autograd.grad(actual,ps,retain_graph=True,allow_unused=True);b=torch.autograd.grad(oracle,ps,allow_unused=True)
    for x,y in zip(a,b):
        assert (x is None)==(y is None)
        if x is not None:torch.testing.assert_close(x,y,atol=1e-11,rtol=1e-10)
    prefix,target=m.teacher_batch((view('generation',seq='N'*9000),vs[1]))
    assert target.shape==(2,9001) and target[0,-1]==2
    assert (target[1,3:]==0).all()
    assert m.codec.decode([1]+target[0].tolist())=='N'*9000


@pytest.mark.parametrize('axis', [(),('RF00001','RF00001'),('wrong',),tuple(reversed(AXIS))])
def test_axis_constructor_rejects_ambiguous_coordinates(axis):
    with pytest.raises(ValueError):module.RfamFamilyModel(build(),axis,manifest_sha256='a'*64)
