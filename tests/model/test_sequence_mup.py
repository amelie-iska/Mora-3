"""Proposed next focused tests, staged during coordinated integrated suite.

No source data or NN diagnostics are executed by writing this proposal.
"""
import copy
import math
import pytest
import torch
from torch.nn import functional as F
from mora3.model.nucleotide_decoder import NucleotideDecoder,NucleotideCodec
from mora3.model.sequence_mup import sequence_mup_parameter_roles,sequence_mup_width_family
from test_annotation_mup import make as annotation


def make(width=8):
    return NucleotideDecoder(annotation(width),parameterization_policy='mup-sequence-v1')


def test_packed_three_gate_native_gru_initialization_and_checkpoint_rng():
    shared=annotation(12);size=NucleotideCodec().size
    torch.manual_seed(2026110671)
    embedding=torch.nn.Embedding(size,12,padding_idx=0)
    rnn=torch.nn.GRU(12,12,batch_first=True)
    output=torch.nn.Linear(12,size)
    rng=torch.get_rng_state().clone()
    torch.manual_seed(2026110671)
    model=NucleotideDecoder(shared,parameterization_policy='mup-sequence-v1')
    assert torch.equal(rng,torch.get_rng_state())
    assert torch.equal(model.embedding.weight,embedding.weight)
    for name,param in rnn.named_parameters():
        factor=math.sqrt(12/8) if name.startswith('bias') else 1
        assert torch.equal(dict(model.rnn.named_parameters())[name],param*factor)
    assert torch.equal(model.output.weight,output.weight*math.sqrt(12/8))
    assert torch.equal(model.output.bias,output.bias*math.sqrt(12/8))
    snapshot=copy.deepcopy(model.state_dict());rng=torch.get_rng_state().clone()
    model.load_state_dict(snapshot)
    assert torch.equal(rng,torch.get_rng_state())


def test_multi_step_native_gru_readout_reference_and_all_active_gradients(monkeypatch):
    m=make(16).double()
    context=torch.linspace(-.5,.7,32,dtype=torch.float64).reshape(2,16).requires_grad_()
    monkeypatch.setattr(m,'condition',lambda conditions:context)
    prefix=torch.tensor([[1,3,4],[1,4,3]])
    conditions=({'prior_domain':'synthetic-condition'},)*2
    hidden,_=m.rnn(m.embedding(prefix),context[None].contiguous())
    expected=F.linear(hidden*(1.5/(16/8)),m.output.weight,m.output.bias)
    actual=m(conditions,prefix)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    parameters=(context,*m.embedding.parameters(),*m.rnn.parameters(),*m.output.parameters())
    for a,b in zip(torch.autograd.grad(actual.sum(),parameters,retain_graph=True),torch.autograd.grad(expected.sum(),parameters)):
        torch.testing.assert_close(a,b,rtol=1e-11,atol=1e-11)


def test_actual_packed_gate_axes_finite_native_alphabet_and_family():
    ms=[make(w) for w in (8,12,16)]
    family=sequence_mup_width_family(*ms,identifier_axis=('protein-coordinate',))
    for model in ms:
        roles=sequence_mup_parameter_roles(model,identifier_axis=('protein-coordinate',))
        assert roles['rnn.weight_ih_l0'].role=='hidden'
        assert roles['embedding.weight'].role=='embedding'
        assert not any(r.role=='custom' for r in roles.values())
    assert family.total_numel==tuple(sum(p.numel() for p in m.parameters()) for m in ms)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_new_explicit_gru_storage_stays_independent_before_and_after_forward(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from mora3.model.sequence_mup import IndependentParameterGRU
    model=IndependentParameterGRU(12,12,batch_first=True).to(device)
    storages=lambda:[(p.untyped_storage().data_ptr(),p.untyped_storage().nbytes()) for p in model.parameters()]
    before=storages();assert len(set(before))==len(before)
    x=torch.randn(2,3,12,device=device);h=torch.randn(1,2,12,device=device)
    model(x,h)
    assert storages()==before


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_actual_native_gru_paired_deepcopy_rebinds_flat_refs_and_optimizer_checkpoint(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from mora3.model.mu_optimizer import MuAdamW
    from mora3.data.nucleotide_prior import SOURCES
    torch.manual_seed(2026110673)
    ms=[make(w).double().to(device) for w in (8,12,16)]
    family=sequence_mup_width_family(*ms,identifier_axis=('protein-coordinate',));model=ms[-1]
    roles=sequence_mup_parameter_roles(model,identifier_axis=('protein-coordinate',))
    optimizer=MuAdamW(model,family,roles,base_lr=.002,weight_decay=0,decay_aliases=())
    conditions=({'prior_domain':next(iter(SOURCES.values()))[0],'source_organism':None},)
    prefix=torch.tensor([[1,3,4]],device=device);target=torch.tensor([[3,4,2]],device=device)
    def update(m,o):
        o.zero_grad(set_to_none=True)
        loss=F.cross_entropy(m(conditions,prefix).flatten(0,1),target.flatten())
        loss.backward();o.step();return loss.detach().clone()
    update(model,optimizer)
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    original_optimizer=copy.deepcopy(optimizer.state_dict())
    original_storage=tuple(p.untyped_storage().data_ptr() for p in model.rnn.parameters())
    assert len(set(original_storage))==len(original_storage)
    # Native RNNBase deepcopy drops/rebuilds weak refs; exercise after a real
    # forward rather than checking only an uninitialized constructor's lists.
    copied,other=copy.deepcopy((model,optimizer))
    for name,param in zip(copied.rnn._flat_weights_names,copied.rnn._flat_weights):
        assert param is getattr(copied.rnn,name) and param is not getattr(model.rnn,name)
    copied(conditions,prefix)
    assert len(copied.rnn._flat_weight_refs)==len(copied.rnn._flat_weights)==4
    assert all(ref() is weight for ref,weight in zip(copied.rnn._flat_weight_refs,copied.rnn._flat_weights))
    assert len({p.untyped_storage().data_ptr() for p in copied.rnn.parameters()})==len(tuple(copied.rnn.parameters()))
    update(copied,other)
    for n,p in model.named_parameters():assert torch.equal(p,before[n])
    for k,slot in original_optimizer['state'].items():
        for name,value in slot.items():
            actual=optimizer.state_dict()['state'][k][name]
            assert torch.equal(value,actual) if isinstance(value,torch.Tensor) else value==actual
    snapshot=copy.deepcopy((model.state_dict(),optimizer.state_dict()))
    expected_loss=update(model,optimizer)
    expected={n:p.detach().clone() for n,p in model.named_parameters()}
    model.load_state_dict(snapshot[0]);optimizer.load_state_dict(snapshot[1])
    actual_loss=update(model,optimizer)
    tolerance=0 if device=='cpu' else 1e-10
    torch.testing.assert_close(actual_loss,expected_loss,rtol=tolerance,atol=tolerance)
    for n,p in model.named_parameters():torch.testing.assert_close(p,expected[n],rtol=tolerance,atol=tolerance)
    assert tuple(p.untyped_storage().data_ptr() for p in model.rnn.parameters())==original_storage


def test_rfam_reuses_exact_new_native_gru_and_qualifies_finite_family_readout_only():
    from mora3.model.rfam_family import RfamFamilyModel
    ms=[]
    for width in (8,12,16):
        decoder=make(width);before={n:p.detach().clone() for n,p in decoder.named_parameters()}
        family=RfamFamilyModel(decoder,('RF00001','RF00002'),manifest_sha256='a'*64,
            parameterization_policy='mup-rfam-v1')
        for name in ('shared','embedding','rnn','output'):assert getattr(family,name) is getattr(decoder,name)
        for n,p in decoder.named_parameters():assert torch.equal(p,before[n])
        ms.append(family)
    fam=sequence_mup_width_family(*ms,identifier_axis=('protein-coordinate',))
    assert fam.total_numel==tuple(sum(p.numel() for p in m.parameters()) for m in ms)
    roles=sequence_mup_parameter_roles(ms[-1],identifier_axis=('protein-coordinate',))
    assert roles['family_output.weight'].role=='output'
    assert 'specificity-limited-diagnostic' in ms[-1].get_extra_state()['mup']['positive_only_prediction']
