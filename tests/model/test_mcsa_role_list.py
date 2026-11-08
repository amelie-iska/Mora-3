"""Sparse observed role LIST/EOS equations; synthetic condition requests only."""
import copy
from dataclasses import replace
import math
import pytest
import torch
from torch.nn import functional as F
from mora3.data.mcsa_weak_sites import WeakRoleVocabulary,WeakSiteBatch,SourceListSupervision
from mora3.model.mcsa_role_list import SourceRoleListDecoder,SourceRoleListModel,source_role_list_parameter_roles,source_role_list_width_family
from test_residue_mup import make as residue


@pytest.fixture(autouse=True)
def seed():torch.manual_seed(2026110663)


def vocab():return WeakRoleVocabulary((('a','f','t','g'),('b','f','t','g')))
def batch():
    v=vocab()
    rows=(SourceListSupervision(0,1,(v.bos_id,0,0),(0,0,v.eos_id),0,0),
          SourceListSupervision(0,2,(v.bos_id,),(v.eos_id,),1,0))
    return WeakSiteBatch(('ACDE',),rows,({'audit':'excluded'},),v.prediction_class_count,v.decoder_input_class_count,v.axis_sha256)
def make(w=8):return SourceRoleListModel(residue(w),vocab(),max_target_length=75)


def test_gru_each_gate_equation_and_readout_all_gradients():
    d=SourceRoleListDecoder(16,vocab(),base_width=8,output_mult=1.5).double()
    h=torch.linspace(-.7,.8,32,dtype=torch.float64).reshape(2,16).requires_grad_()
    ids=torch.tensor([vocab().bos_id,0]);x=d.embedding(ids)
    gi=F.linear(x,d.cell.weight_ih,d.cell.bias_ih).chunk(3,1)
    gh=F.linear(h,d.cell.weight_hh,d.cell.bias_hh).chunk(3,1)
    r=torch.sigmoid(gi[0]+gh[0]);z=torch.sigmoid(gi[1]+gh[1])
    n=torch.tanh(gi[2]+r*gh[2]);expected_state=(1-z)*n+z*h
    expected=F.linear(expected_state*(1.5/(16/8)),d.output.weight,d.output.bias)
    actual,state=d.step(ids,h)
    torch.testing.assert_close(state,expected_state,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    params=(h,*d.parameters())
    for a,b in zip(torch.autograd.grad(actual.sum(),params,retain_graph=True),torch.autograd.grad(expected.sum(),params)):
        torch.testing.assert_close(a,b,rtol=1e-11,atol=1e-11)


def test_fresh_gate_bias_scaling_and_rng_no_extra_draws():
    torch.manual_seed(2026110662)
    emb=torch.nn.Embedding(4,12);cell=torch.nn.GRUCell(12,12);out=torch.nn.Linear(12,3)
    rng=torch.get_rng_state().clone()
    torch.manual_seed(2026110662);d=SourceRoleListDecoder(12,vocab(),base_width=8,output_mult=1.5)
    assert torch.equal(rng,torch.get_rng_state())
    for name,p in cell.named_parameters():
        factor=math.sqrt(12/8) if name.startswith('bias') else 1
        assert torch.equal(dict(d.cell.named_parameters())[name],p*factor)
    assert torch.equal(d.embedding.weight,emb.weight)
    assert torch.equal(d.output.weight,out.weight*math.sqrt(12/8))


def test_native_sequence_only_features_role_history_exclusion_and_exact_list_loss():
    m=make().double();b=batch();features=m.encode_conditions(b)
    changed=replace(b,audit=({'source_id':'changed','target':'excluded'},),
        source_list_supervision=(replace(b.source_list_supervision[0],target_ids=(1,1,2),decoder_input_ids=(3,1,1)),b.source_list_supervision[1]))
    assert all(torch.equal(a,z) for a,z in zip(features,m.encode_conditions(changed)))
    logits=m(b);other=m(changed)
    assert torch.equal(logits[0][0],other[0][0])
    assert len(logits)==2 and [len(z) for z in logits]==[3,1]
    expected=F.cross_entropy(torch.cat(logits),torch.tensor([0,0,2,2]))
    assert torch.equal(m.objective(logits,b,purpose='implementation-diagnostic')['total'],expected)
    with pytest.raises(ValueError,match='admission'):
        m.objective(logits,b,purpose='training')
    # Even manually relabeling a candidate boolean is not a corpus certificate.
    with pytest.raises(ValueError,match='admission'):
        m.objective(logits,replace(b,task_admitted=True),purpose='training')


@pytest.mark.parametrize('change',['axis','prefix','eos','position','length','admission'])
def test_bad_batch_refuses_before_encoder_or_rng(change,monkeypatch):
    m=make();b=batch();row=b.source_list_supervision[0]
    if change=='axis':b=replace(b,vocabulary_axis_sha256='0'*64)
    elif change=='prefix':b=replace(b,source_list_supervision=(replace(row,decoder_input_ids=(3,1,0)),))
    elif change=='eos':b=replace(b,source_list_supervision=(replace(row,target_ids=(0,0,1)),))
    elif change=='position':b=replace(b,source_list_supervision=(replace(row,position_0based=4),))
    elif change=='length':b=replace(b,source_list_supervision=(replace(row,target_ids=(0,)*75+(2,),decoder_input_ids=(3,)+(0,)*75),))
    else:b=replace(b,task_admitted=True)
    monkeypatch.setattr(m.residue,'encode_sequences',lambda *a,**k:pytest.fail('encoder reached'))
    rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):m(b)
    assert torch.equal(rng,torch.get_rng_state())


def test_complete_source_family_and_exact_checkpoint_resume_without_reinitialization():
    ms=[make(w) for w in (8,12,16)]
    family=source_role_list_width_family(*ms,identifier_axis=('protein-coordinate',))
    roles=source_role_list_parameter_roles(ms[-1],identifier_axis=('protein-coordinate',))
    assert roles['decoder.cell.weight_ih'].role=='hidden'
    assert roles['decoder.embedding.weight'].role=='embedding'
    assert not any(r.role=='custom' for r in roles.values())
    assert family.total_numel==tuple(sum(p.numel() for p in m.parameters()) for m in ms)
    state=copy.deepcopy(ms[-1].state_dict());before=torch.get_rng_state().clone()
    ms[-1].load_state_dict(state)
    assert torch.equal(before,torch.get_rng_state())
    state['decoder._extra_state']['axis_sha256']='0'*64
    state['residue.shared.payload.embedding.weight'].fill_(99)
    tensors={k:v.clone() for k,v in ms[-1].state_dict().items() if isinstance(v,torch.Tensor)}
    with pytest.raises(ValueError,match='configuration|axis'):ms[-1].load_state_dict(state)
    for k,v in tensors.items():assert torch.equal(v,ms[-1].state_dict()[k])


def test_bounded_generation_reports_eos_and_exhaustion_without_truncation():
    d=SourceRoleListDecoder(8,vocab(),base_width=8)
    with torch.no_grad():d.output.weight.zero_();d.output.bias.zero_();d.output.bias[2]=10
    assert d.generate(torch.zeros(8),max_target_length=75)==dict(role_ids=(),termination='EOS',symbols=1)
    with torch.no_grad():d.output.bias[2]=0;d.output.bias[0]=10
    result=d.generate(torch.zeros(8),max_target_length=2)
    assert result==dict(role_ids=(0,0),termination='budget-exhausted',symbols=2)


def test_sparse_list_loss_has_no_gradient_at_unlisted_residues(monkeypatch):
    m=make().double();b=batch()
    features=torch.arange(32,dtype=torch.float64).reshape(4,8).div(32).requires_grad_()
    monkeypatch.setattr(m.residue,'encode_sequences',lambda sequences:(features,))
    logits=m(b);loss=m.objective(logits,b,purpose='implementation-diagnostic')['total']
    gradient=torch.autograd.grad(loss,features)[0]
    assert torch.count_nonzero(gradient[[0,3]])==0
    assert torch.count_nonzero(gradient[[1,2]])>0


@pytest.mark.parametrize('change',['axis','gate','base','module','width'])
def test_source_family_refuses_exact_axis_and_packed_operator_drift(change):
    ms=[make(w) for w in (8,12,16)];d=ms[-1].decoder
    if change=='axis':d.vocabulary=WeakRoleVocabulary((('c','f','t','g'),('d','f','t','g')))
    elif change=='gate':d.cell.weight_ih=torch.nn.Parameter(torch.empty(6*d.width,d.width))
    elif change=='base':d.base_width=4
    elif change=='module':d.cell.extra=torch.nn.Linear(d.width,d.width)
    else:d.cell.hidden_size=8
    with pytest.raises(ValueError):source_role_list_width_family(*ms,identifier_axis=('protein-coordinate',))


def test_width_independent_admission_binding_and_unauthenticated_purpose_refusal(monkeypatch):
    ms=[make(w) for w in (8,12,16)]
    bindings=[m.admission_binding() for m in ms]
    assert bindings[0]==bindings[1]==bindings[2]
    assert bindings[0]['source_family']=='MCSA-source-role-list-current-reference'
    assert bindings[0]['axis_sha256']==vocab().axis_sha256
    assert bindings[0]['biological_negatives'] is False
    assert bindings[0]['payload_max_tokens']==512
    assert bindings[0]['encoder_sequence_contract']=='complete-native-sequence-no-crop-or-truncation'
    m=ms[0]
    monkeypatch.setattr(m.residue,'encode_sequences',lambda *a:pytest.fail('encoder reached'))
    for purpose,split in [('training','train'),('training','validation'),('evaluation','validation')]:
        with pytest.raises(ValueError,match='admission'):
            m(batch(),purpose=purpose,split=split)
    with pytest.raises(ValueError,match='admission'):
        m(batch(),purpose='training',split='train',admission=object())
