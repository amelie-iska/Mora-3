"""Source-list semantics/equations; synthetic labels, never biological truth."""
import copy
import pytest
import torch
from torch.nn import functional as F
from mora3.model.source_annotation_list import (SourceAnnotationListHead,
    source_annotation_list_parameter_roles,source_annotation_list_width_family)


def make(width=8,namespace='synthetic-source-list'):
    return SourceAnnotationListHead(width,('source-a','source-b'),source_namespace=namespace,
        base_width=8,output_mult=1.5,max_target_length=8)


def test_observed_repetition_order_and_empty_list_eos_cross_entropy():
    torch.manual_seed(2026110674);head=make(12).double()
    observed=((0,0,1),())
    context=torch.linspace(-1,1,24,dtype=torch.float64).reshape(2,12).requires_grad_()
    prefixes=head.teacher_prefixes(observed)
    assert prefixes==((3,0,0,1),(3,))
    logits=head(context,prefixes)
    expected=F.cross_entropy(torch.cat(logits),torch.tensor([0,0,1,2,2]))
    assert torch.equal(head.objective(logits,observed),expected)
    gradients=torch.autograd.grad(expected,(context,*head.parameters()))
    assert all(torch.isfinite(g).all() for g in gradients)
    # Missing annotation is never recoded as an observed empty list.
    for missing in (None,(None,),()):
        with pytest.raises(ValueError):head.teacher_prefixes(missing)


def test_each_native_gate_readout_reference_all_gradients_and_causal_prefix():
    torch.manual_seed(2026110675);h=make(16).double()
    c=torch.linspace(-.8,.9,16,dtype=torch.float64)[None].requires_grad_()
    ids=torch.tensor([h.bos]);x=h.embedding(ids)
    gi=F.linear(x,h.cell.weight_ih,h.cell.bias_ih).chunk(3,1)
    gh=F.linear(c,h.cell.weight_hh,h.cell.bias_hh).chunk(3,1)
    reset=torch.sigmoid(gi[0]+gh[0]);update=torch.sigmoid(gi[1]+gh[1])
    new=torch.tanh(gi[2]+reset*gh[2]);state=(1-update)*new+update*c
    expected=F.linear(state*(1.5/(16/8)),h.output.weight,h.output.bias)
    actual,_=h.step(ids,c)
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    params=(c,*h.parameters())
    for a,b in zip(torch.autograd.grad(actual.sum(),params,retain_graph=True),torch.autograd.grad(expected.sum(),params)):
        torch.testing.assert_close(a,b,rtol=1e-11,atol=1e-11)
    a=h(c,((h.bos,0,0),))[0];b=h(c,((h.bos,0,1),))[0]
    assert torch.equal(a[:2],b[:2])


def test_complete_explicit_source_namespace_family_and_checkpoint_refusal():
    ms=[make(w) for w in (8,12,16)]
    family=source_annotation_list_width_family(*ms)
    assert family.total_numel==tuple(sum(p.numel() for p in m.parameters()) for m in ms)
    roles=source_annotation_list_parameter_roles(ms[-1])
    assert not any(r.role=='custom' for r in roles.values())
    assert roles['embedding.weight'].role=='embedding'
    state=copy.deepcopy(ms[-1].state_dict());state['_extra_state']['source_namespace']='forged-other-source'
    state['embedding.weight'].fill_(99)
    before={n:p.clone() for n,p in ms[-1].named_parameters()}
    with pytest.raises(ValueError,match='source|policy'):ms[-1].load_state_dict(state)
    for n,p in ms[-1].named_parameters():assert torch.equal(p,before[n])
    with pytest.raises(ValueError):source_annotation_list_width_family(ms[0],ms[1],make(16,'other-source'))


def test_eos_and_budget_exhaustion_are_both_explicit_hypotheses():
    h=make()
    with torch.no_grad():h.output.weight.zero_();h.output.bias.zero_();h.output.bias[h.eos]=10
    result=h.generate(torch.zeros(8),max_symbols=8)
    assert result['termination']=='EOS' and result['annotation_ids']==()
    with torch.no_grad():h.output.bias[h.eos]=0;h.output.bias[0]=10
    result=h.generate(torch.zeros(8),max_symbols=2)
    assert result['termination']=='budget-exhausted' and result['annotation_ids']==('source-a','source-a')
    assert 'hypothesis' in result['semantics']
