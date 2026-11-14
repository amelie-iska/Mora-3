"""Explicit larger source-payload envelopes; no truncation or training updates."""
import copy
import pytest
import torch
from mora3.model.tokengt import TokenGT
from mora3.model.geometry_conditioning import GeometryConditioner
from mora3.data.schema import GraphRecord,Node,Measurement
from mora3.data.provenance import Provenance
from mora3.data.tasks import ConditionalTask


def build(cap=8192):
    torch.manual_seed(17013)
    b=TokenGT(8,2,width=8,heads=2,layers=1,ffn_width=8,
        scaling_policy='mup-v1',initialization_policy='mup-transformer-v1',
        parameterization_policy='mup-tokengt-v1',base_width=8,base_ffn_width=8)
    return GeometryConditioner(b,id_width=8,parameterization_policy='mup-geometry-v1',base_width=8,payload_max_tokens=cap).double()


def problem(length,target='hidden'):
    g=GraphRecord('source-capacity-test',(Node('source',(
        Measurement('condition','x'*length,None,'observed',True),
        Measurement('target',target,None,'observed',True))),),(),
        (Provenance('synthetic','v1','capacity-fixture','a'*64,'test-only'),))
    t=ConditionalTask(((('node',0,'condition'),'condition'),(('node',0,'target'),'target')),topology_known=True)
    return g,t


def test_default_stays_v2_and_explicit_cap_is_separately_versioned():
    standard=build();large=build(32768)
    assert standard.get_extra_state()['version']==2 and 'payload_max_tokens'not in standard.get_extra_state()
    state=large.get_extra_state()
    assert state['version']==3 and state['payload_max_tokens']==32768
    assert state['payload_input_policy']=='complete-explicit-token-cap-v1'
    assert large.payload.max_tokens==32768
    clone=copy.deepcopy(large);clone.load_state_dict(copy.deepcopy(large.state_dict()))
    assert clone.get_extra_state()==state
    large.payload.max_tokens=8192
    with pytest.raises(ValueError):large.get_extra_state()


def test_short_input_outputs_and_all_native_parameter_gradients_match_default():
    a=build();b=build(32768);g,t=problem(20)
    for (n,p),(m,q)in zip(a.named_parameters(),b.named_parameters()):
        assert n==m and torch.equal(p,q)
    x=a(g,t);y=b(g,t);torch.testing.assert_close(x,y,atol=0,rtol=0)
    x.square().sum().backward();y.square().sum().backward()
    for (n,p),(m,q)in zip(a.named_parameters(),b.named_parameters()):
        assert (p.grad is None)==(q.grad is None)
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)


def test_complete_long_payload_runs_without_using_hidden_target_or_changing_rank():
    a=build();b=build(32768);g,t=problem(10000)
    with pytest.raises(ValueError,match='bounds'):a(g,t)
    value=b(g,t);assert torch.isfinite(value).all() and b.config['id_width']==8
    other,task=problem(10000,'y'*60000)
    torch.testing.assert_close(value,b(other,task),atol=0,rtol=0)
    value.square().sum().backward()
    assert b.payload.embedding.weight.grad.abs().sum()>0


@pytest.mark.parametrize('cap',(0,True,-1,262145,1.5))
def test_invalid_capacity_refuses_before_initialization_rng(cap):
    b=TokenGT(8,2,width=8,heads=2,layers=1,ffn_width=8)
    before=torch.get_rng_state().clone()
    with pytest.raises(ValueError):GeometryConditioner(b,id_width=8,payload_max_tokens=cap)
    assert torch.equal(before,torch.get_rng_state())


def test_legacy_owner_cannot_silently_adopt_the_new_payload_policy():
    b=TokenGT(8,2,width=8,heads=2,layers=1,ffn_width=8);rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match='explicit'):GeometryConditioner(b,id_width=8,payload_max_tokens=32768)
    assert torch.equal(rng,torch.get_rng_state())


def test_old_and_new_payload_policy_checkpoints_cannot_cross_before_tensor_copy():
    a=build();b=build(32768);state=a.state_dict();before=b.payload.embedding.weight.detach().clone()
    state['payload.embedding.weight'].add_(7)
    with pytest.raises(ValueError):b.load_state_dict(state)
    assert torch.equal(before,b.payload.embedding.weight)
