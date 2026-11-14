"""Test-only prefix-causal forest conditioning; no optimizer updates."""
from dataclasses import replace
from hashlib import sha256
import pytest
import torch

from test_forest_window_continuation import setup
from mora3.model.forest_token_decoder import _tensor_sha
from mora3.model.forest_causal_prefix import KnownForestCausalPrefix, causal_node_view


def case():
    operator, record, task, template, _ = setup()
    known_values = torch.tensor([[0.2, 0.4], [0.5, 0.9]], dtype=torch.float64)
    prefix = KnownForestCausalPrefix(
        source_snapshot_sha256=operator.decoder.field.spec.source_snapshot_sha256,
        context_sha256=template.context_sha256,
        coordinate_sha256=template.coordinate_sha256,
        evidence_sha256='e'*64,
        values_sha256=_tensor_sha(known_values),
        node_ids=('root','left'), states=known_values)
    stop = operator.decoder.stop_token
    target = torch.tensor([ord('a'),ord('b'),ord('c'),ord('d'),ord('e'),stop])
    return operator,record,task,template,prefix,target


def test_reduced_known_ancestor_chain_excludes_all_full_future_or_target_states():
    op,r,t,f,p,target=case()
    state,partial,index=causal_node_view(op.decoder,f,2,p)
    assert index == 2
    assert partial.node_ids==('root','left','left-end')
    assert partial.parents==(-1,0,1)
    assert partial.clamped==(0,1)
    assert partial.statuses==('open','open','terminal')
    assert torch.equal(state[0,:2],p.states)
    assert torch.equal(state[0,2],torch.zeros_like(state[0,2]))
    assert torch.equal(partial.states[2],torch.zeros_like(partial.states[2]))
    base=op.teacher_logits_prefix(r,t,f,2,p,target)
    altered=f.states.clone();altered[1:]=1234567
    f2=replace(f,states=altered,
        node_ids=('root','left','left-end','new-sibling','new-sibling-leaf'),
        directions=('root','computational','computational','observational','observational'))
    changed=op.teacher_logits_prefix(r,t,f2,2,p,target)
    assert torch.equal(base,changed)
    assert torch.equal(base,op.teacher_logits_prefix(r,t,f,2,p,target))
    assert torch.isfinite(op.teacher_loss_prefix(r,t,f,2,p,target))


def test_source_identity_and_exact_ancestry_are_fail_closed():
    op,r,t,f,p,target=case()
    failures=[
        replace(p,context_sha256='f'*64),
        replace(p,coordinate_sha256='f'*64),
        replace(p,source_snapshot_sha256='f'*64),
        replace(p,node_ids=('root','right')),
        replace(p,node_ids=('root','left-end')),
    ]
    for wrong in failures:
        with pytest.raises(ValueError):
            op.teacher_logits_prefix(r,t,f,2,wrong,target)
    with pytest.raises(ValueError):
        op.teacher_logits_prefix(r,t,f,0,p,target)
    with pytest.raises(ValueError):
        op.teacher_logits_prefix(r,t,f,4,p,target)
    with pytest.raises(ValueError):
        replace(p,values_sha256='f'*64).__post_init__()
    with pytest.raises(ValueError):
        replace(p,states=torch.ones(2,2,dtype=torch.float64,requires_grad=True)).__post_init__()


def test_source_prefix_state_mutation_refused_and_gradients_remain_trainable():
    op,r,t,f,p,target=case()
    rng=torch.get_rng_state().clone()
    prior=p.states.clone()
    loss=op.teacher_loss_prefix(r,t,f,2,p,target)
    loss.backward()
    for parameter in (op.prefix_input.weight,op.causal_gru.weight_ih_l0,op.decoder.readout.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert torch.equal(rng,torch.get_rng_state())
    assert torch.equal(p.states,prior)
    p.states[0,0]+=1
    with pytest.raises(ValueError,match='hash'):
        op.teacher_loss_prefix(r,t,f,2,p,target)
