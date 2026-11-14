"""Actual shared native topology policy; synthetic correctness, no optimizer steps."""
import copy
from dataclasses import replace
import math
import pytest
import torch
from test_forest_flow import make,problem,SOURCE,COORD
from mora3.reasoning.forest_topology import TopologyBounds,prefix_state,canonical_forest_actions,CLOSE,ADD_CHILD
from mora3.model.forest_topology import (ForestTopologySpec,NativeForestTopologyHead,
    topology_log_probability,forest_topology_loss,sample_forest_topologies,
    sampled_topology_forest,forest_topology_parameter_roles)


def policy(nodes=4):
    torch.manual_seed(20261114)
    flow=make()
    return NativeForestTopologyHead(flow,ForestTopologySpec(TopologyBounds(nodes,nodes-1,nodes-1),'a'*64)).double()


def outcomes(bounds):
    pending=[()];out=[]
    while pending:
        a=pending.pop();s=prefix_state(bounds,1,a)
        if s.complete:out.append(s.parents)
        else:pending.extend(a+(x,)for x in s.allowed_actions)
    return out


def test_real_normalized_policy_enumeration_and_original_logsumexp_reference():
    h=policy();r,t=problem();logs=[]
    for parents in outcomes(h.spec.bounds):
        x=topology_log_probability(h,r,t,parents,root_count=1)
        logs.append(x['log_probability'])
        canonical=canonical_forest_actions(h.spec.bounds,parents,root_count=1)
        terms=[]
        for i,action in enumerate(canonical.actions):
            logits=h(r,t,root_count=1,actions=canonical.actions[:i])
            valid=prefix_state(h.spec.bounds,1,canonical.actions[:i]).allowed_actions
            terms.append(logits[action]-torch.logsumexp(logits[list(valid)],dim=0))
        torch.testing.assert_close(x['log_probability'],sum(terms),atol=1e-14,rtol=1e-14)
    torch.testing.assert_close(torch.stack(logs).exp().sum(),torch.tensor(1.,dtype=torch.float64),atol=1e-12,rtol=0)


def test_teacher_loss_matches_trace_likelihood_and_native_shared_gradients():
    h=policy();r,t=problem();forests=((-1,0,0),(-1,0,1))
    actual=forest_topology_loss(h,r,t,forests,root_count=1,split='train')['loss']
    ref=-torch.stack([topology_log_probability(h,r,t,p,root_count=1)['log_probability']for p in forests]).mean()
    torch.testing.assert_close(actual,ref,atol=0,rtol=0)
    parameters=(h.action_readout.weight,h.structure_projection.weight,h.field.geometry.graph.feature.weight)
    ga=torch.autograd.grad(actual,parameters,retain_graph=True);gb=torch.autograd.grad(ref,parameters)
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b,atol=1e-13,rtol=1e-13)
    assert all(g.abs().sum()>0 for g in ga)


def test_hidden_targets_and_future_subtree_never_change_prefix_policy():
    h=policy();r,t=problem();r2=replace(r,nodes=(replace(r.nodes[0],measurements=tuple(
        replace(m,value=999999.)if m.name=='heldout'else m for m in r.nodes[0].measurements)),*r.nodes[1:]))
    for actions in((),(ADD_CHILD,),(ADD_CHILD,CLOSE)):
        torch.testing.assert_close(h(r,t,root_count=1,actions=actions),h(r2,t,root_count=1,actions=actions),atol=0,rtol=0)
    a=canonical_forest_actions(h.spec.bounds,(-1,0),root_count=1)
    b=canonical_forest_actions(h.spec.bounds,(-1,0,0),root_count=1)
    assert a.actions[:2]==b.actions[:2]
    assert torch.equal(h(r,t,root_count=1,actions=a.actions[:2]),h(r,t,root_count=1,actions=b.actions[:2]))


def test_sampling_uses_explicit_rng_and_replays_exact_complete_probabilities():
    h=policy().eval();r,t=problem();g=torch.Generator().manual_seed(81);global_before=torch.get_rng_state().clone()
    first=sample_forest_topologies(h,r,t,root_count=1,count=12,generator=g)
    g.set_state(first['generator_before']);second=sample_forest_topologies(h,r,t,root_count=1,count=12,generator=g)
    assert first['samples']==second['samples']
    assert torch.equal(g.get_state(),first['generator_after'])and torch.equal(global_before,torch.get_rng_state())
    assert first['on_policy_for_bound_topology_policy']is True and first['reward_proportionality_verified']is False
    for sample in first['samples']:
        s=prefix_state(h.spec.bounds,1,tuple(sample['actions']))
        assert s.complete and list(s.parents)==sample['parents']
        ref=topology_log_probability(h,r,t,s.parents,root_count=1)['log_probability']
        assert math.isclose(sample['log_probability'],float(ref.detach()),abs_tol=1e-13)


def test_generated_structure_connects_to_actual_existing_flow_without_gold_states():
    from mora3.reasoning.flow_forest_refinement import sample_flow_forests
    h=policy().eval();r,t=problem();g=torch.Generator().manual_seed(36)
    sampled=sample_forest_topologies(h,r,t,root_count=1,count=2,generator=g)
    roots=torch.tensor([[2.,3.]],dtype=torch.float64)
    state=sampled_topology_forest(h,r,t,sampled['samples'][0],root_ids=('known',),root_states=roots)
    assert state.skeleton_origin=='generated'and state.node_ids[0]=='known'
    assert torch.equal(state.states[0],roots[0])and not state.states[1:].any()
    assert state.context_sha256==sampled['context_sha256']and state.coordinate_sha256==COORD
    field=h.field.bind_context(r,t)
    result=sample_flow_forests(state,field,count=2,steps=2,generator=g,field_sha256=field.sha256)
    assert len(result['forests'])==2
    assert all(torch.equal(f.states[0],roots[0])for f in result['forests'])
    assert all(f.skeleton_origin=='generated'for f in result['forests'])


@pytest.mark.parametrize('split',('validation','test'))
def test_heldout_teacher_loss_refuses_before_neural_evaluation(split):
    h=policy();r,t=problem();calls=[];handle=h.register_forward_pre_hook(lambda *a:calls.append(1))
    try:
        with pytest.raises(ValueError):forest_topology_loss(h,r,t,((-1,0),),root_count=1,split=split)
        assert calls==[]
    finally:handle.remove()


@pytest.mark.parametrize('fault',('wrong-root-count','overflow','invalid-parent','bool-action','complete-prefix','bad-count','train-mode'))
def test_preflight_failure_leaves_generator_and_global_rng_unchanged(fault):
    h=policy().eval();r,t=problem();g=torch.Generator().manual_seed(8)
    before=g.get_state().clone();global_before=torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        if fault=='wrong-root-count':topology_log_probability(h,r,t,(-1,-1),root_count=1)
        elif fault=='overflow':topology_log_probability(h,r,t,(-1,0,0,0,0),root_count=1)
        elif fault=='invalid-parent':topology_log_probability(h,r,t,(-1,2),root_count=1)
        elif fault=='bool-action':h(r,t,root_count=1,actions=(True,))
        elif fault=='complete-prefix':h(r,t,root_count=1,actions=(CLOSE,))
        else:
            if fault=='train-mode':h.train()
            sample_forest_topologies(h,r,t,root_count=1,count=True if fault=='bad-count'else 2,generator=g)
    assert torch.equal(before,g.get_state())and torch.equal(global_before,torch.get_rng_state())


def test_failed_sampling_rolls_back_owned_generator(monkeypatch):
    h=policy().eval();r,t=problem();g=torch.Generator().manual_seed(10);before=g.get_state().clone()
    original=h.forward;calls=[]
    def fail(*a,**kw):
        calls.append(1)
        if len(calls)==2:raise ValueError('injected computation refusal')
        return original(*a,**kw)
    monkeypatch.setattr(h,'forward',fail)
    with pytest.raises(ValueError):sample_forest_topologies(h,r,t,root_count=1,count=3,generator=g)
    assert torch.equal(before,g.get_state())


def test_checkpoint_mismatch_rejects_before_any_shared_parameter_copy():
    h=policy();state=copy.deepcopy(h.state_dict());before={n:p.clone()for n,p in h.named_parameters()}
    state['action_readout.weight'].add_(4);state['_extra_state']['spec']['construction_policy_sha256']='b'*64
    with pytest.raises(ValueError):h.load_state_dict(state)
    assert all(torch.equal(before[n],p)for n,p in h.named_parameters())
    clone=copy.deepcopy(h);clone.load_state_dict(copy.deepcopy(h.state_dict()))
    r,t=problem();assert torch.equal(h(r,t,root_count=1,actions=()),clone(r,t,root_count=1,actions=()))
    assert clone.field.geometry.graph.feature is clone.field.geometry._source_backbone_reference[0].feature


def test_full_parameter_role_coverage_and_shared_aliases():
    h=policy();roles=forest_topology_parameter_roles(h,identifier_axis=tuple('id:'+str(i)for i in range(8)))
    assert set(roles)==set(dict(h.named_parameters(remove_duplicate=False)))
    assert roles['action_readout.weight'].role=='output'
    assert roles['structure_projection.weight'].axes[1].kind=='finite'
    combined=torch.nn.ModuleDict({'flow':h.field,'topology':h})
    assert len(tuple(combined.parameters()))<len(tuple(combined.named_parameters(remove_duplicate=False)))
    assert sum(p.numel()for p in h.parameters())-sum(p.numel()for p in h.field.parameters())==82


def test_changed_policy_or_unknown_sample_keys_cannot_be_silently_reused():
    h=policy().eval();r,t=problem();result=sample_forest_topologies(h,r,t,root_count=1,count=1,generator=torch.Generator().manual_seed(1))
    s=dict(result['samples'][0]);s['source_snapshot_sha256']='0'*64
    with pytest.raises(ValueError):sampled_topology_forest(h,r,t,s,root_ids=('root',),root_states=torch.zeros(1,2,dtype=torch.float64))
    with torch.no_grad():h.action_readout.weight.add_(1)
    with pytest.raises(ValueError):sampled_topology_forest(h,r,t,result['samples'][0],root_ids=('root',),root_states=torch.zeros(1,2,dtype=torch.float64))
