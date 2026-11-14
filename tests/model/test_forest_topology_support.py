"""A normalized topology policy must fit its entire reachable native support."""
import pytest
import torch
from test_forest_flow import make,problem,SOURCE,COORD
from mora3.model.forest_flow import ForestFlowSpec
from mora3.model.forest_topology import ForestTopologySpec,NativeForestTopologyHead,sample_forest_topologies
from mora3.reasoning.forest_topology import TopologyBounds,prefix_state,CLOSE,ADD_CHILD


def head(bounds,budget):
    flow=make(spec=ForestFlowSpec(dimension=2,source_snapshot_sha256=SOURCE,
        coordinate_sha256=COORD,max_nodes=8,max_batch_size=8,max_attention_elements=budget))
    return NativeForestTopologyHead(flow,ForestTopologySpec(bounds,'a'*64)).eval()


def test_full_grammar_overflow_refuses_before_any_neural_call_or_rng():
    h=head(TopologyBounds(8,7,7),64);r,t=problem();g=torch.Generator().manual_seed(4)
    owned=g.get_state().clone();global_rng=torch.get_rng_state().clone();calls=[]
    hook=h.field.geometry.register_forward_pre_hook(lambda *a:calls.append(True))
    try:
        with pytest.raises(ValueError,match='support'):
            h.preflight(r,t,root_count=1,actions=())
        with pytest.raises(ValueError,match='support'):
            sample_forest_topologies(h,r,t,root_count=1,count=1,generator=g)
        assert calls==[]
        assert torch.equal(g.get_state(),owned)and torch.equal(torch.get_rng_state(),global_rng)
    finally:hook.remove()


@pytest.mark.parametrize('bounds',(TopologyBounds(8,0,7),TopologyBounds(8,7,0)))
def test_unreachable_node_capacity_does_not_cause_false_refusal(bounds):
    h=head(bounds,32);r,t=problem();g=torch.Generator().manual_seed(6)
    s=sample_forest_topologies(h,r,t,root_count=1,count=2,generator=g)
    assert all(x['parents']==[-1]and x['actions']==[CLOSE]and x['log_probability']==0. for x in s['samples'])


def test_exact_multiroot_support_boundary_and_checkpoint_policy():
    # Two depth-one roots can each add one child: Nmax=4, tokens=1+8-2=7.
    r,t=problem();bounds=TopologyBounds(8,1,1)
    good=head(bounds,98);bad=head(bounds,97)
    good.preflight(r,t,root_count=2,actions=())
    with pytest.raises(ValueError,match='support'):bad.preflight(r,t,root_count=2,actions=())
    config=good.get_extra_state()
    assert config['schema']=='Mora3.NativeForestTopology/v2'
    assert config['resource_policy']=='complete-reachable-grammar-attention-v1'


def test_effective_support_bound_matches_independent_enumeration():
    from mora3.reasoning.forest_topology import maximum_forest_nodes
    for nodes in range(1,7):
        for roots in range(1,nodes+1):
            for depth in range(nodes):
                for children in range(nodes):
                    b=TopologyBounds(nodes,depth,children);pending=[()];maximum=roots
                    while pending:
                        actions=pending.pop();s=prefix_state(b,roots,actions)
                        if s.complete:maximum=max(maximum,len(s.parents))
                        else:pending.extend(actions+(x,)for x in s.allowed_actions)
                    assert maximum_forest_nodes(b,roots)==maximum


@pytest.mark.parametrize('roots',(True,0,9,-1,1.0))
def test_effective_support_rejects_invalid_roots(roots):
    from mora3.reasoning.forest_topology import maximum_forest_nodes
    with pytest.raises(ValueError):maximum_forest_nodes(TopologyBounds(8,7,7),roots)


def test_older_partial_support_checkpoint_refuses_before_tensor_copy():
    import copy
    h=head(TopologyBounds(4,3,3),100000)
    before={n:p.detach().clone()for n,p in h.named_parameters()}
    state=copy.deepcopy(h.state_dict());state['action_readout.weight'].add_(7)
    state['_extra_state']['schema']='Mora3.NativeForestTopology/v1'
    state['_extra_state'].pop('resource_policy')
    with pytest.raises(ValueError):h.load_state_dict(state)
    assert all(torch.equal(before[n],p)for n,p in h.named_parameters())
