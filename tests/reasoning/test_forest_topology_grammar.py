"""Finite ordered-forest grammar: exact construction proofs checked on small sets."""
from dataclasses import replace
import math
import pytest
from mora3.reasoning.forest_topology import (TopologyBounds, prefix_state,
    canonical_forest_actions, ADD_CHILD, CLOSE)


def leaves(bounds, roots=1):
    pending=[((),1.)];out=[]
    while pending:
        actions,prob=pending.pop();state=prefix_state(bounds,roots,actions)
        if state.complete:out.append((state,prob));continue
        allowed=state.allowed_actions
        # An arbitrary positive state-dependent normalized kernel, not a model fixture result.
        weights={a:(1.+len(actions)/7 if a==ADD_CHILD else 2.)for a in allowed}
        for a in allowed:pending.append((actions+(a,),prob*weights[a]/sum(weights.values())))
    return out


def test_every_bounded_outcome_has_unique_trace_and_normalized_probability():
    b=TopologyBounds(4,3,3)
    result=leaves(b)
    assert len(result)==9
    assert math.isclose(sum(p for _,p in result),1.,rel_tol=0,abs_tol=1e-14)
    assert len({s.parents for s,_ in result})==len(result)
    for state,p in result:
        assert len(state.actions)==2*len(state.parents)-1
        back=canonical_forest_actions(b,state.parents,root_count=1)
        assert back.actions==state.actions and back.parents==state.parents
        assert back.canonical_to_original==tuple(range(len(state.parents)))


@pytest.mark.parametrize('roots',(1,2,3))
def test_multiple_fixed_roots_never_get_generated_or_reordered(roots):
    b=TopologyBounds(4,3,3)
    result=leaves(b,roots)
    assert math.isclose(sum(p for _,p in result),1.,abs_tol=1e-14)
    assert len({s.parents for s,_ in result})==len(result)
    for s,_ in result:
        assert s.parents[:roots]==(-1,)*roots
        assert sum(p==-1 for p in s.parents)==roots
        assert len(s.actions)==2*len(s.parents)-roots


def test_source_canonicalization_keeps_every_original_node_without_future_parent():
    b=TopologyBounds(8,7,7)
    # Source is topologically sorted, but breadth-first and with interspersed roots.
    original=(-1,0,0,-1,1,3,1)
    x=canonical_forest_actions(b,original,root_count=2)
    assert x.canonical_to_original==(0,3,1,4,6,2,5)
    assert x.parents==(-1,-1,0,2,2,0,1)
    inverse={old:new for new,old in enumerate(x.canonical_to_original)}
    assert tuple(-1 if original[old]<0 else inverse[original[old]]for old in x.canonical_to_original)==x.parents
    assert prefix_state(b,2,x.actions).complete


def test_prefix_does_not_depend_on_unseen_future_siblings():
    b=TopologyBounds(6,5,5)
    a=canonical_forest_actions(b,(-1,0),root_count=1)
    c=canonical_forest_actions(b,(-1,0,0,2),root_count=1)
    assert a.actions[:2]==c.actions[:2]
    assert prefix_state(b,1,a.actions[:2])==prefix_state(b,1,c.actions[:2])
    assert prefix_state(b,1,(ADD_CHILD,)).parents==(-1,0)


def test_capacity_is_a_declared_support_boundary_not_truncation():
    b=TopologyBounds(3,1,1)
    assert prefix_state(b,1,()).allowed_actions==(CLOSE,ADD_CHILD)
    child=prefix_state(b,1,(ADD_CHILD,))
    assert child.allowed_actions==(CLOSE,)
    root=prefix_state(b,1,(ADD_CHILD,CLOSE))
    assert root.allowed_actions==(CLOSE,)
    with pytest.raises(ValueError):prefix_state(b,1,(ADD_CHILD,ADD_CHILD))
    with pytest.raises(ValueError):canonical_forest_actions(b,(-1,0,0),root_count=1)
    with pytest.raises(ValueError):canonical_forest_actions(b,(-1,0,1),root_count=1)


@pytest.mark.parametrize('args',((True,2,2),(0,1,1),(513,2,2),(3,-1,1),(3,3,1),(3,1,True),(3,1,3)))
def test_invalid_bounds(args):
    with pytest.raises(ValueError):TopologyBounds(*args)


@pytest.mark.parametrize('roots,actions',((True,()),(0,()),(5,()),(1,[]),(1,(True,)),(1,(2,)),(1,(CLOSE,CLOSE))))
def test_invalid_prefixes(roots,actions):
    with pytest.raises(ValueError):prefix_state(TopologyBounds(4,3,3),roots,actions)


@pytest.mark.parametrize('parents',((0,),(-1,2,0),(-1,False),(),[-1,0],(-2,)))
def test_invalid_original_forest(parents):
    with pytest.raises(ValueError):canonical_forest_actions(TopologyBounds(4,3,3),parents,root_count=1)


def test_source_root_count_cannot_sneak_in_as_unknown_target_condition():
    with pytest.raises(ValueError):canonical_forest_actions(TopologyBounds(4,3,3),(-1,-1),root_count=1)


def test_zero_depth_or_zero_children_only_allows_closing_given_roots():
    for b in(TopologyBounds(4,0,3),TopologyBounds(4,3,0)):
        assert len(leaves(b,2))==1
        assert leaves(b,2)[0][0].actions==(CLOSE,CLOSE)
