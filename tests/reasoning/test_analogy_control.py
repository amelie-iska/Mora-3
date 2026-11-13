"""Real FoT composition, independent structural checks and retrieval leakage rules."""
from dataclasses import replace
import numpy as np
import pytest
import torch
from test_directed_topology import diamond,H
from mora3.reasoning.directed_topology import DirectedFunctor
from mora3.reasoning.persistent_topology import rips_persistence
from mora3.reasoning.analogy_control import (AnalogyDecision,certify_analogy,state_fingerprint,
    grow_analogy_forest,TopologyMemory,rank_topological_memories,functor_transport_loss)


def identity(source,target):
    return DirectedFunctor(source,target,{v:v for v in source.vertices},
        {a.name:(a.name,)for a in source.arrows},{'r':('r',),'s':('s',)},H)


def test_topological_similarity_does_not_override_invalid_directed_functor():
    metric=np.array([[0.,1.],[1.,0.]]);diagram=rips_persistence(metric)
    decision=certify_analogy(identity(diamond(True),diamond(False)),state_sha256='b'*64,
        context_sha256='c'*64,source_topology=diagram,target_topology=diagram,
        topology_tolerance=0.,filtration_contract=('rips-physical','dimensionless','d'*64),max_states=200)
    assert decision.status=='refused' and decision.biological_validity is False
    good=certify_analogy(identity(diamond(True),diamond(True)),state_sha256='b'*64,
        context_sha256='c'*64,source_topology=diagram,target_topology=diagram,
        topology_tolerance=0.,filtration_contract=('rips-physical','dimensionless','d'*64),max_states=200)
    assert good.status=='structurally_admissible'and good.penalty==0


def test_censored_or_proof_budget_exhausted_results_abstain():
    d=rips_persistence(np.array([[0.,1.],[1.,0.]]),max_radius=.5)
    value=certify_analogy(identity(diamond(True),diamond(True)),state_sha256='b'*64,
        context_sha256='c'*64,source_topology=d,target_topology=d,topology_tolerance=0.,
        filtration_contract=('rips-physical','dimensionless','d'*64))
    assert value.status=='unknown'


def test_real_fot_filters_invalid_high_score_and_retains_every_certificate():
    roots=torch.zeros((1,2),dtype=torch.float64);calls=[]
    def expand(state,depth,count):
        assert count==3
        return (torch.tensor([[1.,0.],[2.,0.],[3.,0.]],dtype=state.dtype),
                torch.tensor([1.,100.,50.],dtype=state.dtype),torch.ones(3,dtype=torch.bool),torch.ones(3,dtype=torch.bool))
    def certify(state,depth):
        calls.append(state.clone())
        status='structurally_admissible'if state[0]==1 else'refused'if state[0]==2 else'unknown'
        return AnalogyDecision(status,0.,state_fingerprint(state),'c'*64,'d'*64,'explicit fixture structural check')
    result=grow_analogy_forest(roots,expand,certify,context_sha256='c'*64,branches=3,beam=1,max_depth=1,max_candidates=3)
    leaf=result['nodes'][result['leaves'][0][0]]
    assert leaf['state'][0]==1 and len(calls)==3
    assert len(result['analogy_certificates'])==3
    assert result['training_admission']is False
    assert result['gfn_sampling_kernel']is False


def test_certificate_cannot_be_reused_for_another_state_or_context():
    roots=torch.zeros((1,2),dtype=torch.float64)
    def expand(s,d,n):return torch.ones((n,2),dtype=s.dtype),torch.ones(n,dtype=s.dtype),torch.ones(n,dtype=torch.bool),torch.ones(n,dtype=torch.bool)
    def bad(s,d):return AnalogyDecision('structurally_admissible',0.,'a'*64,'c'*64,'d'*64,'wrong binding')
    with pytest.raises(ValueError):grow_analogy_forest(roots,expand,bad,context_sha256='c'*64,branches=1,beam=1,max_depth=1,max_candidates=1)


def test_memory_retrieval_excludes_test_and_shared_groups_without_promoting_homology_to_truth():
    d=rips_persistence(np.ones((3,3))-np.eye(3))
    query=TopologyMemory('query','condition','validation',('study:q',),'a'*64,'b'*64,'physical-rips/dimensionless',d)
    good=replace(query,identity='good',split='train',groups=('study:a',))
    leak=replace(good,identity='leak',groups=query.groups)
    test=replace(good,identity='test',split='test')
    candidates=rank_topological_memories(query,(good,leak,test),limit=3)
    assert [x['identity']for x in candidates]==['good']
    assert candidates[0]['biological_equivalence_proved']is False
    with pytest.raises(ValueError):rank_topological_memories(query,(replace(good,group_policy_sha256='c'*64),),limit=1)


def test_noncommutative_latent_actions_only_constrained_by_declared_squares():
    x=diamond(True)
    a=torch.tensor([[1.,1.],[0.,1.]],dtype=torch.float64,requires_grad=True)
    b=torch.tensor([[1.,0.],[1.,1.]],dtype=torch.float64,requires_grad=True)
    operators={'a':a,'b':b,'c':b,'d':a}
    result=functor_transport_loss(x,operators)
    assert result['observed_squares']==1 and result['loss'].item()>0
    result['loss'].backward();assert a.grad.abs().sum()>0 and b.grad.abs().sum()>0
    result=functor_transport_loss(diamond(False),operators)
    assert result['observed_squares']==0 and result['loss'].item()==0
    # Matching latent composites can be checked but are not a new biological square.
    operators={'a':a,'b':b,'c':a,'d':b}
    assert functor_transport_loss(x,operators)['loss'].item()==0
