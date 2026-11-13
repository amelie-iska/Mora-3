"""Complete reasoning graphs: prefix causality is not biological causation."""
from dataclasses import replace
import json
import pytest
from mora3.reasoning.trajectory_contracts import (
    TraceFact, TraceNode, TraceStep, ReasoningTrace, prefix_record,
    prefix_attention_mask, check_trace_partitions, verify_trajectory_analogy,
)
from mora3.reasoning.directed_topology import DirectedFunctor, Square

H = 'a'*64


def branching_trace(split='train', group='problem:one'):
    facts = (
        TraceFact('question', '{"constraint":2}', 'condition', None, (), 'observed', H),
        TraceFact('left_result', '{"estimate":3}', 'derived', 'l', ('question',), 'model_generated', H),
        TraceFact('right_result', '{"estimate":4}', 'derived', 'r', ('question',), 'model_generated', H),
        TraceFact('answer', '{"gold":7}', 'target', 'end', (), 'observed', H),
    )
    nodes = (TraceNode('root',0,('question',),'open'),
             TraceNode('l',1,('question','left_result'),'open'),
             TraceNode('r',1,('question','right_result'),'open'),
             TraceNode('left_end',2,('question','left_result'),'terminal'),
             TraceNode('end',2,('question','right_result'),'terminal'))
    edges = (TraceStep('a','root','l','derive-left','computational',('question',),H),
             TraceStep('b','root','r','derive-right','computational',('question',),H),
             TraceStep('c','l','left_end','finish-left','computational',('left_result',),H),
             TraceStep('d','r','end','finish-right','computational',('right_result',),H))
    return ReasoningTrace('trace-a',H,'b'*64,'c'*64,split,(group,),nodes,edges,facts,())


def test_complete_path_enumeration_and_branch_local_prefix():
    trace = branching_trace()
    assert trace.complete_paths() == (('a','c'), ('b','d'))
    prefix = prefix_record(trace, ('a',))
    assert prefix['state']['layer'] == 1
    names = [f['name'] for f in prefix['state']['facts']]
    assert names == ['question','left_result']
    wire = json.dumps(prefix['state'], sort_keys=True)
    assert 'right_result' not in wire and 'gold' not in wire and 'trace-a' not in wire
    assert prefix['biological_causality_verified'] is False
    assert prefix['direction_evidence'] == ['computational']


def test_gold_and_unvisited_branch_edits_do_not_change_prefix_features():
    trace = branching_trace()
    facts = tuple(replace(f,value_json='{"changed":999}') if f.name in ('answer','right_result') else f for f in trace.facts)
    altered = replace(trace,identity='another-audit-id',facts=facts)
    a,b = prefix_record(trace,('a',)),prefix_record(altered,('a',))
    assert a['state'] == b['state']
    assert a['condition_sha256'] == b['condition_sha256']


def test_attention_mask_is_ancestral_not_bidirectional_or_cross_branch():
    t = branching_trace(); mask = prefix_attention_mask(t)
    assert mask[1][0] and mask[1][1]
    assert not mask[1][2] and not mask[0][1]
    assert mask[3][1] and not mask[3][2]


@pytest.mark.parametrize('problem',('future','sibling','target','dependency','backward','causal-upgrade'))
def test_invalid_source_dependencies_and_causal_claims_are_refused(problem):
    t = branching_trace()
    if problem in ('future','sibling','target'):
        forbidden = {'future':'left_result','sibling':'right_result','target':'answer'}[problem]
        i = 0 if problem=='future' else 1
        nodes = list(t.nodes);nodes[i] = replace(nodes[i],visible_facts=nodes[i].visible_facts+(forbidden,))
        with pytest.raises(ValueError):replace(t,nodes=tuple(nodes))
    elif problem=='dependency':
        facts = tuple(replace(f,depends_on=('right_result',)) if f.name=='left_result' else f for f in t.facts)
        with pytest.raises(ValueError):replace(t,facts=facts)
    elif problem=='backward':
        edges = (replace(t.steps[0],source='l',target='root'),*t.steps[1:])
        with pytest.raises(ValueError):replace(t,steps=edges)
    else:
        with pytest.raises(ValueError):replace(t.steps[0],direction_kind='interventional')


def test_censored_leaf_is_retained_but_never_a_completed_training_trajectory():
    t=branching_trace();nodes=tuple(replace(n,status='censored')if n.key=='end'else n for n in t.nodes)
    t=replace(t,nodes=nodes)
    assert t.complete_paths()==(('a','c'),)
    assert t.maximal_paths()==((('a','c'),'terminal'),(('b','d'),'censored'))
    with pytest.raises(ValueError):prefix_record(t,('a','d'))
    with pytest.raises(ValueError,match='budget'):t.complete_paths(max_paths=1)


def test_group_closure_rejects_cross_split_problem_or_source_overlap():
    t=branching_trace();other=replace(t,identity='other',split='validation')
    with pytest.raises(ValueError,match='group'):check_trace_partitions((t,other))
    good=replace(other,groups=('problem:two',))
    assert check_trace_partitions((t,good))['traces']==2


def test_whole_trajectory_map_checks_every_branch_and_terminal():
    source=branching_trace();target=replace(source,identity='target',split='validation',groups=('problem:target',))
    sc,tc=source.directed_complex(),target.directed_complex()
    f=DirectedFunctor(sc,tc,{n.key:n.key for n in source.nodes},
        {e.key:(e.key,)for e in source.steps},{e.action:(e.action,)for e in source.steps},H)
    result=verify_trajectory_analogy(source,target,f)
    assert result['status']=='proved-in-declared-presentations'
    assert result['checked_complete_source_paths']==2 and result['checked_source_steps']==4
    assert result['endpoint_only_comparison'] is False
    assert result['biological_causality_transferred'] is False
    with pytest.raises(ValueError):verify_trajectory_analogy(source,target,f,max_paths=1)


def test_merging_histories_requires_explicit_state_equivalence_and_no_branch_fact_leakage():
    t=branching_trace()
    nodes=(t.nodes[0],TraceNode('l',1,('question',),'open'),TraceNode('r',1,('question',),'open'),
           TraceNode('end',2,('question',),'terminal',merge_evidence_sha256=H))
    steps=(t.steps[0],t.steps[1],replace(t.steps[2],target='end',requires=('question',)),
           replace(t.steps[3],requires=('question',)))
    merged=replace(t,nodes=nodes,steps=steps,facts=(t.facts[0],),
        squares=(Square(('a','c'),('b','d'),H),))
    assert len(merged.complete_paths())==2
    with pytest.raises(ValueError):replace(merged,nodes=(*nodes[:-1],replace(nodes[-1],merge_evidence_sha256=None)))
    # Merely close embeddings would not be a valid merge certificate either.
    assert merged.directed_complex().equivalent('root',('a','c'),('b','d')).status=='proved'


def test_multistage_analogy_checks_eight_six_step_paths_and_every_commuting_square():
    nodes=[TraceNode('v0',0,('question',),'open')];steps=[];squares=[]
    for stage in range(3):
        root=f'v{stage}';end=f'v{stage+1}';left=f'l{stage}';right=f'r{stage}'
        nodes.extend((TraceNode(left,2*stage+1,('question',),'open'),
                      TraceNode(right,2*stage+1,('question',),'open'),
                      TraceNode(end,2*stage+2,('question',),'terminal'if stage==2 else'open',H)))
        a,b,c,d=(f'{letter}{stage}'for letter in 'abcd')
        steps.extend((TraceStep(a,root,left,'subgoal-A','computational',('question',),H),
                      TraceStep(b,left,end,'subgoal-B','computational',('question',),H),
                      TraceStep(c,root,right,'subgoal-B','computational',('question',),H),
                      TraceStep(d,right,end,'subgoal-A','computational',('question',),H)))
        squares.append(Square((a,b),(c,d),H))
    source=ReasoningTrace('six-step-source',H,'b'*64,'c'*64,'train',('source-problem',),
        tuple(nodes),tuple(steps),(TraceFact('question','{"goal":"compose-three-subproblems"}','condition',None,(),'synthetic_fixture',H),),tuple(squares))
    target=replace(source,identity='six-step-target',split='validation',groups=('target-problem',))
    def mapping(target):
        return DirectedFunctor(source.directed_complex(),target.directed_complex(),{n.key:n.key for n in nodes},
            {e.key:(e.key,)for e in steps},{'subgoal-A':('subgoal-A',),'subgoal-B':('subgoal-B',)},H)
    result=verify_trajectory_analogy(source,target,mapping(target))
    assert result['checked_complete_source_paths']==8
    assert result['checked_source_steps']==12
    assert all(len(row['source_path'])==6 for row in result['mapped_trajectories'])
    missing=replace(target,squares=target.squares[:-1])
    assert verify_trajectory_analogy(source,missing,mapping(missing))['status']=='refused'
