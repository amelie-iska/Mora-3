"""Synthetic software references only; no optimizer updates or learned-quality claim."""
from dataclasses import replace
import math
import pytest
import torch
from mora3.reasoning.flow_forest_refinement import (
    ForestState, ForestAssessment, DecodedForest, DecodedAssessment,
    ForestRevision, RefinementPolicy, forest_fingerprint, decoded_fingerprint,
    sample_flow_forests, replace_subtree, run_flow_seeded_refinement,
)

C='a'*64; G='b'*64; E='c'*64; D='d'*64; V='e'*64; F='f'*64


def template():
    return ForestState(('root','left','left-end','right','right-end'),
        (-1,0,1,0,3),('open','open','terminal','open','terminal'),
        torch.tensor([[7.,3.],[0.,0.],[0.,0.],[0.,0.],[0.,0.]],dtype=torch.float64),
        (0,),('root','computational','computational','computational','computational'),C,G)


def assessment(forest,score=1.,status='pass',repairable=False):
    return ForestAssessment(forest_fingerprint(forest),C,E,status,float(score),
        tuple(float(score) for _ in forest.node_ids),F,repairable)


def decoder(forest):
    tokens=tuple(int(round(float(x)*1000))+100000 for x in forest.states.flatten())
    return DecodedForest(tokens,str(tokens),forest_fingerprint(forest),C,D)


def decoded_assessment(forest,decoded,score=1.,status='pass',repairable=False):
    return DecodedAssessment(decoded_fingerprint(decoded),C,V,status,float(score),F,repairable)


def policy(**changes):
    return replace(RefinementPolicy(6,4,2,2,2,22,.5,.5,.7,0.,1),**changes)


def run(**kwargs):
    defaults=dict(steps=3,generator=torch.Generator().manual_seed(17),policy=policy(),
        assess=assessment,edit=lambda *a,**k:(),decode=decoder,assess_decoded=decoded_assessment,
        field_sha256=F,evaluator_sha256=E,decoder_sha256=D,decoded_evaluator_sha256=V)
    defaults.update(kwargs)
    return run_flow_seeded_refinement(template(),lambda x,t,s:torch.zeros_like(x),**defaults)


def test_batched_flow_clamps_known_nodes_and_uses_private_generator():
    torch.manual_seed(123); before=torch.get_rng_state().clone(); seen=[]
    def field(x,t,s):
        seen.append((x.shape,t.clone(),s.states.clone()))
        assert torch.count_nonzero(s.states[1:])==0
        return torch.ones_like(x)
    a=sample_flow_forests(template(),field,count=4,steps=3,generator=torch.Generator().manual_seed(9),field_sha256=F)
    b=sample_flow_forests(template(),field,count=4,steps=3,generator=torch.Generator().manual_seed(9),field_sha256=F)
    assert torch.equal(before,torch.get_rng_state())
    assert a['velocity_evaluations']==3 and a['candidate_equivalent_evaluations']==12
    assert len(a['forests'])==4 and len({forest_fingerprint(x)for x in a['forests']})==4
    assert all(torch.equal(x.states[0],template().states[0]) for x in a['forests'])
    assert [forest_fingerprint(x) for x in a['forests']]==[forest_fingerprint(x)for x in b['forests']]
    assert not a['reward_proportionality_verified'] and not a['gfn_on_policy']


def test_free_template_values_cannot_leak_into_sampling():
    base=template(); changed=base.copy();changed.states[1:]=123456
    def sample(f):return sample_flow_forests(f,lambda x,t,s:s.states.expand_as(x),count=2,steps=2,generator=torch.Generator().manual_seed(3),field_sha256=F)
    assert [forest_fingerprint(f)for f in sample(base)['forests']]==[forest_fingerprint(f)for f in sample(changed)['forests']]


def test_full_trajectory_not_only_endpoint_controls_fingerprint():
    a=template();b=a.copy();b.states[1,0]=1
    assert torch.equal(a.states[2],b.states[2]) and forest_fingerprint(a)!=forest_fingerprint(b)


def test_subtree_replacement_retains_all_other_branches_and_clamps():
    a=template(); before=forest_fingerprint(a)
    b=replace_subtree(a,'left',torch.tensor([[2.,3.],[4.,5.],[6.,7.]],dtype=a.states.dtype),(-1,0,0),('open','terminal','terminal'))
    assert forest_fingerprint(a)==before and len(b.node_ids)==6
    assert all(key in b.node_ids for key in ('root','right','right-end'))
    assert 'left'not in b.node_ids and 'left-end'not in b.node_ids
    assert torch.equal(b.states[b.node_ids.index('right-end')],a.states[4])
    assert b.is_complete
    assert len([s for s in b.statuses if s=='terminal'])==3


def test_subtree_cannot_remove_a_known_condition():
    with pytest.raises(ValueError,match='clamp'):
        replace_subtree(template(),'root',torch.ones((1,2),dtype=torch.float64),(-1,),('terminal',))


def test_two_distinct_decoded_outputs_and_raw_proposals_are_retained():
    out=run()
    assert len(out['outputs'])==2 and len(out['initial_candidate_ids'])==6
    assert len({o['decoded'].semantic_key for o in out['outputs']})==2
    assert out['status']=='qualified-multiple-outputs'
    assert out['gfn_on_policy']is False and out['training_admission']is False


def test_invalid_high_score_is_never_a_success():
    n=[0]
    def score(f):n[0]+=1;return assessment(f,1000 if n[0]==1 else 1,'refused'if n[0]==1 else'pass')
    out=run(assess=score)
    assert len(out['outputs'])==2 and all(x['candidate_id']!=0 for x in out['outputs'])
    assert out['candidates'][0]['assessment'].score==1000


def test_unknown_verdict_and_shortfall_are_not_padded():
    out=run(assess=lambda f:assessment(f,100,'unknown'))
    assert out['outputs']==() and out['status']=='insufficient-qualified-outputs'
    assert out['requested_outputs']==2


def test_low_score_correction_branches_and_preserves_original_elites():
    def score(f):return assessment(f, float(f.states[:,0].max())/10,repairable=True)
    def edit(f,a,**kw):
        return tuple(ForestRevision(replace_subtree(f,'left',torch.tensor([[10.+j,0.],[12.+j,0.]],dtype=f.states.dtype),(-1,0),('open','terminal')),'rule_correction','left')for j in range(kw['count'])) if 'left'in f.node_ids else ()
    out=run(assess=score,edit=edit,policy=policy(correction_threshold=.9))
    assert len(out['outputs'])==2
    assert any(x['parent_id']is not None for x in out['candidates'])
    assert out['candidates'][0]['parent_id']is None
    assert any(e.get('reason')=='correct' for e in out['events'])


def test_bad_edits_cannot_replace_better_originals():
    def score(f):return assessment(f,1. if 'left'in f.node_ids else .1)
    def edit(f,a,**kw):return (ForestRevision(replace_subtree(f,'left',torch.zeros((1,2),dtype=f.states.dtype),(-1,),('terminal',)),'subtree_regeneration','left'),) if 'left'in f.node_ids else ()
    out=run(assess=score,edit=edit)
    assert len(out['outputs'])==2 and all(o['candidate_id']in out['initial_candidate_ids']for o in out['outputs'])


def test_decoding_is_checked_and_duplicate_answers_do_not_fill_output_quota():
    def duplicate(f):return DecodedForest((1,2,3),'same-answer',forest_fingerprint(f),C,D)
    out=run(decode=duplicate)
    assert len(out['outputs'])==1 and out['status']=='insufficient-qualified-outputs'


def test_decode_rejection_is_not_overridden_by_latent_score():
    out=run(assess_decoded=lambda f,d:decoded_assessment(f,d,1000,'refused'))
    assert out['outputs']==()


def test_postdecode_correction_rechecks_the_whole_output():
    def final(f,d):return decoded_assessment(f,d,1.,'pass'if 'left'not in f.node_ids else'refused',True)
    def edit(f,a,**kw):
        if kw['reason']!='post_decode' or 'left'not in f.node_ids:return ()
        assert kw['feedback']['assessment'].status=='refused'
        return (ForestRevision(replace_subtree(f,'left',torch.tensor([[1.,2.]],dtype=f.states.dtype),(-1,),('terminal',)),'post_decode_correction','left'),)
    out=run(edit=edit,assess_decoded=final,policy=policy(edit_rounds=0,post_decode_rounds=1))
    assert len(out['outputs'])==2 and out['post_decode_rounds_completed']==1


def test_candidate_and_round_budgets_are_hard_limits():
    calls=[]
    def edit(f,a,**kw):
        calls.append(kw['count'])
        key=next(k for i,k in enumerate(f.node_ids)if i not in f.clamped)
        return tuple(ForestRevision(replace_subtree(f,key,torch.tensor([[10.+i,2.]],dtype=f.states.dtype),(-1,),('terminal',)),'rule_correction',key)for i in range(kw['count']))
    out=run(edit=edit,policy=policy(total_candidates=8,edit_rounds=5))
    assert out['candidate_attempts']==8 and sum(calls)==2


@pytest.mark.parametrize('change',[
    {'return_outputs':1},{'initial_candidates':1},{'active_candidates':1},
    {'edit_rounds':-1},{'total_candidates':2},{'minimum_score':float('nan')},
    {'minimum_distance':-1.},{'initial_candidates':True},
])
def test_bad_policy_refuses_before_callbacks(change):
    with pytest.raises(ValueError):policy(**change)


def test_stale_evaluator_certificate_is_rejected():
    with pytest.raises(ValueError,match='bind'):
        run(assess=lambda f:replace(assessment(f),candidate_sha256='0'*64))


def test_wrong_decode_revision_is_rejected():
    with pytest.raises(ValueError,match='decoder'):
        run(decode=lambda f:replace(decoder(f),decoder_sha256='0'*64))


def test_wrong_decoded_evidence_is_rejected():
    with pytest.raises(ValueError,match='decoded'):
        run(assess_decoded=lambda f,d:replace(decoded_assessment(f,d),decoded_sha256='0'*64))


def test_scorer_mutation_is_detected_without_touching_original():
    def score(f):f.states.add_(1);return assessment(f)
    with pytest.raises(ValueError,match='mutat'):run(assess=score)
    assert torch.equal(template().states[0],torch.tensor([7.,3.],dtype=torch.float64))


def test_editing_unrelated_branch_is_rejected():
    def edit(f,a,**kw):
        x=replace_subtree(f,'left',torch.zeros((1,2),dtype=f.states.dtype),(-1,),('terminal',))
        x.states[x.node_ids.index('right-end')]+=1
        return (ForestRevision(x,'rule_correction','left'),)
    with pytest.raises(ValueError,match='unrelated'):run(edit=edit)


def test_censored_leaf_is_not_a_completed_output():
    t=replace(template(),statuses=('open','open','censored','open','terminal'))
    out=run_flow_seeded_refinement(t,lambda x,t,s:torch.zeros_like(x),steps=1,generator=torch.Generator().manual_seed(3),policy=policy(),assess=assessment,edit=lambda*a,**k:(),decode=decoder,assess_decoded=decoded_assessment,field_sha256=F,evaluator_sha256=E,decoder_sha256=D,decoded_evaluator_sha256=V)
    assert out['outputs']==()


def test_identical_tokens_with_different_semantic_keys_still_deduplicate():
    def dec(f):return DecodedForest((1,2),forest_fingerprint(f),forest_fingerprint(f),C,D)
    assert len(run(decode=dec)['outputs'])==1


def test_full_forest_diversity_changes_selection_not_scores():
    out=run(policy=policy(minimum_distance=1e9))
    assert len(out['outputs'])==1
    assert all(r['assessment'].score==1 for r in out['candidates'])


def test_initial_noise_and_complete_state_are_not_modified_by_flow():
    t=template();before=forest_fingerprint(t)
    sample_flow_forests(t,lambda x,t,s:torch.zeros_like(x),count=2,steps=1,generator=torch.Generator().manual_seed(1),field_sha256=F)
    assert forest_fingerprint(t)==before


def test_equivalent_subtree_revision_cannot_evade_l2_diversity_by_renaming():
    from mora3.reasoning.flow_forest_refinement import _distance
    original=template()
    edited=replace_subtree(original,'left',original.states[1:3],(-1,0),('open','terminal'))
    assert edited.parents==original.parents
    assert torch.equal(edited.states,original.states)
    assert _distance(original,edited,None)==0.


def test_actual_module_field_replays_without_updating_weights():
    from torch import nn
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42);head=nn.Linear(3,2,dtype=torch.float64)
    original={k:v.detach().clone()for k,v in head.state_dict().items()}
    def field(x,t,condition):
        return head(torch.cat((x,t[:,None,None].expand(x.shape[0],x.shape[1],1)),dim=-1))
    result=run_flow_seeded_refinement(template(),field,steps=4,generator=torch.Generator().manual_seed(17),
        policy=policy(edit_rounds=0),assess=assessment,edit=lambda*a,**k:(),decode=decoder,
        assess_decoded=decoded_assessment,field_sha256=F,evaluator_sha256=E,decoder_sha256=D,
        decoded_evaluator_sha256=V)
    assert len(result['outputs'])==2 and result['velocity_evaluations']==4
    assert all(torch.equal(v,head.state_dict()[k])for k,v in original.items())
    assert all(p.grad is None for p in head.parameters())


@pytest.mark.parametrize('corruption',['cycle','parent_future','hidden_target_skeleton','clamp_missing','terminal_parent'])
def test_forest_graph_invariants(corruption):
    f=template();bad={
        'cycle':dict(parents=(-1,2,1,0,3)),
        'parent_future':dict(parents=(-1,4,1,0,3)),
        'hidden_target_skeleton':dict(skeleton_origin='target'),
        'clamp_missing':dict(clamped=()),
        'terminal_parent':dict(statuses=('terminal','open','terminal','open','terminal')),
    }[corruption]
    with pytest.raises(ValueError):replace(f,**bad)


def test_postdecode_editor_cannot_overwrite_retained_failure_evidence():
    from types import MappingProxyType
    seen=[]
    def final(f,d):return decoded_assessment(f,d,1.,'refused',True)
    def edit(f,a,**kw):
        feedback=kw['feedback'];seen.append(feedback)
        assert type(feedback)is MappingProxyType
        with pytest.raises(TypeError):feedback['assessment']=replace(feedback['assessment'],status='pass')
        return ()
    out=run(edit=edit,assess_decoded=final,policy=policy(edit_rounds=0,post_decode_rounds=1))
    assert seen and out['outputs']==()
    assert all(d['assessment'].status=='refused'for d in out['decoded_candidates'])


def test_incomplete_neural_decode_is_retained_and_can_trigger_new_edits():
    from mora3.reasoning.flow_forest_refinement import DecodeRefusal
    seen=[]
    def incomplete(f):return DecodeRefusal(forest_fingerprint(f),C,D,'missing-stop',F,True)
    def edit(f,a,**kw):
        assert kw['reason']=='post_decode'and kw['feedback']['assessment']is None
        assert type(kw['feedback']['failure'])is DecodeRefusal
        seen.append(kw['feedback']['failure'].reason)
        return ()
    out=run(decode=incomplete,edit=edit,assess_decoded=lambda *a:pytest.fail('no completed output to certify'),
        policy=policy(edit_rounds=0,post_decode_rounds=1))
    assert seen and out['outputs']==()
    assert all(x['decoded']is None and x['assessment']is None for x in out['decoded_candidates'])
    assert out['inference_counts']['decoded_assessments']==0
    assert any(e['kind']=='decode_refused'for e in out['events'])


def test_decode_refusal_cannot_use_stale_candidate_or_decoder_bindings():
    from mora3.reasoning.flow_forest_refinement import DecodeRefusal
    with pytest.raises(ValueError,match='bind'):
        run(decode=lambda f:DecodeRefusal('0'*64,C,D,'missing-stop',F,True),policy=policy(edit_rounds=0))
