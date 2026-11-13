"""History-aware native FoT: equal endpoints do not erase distinct reasoning paths."""
import torch
import pytest
from mora3.reasoning.fot_forest import grow_forest
from mora3.reasoning.trajectory_forest import grow_trajectory_forest,history_fingerprint,TrajectoryDecision

H='a'*64


def test_existing_forest_exposes_cloned_complete_ancestry_without_changing_default():
    roots=torch.zeros((1,2),dtype=torch.float64); seen=[]
    def expand(state,depth,count):
        return state[None].repeat(count,1)+depth,torch.ones(count,dtype=state.dtype),torch.ones(count,dtype=torch.bool),torch.full((count,),depth==3,dtype=torch.bool)
    def history_expand(history,depth,count):
        seen.append(tuple(n['depth']for n in history));state=history[-1]['state'].clone()
        history[0]['state'].add_(999)
        return expand(state,depth,count)
    basic=grow_forest(roots,expand,branches=1,beam=1,max_depth=3,max_candidates=3)
    rich=grow_forest(roots,expand,history_expand=history_expand,branches=1,beam=1,max_depth=3,max_candidates=3)
    assert seen==[(0,),(0,1),(0,1,2)]
    assert all(torch.equal(x['state'],y['state'])for x,y in zip(basic['nodes'],rich['nodes']))
    assert rich['nodes'][0]['state'].sum()==0


def test_full_path_verifier_distinguishes_equal_endpoints_and_keeps_all_proposals():
    roots=torch.zeros((1,1),dtype=torch.float64)
    def expand(history,depth,count):
        if depth==1:
            x=torch.tensor([[1.],[-1.]],dtype=roots.dtype)[:count];score=torch.zeros(count,dtype=roots.dtype)
        else:
            x=torch.zeros((count,1),dtype=roots.dtype)
            score=torch.full((count,),100. if history[-1]['state'][0]<0 else 1.,dtype=roots.dtype)
        return x,score,torch.ones(count,dtype=torch.bool),torch.full((count,),depth==2,dtype=torch.bool)
    seen=[]
    def certify(history):
        seen.append(history_fingerprint(history))
        valid=len(history)<3 or history[1]['state'][0]>0
        return TrajectoryDecision('admissible'if valid else'refused',history_fingerprint(history),H,'b'*64)
    out=grow_trajectory_forest(roots,expand,certify,context_sha256=H,branches=2,beam=2,max_depth=2,max_candidates=6)
    complete=out['complete_paths'][0]
    assert len(complete)==2
    assert all(out['nodes'][path[1]]['state'][0]>0 for path in complete)
    assert len(out['trajectory_certificates'])==6
    assert out['gfn_on_policy']is False
    assert len(set(seen))==6


def test_truncated_leaf_never_becomes_a_completed_gfn_trajectory():
    roots=torch.zeros((1,1),dtype=torch.float64)
    def expand(h,d,n):return torch.ones((n,1),dtype=roots.dtype)*d,torch.zeros(n,dtype=roots.dtype),torch.ones(n,dtype=torch.bool),torch.zeros(n,dtype=torch.bool)
    def certify(h):return TrajectoryDecision('admissible',history_fingerprint(h),H,'b'*64)
    out=grow_trajectory_forest(roots,expand,certify,context_sha256=H,branches=1,beam=1,max_depth=2,max_candidates=2)
    assert out['complete_paths']==((),)
    assert len(out['censored_paths'][0])==1
    assert out['nodes'][out['censored_paths'][0][0][-1]]['terminal']is False


def test_past_history_or_context_certificate_cannot_be_reused():
    roots=torch.zeros((1,1),dtype=torch.float64)
    def expand(h,d,n):return torch.ones((n,1),dtype=roots.dtype),torch.zeros(n,dtype=roots.dtype),torch.ones(n,dtype=torch.bool),torch.ones(n,dtype=torch.bool)
    def stale(h):return TrajectoryDecision('admissible','f'*64,H,'b'*64)
    with pytest.raises(ValueError):grow_trajectory_forest(roots,expand,stale,context_sha256=H,branches=1,beam=1,max_depth=1,max_candidates=1)
