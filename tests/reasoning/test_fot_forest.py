import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import pytest
import torch


def run(*args, **kwargs):
    from mora3.reasoning.fot_forest import grow_forest
    return grow_forest(*args, **kwargs)


def expand(parent, depth, count):
    x = parent[None, :] + torch.arange(1, count+1, device=parent.device, dtype=parent.dtype)[:, None]
    return x, x[:, 0].clone(), torch.ones(count, dtype=torch.bool, device=x.device), torch.zeros(count, dtype=torch.bool, device=x.device)


def test_per_tree_top_m_and_parent_paths():
    roots = torch.tensor([[0.], [100.]], requires_grad=True)
    r = run(roots, expand, branches=3, beam=2, max_depth=2, max_candidates=20)
    assert r['reasons'] == ['depth', 'depth']
    assert r['counts'] == [9, 9]
    for tree, leaves in enumerate(r['leaves']):
        assert [r['nodes'][i]['state'].item() for i in leaves] == [6+100*tree, 5+100*tree]
        for i in leaves:
            n = r['nodes'][i]; p = r['nodes'][n['parent']]
            assert n['tree'] == p['tree'] == tree and n['depth'] == p['depth']+1
            assert r['nodes'][p['parent']]['parent'] is None
    assert not r['nodes'][-1]['state'].requires_grad
    assert roots.grad is None


def test_budget_is_hard_bound_with_partial_parent_expansion():
    calls = []
    def tracked(p, d, n):
        calls.append(n)
        return expand(p, d, n)
    r = run(torch.zeros(1, 1), tracked, branches=3, beam=2, max_depth=8, max_candidates=5)
    assert calls == [3, 2]
    assert r['counts'] == [5] and r['reasons'] == ['budget']
    assert len(r['nodes']) == 6


def test_invalid_tree_and_terminal_branch_are_not_expanded():
    def callback(p, d, n):
        x,s,v,t = expand(p,d,n)
        v[:] = p[0] >= 0
        t[:] = True
        return x,s,v,t
    r = run(torch.tensor([[-10.],[0.]]), callback, branches=2, beam=2, max_depth=4, max_candidates=20)
    assert r['reasons'] == ['invalid', 'terminal']
    assert r['leaves'][0] == [] and len(r['leaves'][1]) == 2
    assert r['counts'] == [2,2]


def test_terminal_leaves_survive_other_branch_failure():
    def callback(p,d,n):
        x,s,v,t=expand(p,d,n)
        if d==1: t[0]=True
        else: v[:]=False
        return x,s,v,t
    r=run(torch.zeros(1,1),callback,branches=2,beam=2,max_depth=3,max_candidates=20)
    assert r['reasons']==['invalid']
    assert len(r['leaves'][0])==1
    assert r['nodes'][r['leaves'][0][0]]['terminal']


def test_distance_and_stable_ties():
    def callback(p,d,n):
        x,s,v,t=expand(p,d,n);s[:]=1
        return x,s,v,t
    r=run(torch.zeros(1,1),callback,branches=4,beam=4,max_depth=1,max_candidates=10,min_distance=2.)
    assert [r['nodes'][i]['state'].item() for i in r['leaves'][0]]==[1.,3.]


@pytest.mark.parametrize('kw', [{'branches':0},{'beam':True},{'max_depth':0},{'max_candidates':-1},{'min_distance':float('nan')}])
def test_invalid_config_before_callback(kw):
    def bad(*a): raise AssertionError('called')
    args=dict(branches=2,beam=2,max_depth=2,max_candidates=10);args.update(kw)
    with pytest.raises(ValueError):run(torch.zeros(1,1),bad,**args)


def test_invalid_callback_and_roots():
    args=dict(branches=2,beam=2,max_depth=2,max_candidates=10)
    with pytest.raises(ValueError):run(torch.empty(0,1),expand,**args)
    def bad(p,d,n):
        x,s,v,t=expand(p,d,n);x[0]=float('nan');v[0]=False
        return x,s,v,t
    with pytest.raises(ValueError):run(torch.zeros(1,1),bad,**args)
    def bad_shape(p,d,n):return expand(p,d,n+1)
    with pytest.raises(ValueError):run(torch.zeros(1,1),bad_shape,**args)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_gpu_matches_cpu():
    args=dict(branches=3,beam=2,max_depth=3,max_candidates=14,min_distance=1.)
    a=run(torch.zeros(2,2,dtype=torch.double),expand,**args)
    b=run(torch.zeros(2,2,dtype=torch.double,device='cuda'),expand,**args)
    assert a['leaves']==b['leaves'] and a['reasons']==b['reasons'] and a['counts']==b['counts']
    for x,y in zip(a['nodes'],b['nodes']):
        assert torch.equal(x['state'],y['state'].cpu())
        assert {k:v for k,v in x.items() if k!='state'}=={k:v for k,v in y.items() if k!='state'}
