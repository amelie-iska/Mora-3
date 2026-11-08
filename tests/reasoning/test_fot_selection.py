"""Selection oracle: top score followed by hard Euclidean separation.
Removing separation, eligibility, stable ties or greedy order breaks these tests.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
import pytest
import torch


def select(*args, **kwargs):
    from mora3.reasoning.fot_selection import select_branches
    return select_branches(*args, **kwargs)


def test_score_order_diversity_and_ties():
    x=torch.tensor([[0.], [.1], [3.], [6.]],dtype=torch.double)
    score=torch.tensor([4.,3.,2.,2.],dtype=torch.double)
    assert select(x,score,4,min_distance=1.).tolist()==[0,2,3]
    assert select(x,score,2).tolist()==[0,1]
    assert select(x,score,4,min_distance=3.).tolist()==[0,2,3]


def test_eligibility_empty_and_no_mutation():
    x=torch.tensor([[0.],[1.],[2.]])
    s=torch.tensor([3.,2.,1.]);before=x.clone()
    assert select(x,s,3,eligible=torch.tensor([False,True,True])).tolist()==[1,2]
    assert select(x,s,3,eligible=torch.zeros(3,dtype=torch.bool)).shape==(0,)
    assert select(x[:0],s[:0],2).numel()==0
    assert torch.equal(x,before)


def test_scalar_greedy_oracle_and_detached_output():
    g=torch.Generator().manual_seed(19)
    x=torch.randn(25,3,generator=g,dtype=torch.double,requires_grad=True)
    s=torch.randn(25,generator=g,dtype=torch.double,requires_grad=True)
    expected=[]
    for i in sorted(range(25),key=lambda i:(-s[i].item(),i)):
        if all(sum((x[i,k].item()-x[j,k].item())**2 for k in range(3))>=1.2**2 for j in expected):
            expected.append(i)
        if len(expected)==7:break
    actual=select(x,s,7,min_distance=1.2)
    assert actual.tolist()==expected
    x[actual].sum().backward()
    assert torch.equal(x.grad.sum(1)>0,torch.isin(torch.arange(25),actual))
    assert s.grad is None


@pytest.mark.parametrize('change', ['nan','score_shape','dtype','budget','distance','eligible','width'])
def test_contracts(change):
    x=torch.zeros(3,2);s=torch.ones(3);kw={}
    budget=2
    if change=='nan':x[0,0]=float('nan')
    if change=='score_shape':s=s[:,None]
    if change=='dtype':s=s.double()
    if change=='budget':budget=True
    if change=='distance':kw['min_distance']=-1
    if change=='eligible':kw['eligible']=torch.ones(3)
    if change=='width':x=x[:,:0]
    with pytest.raises(ValueError):select(x,s,budget,**kw)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_gpu_matches_cpu():
    torch.manual_seed(19)
    x=torch.randn(100,8,dtype=torch.double);s=torch.randn(100,dtype=torch.double)
    expected=select(x,s,12,min_distance=2.)
    actual=select(x.cuda(),s.cuda(),12,min_distance=2.)
    assert actual.device.type=='cuda'
    assert torch.equal(actual.cpu(),expected)
