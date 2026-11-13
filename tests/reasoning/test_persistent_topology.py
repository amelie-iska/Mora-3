"""Exact small-complex references and independent installed GUDHI oracle."""
import math
import numpy as np
import pytest
import torch
import gudhi
from mora3.reasoning.persistent_topology import rips_persistence,bottleneck_distance,topological_metric_loss


def square():
    x=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
    return np.linalg.norm(x[:,None]-x[None,:],axis=-1)


def test_square_h1_interval_and_essential_h0():
    p=rips_persistence(square())
    h1=p.intervals(1)
    assert len(h1)==1 and h1[0][:2]==pytest.approx((1.,math.sqrt(2)))
    assert h1[0][2]=='finite'
    assert sum(b.death is None for b in p.bars if b.dimension==0)==1
    assert p.coefficient_field==2 and p.complete_through_dimension==1
    assert p.simplex_count==14


def test_filled_triangle_no_positive_h1_and_no_fake_h2():
    p=rips_persistence(np.ones((3,3))-np.eye(3))
    assert p.intervals(1)==()
    with pytest.raises(ValueError):p.intervals(2)


def test_exact_diagrams_against_gudhi_and_permutation():
    rng=np.random.default_rng(20261110)
    for n in range(3,10):
        x=rng.normal(size=(n,3));d=np.linalg.norm(x[:,None]-x[None,:],axis=-1)
        actual=rips_persistence(d)
        st=gudhi.RipsComplex(distance_matrix=d).create_simplex_tree(max_dimension=2)
        st.persistence(homology_coeff_field=2,min_persistence=0,persistence_dim_max=True)
        for dim in (0,1):
            expected=sorted(tuple(row) for row in st.persistence_intervals_in_dimension(dim))
            observed=sorted((a,math.inf if b is None else b)for a,b,_ in actual.intervals(dim))
            np.testing.assert_allclose(observed,expected,rtol=1e-12,atol=1e-12)
        order=rng.permutation(n);other=rips_persistence(d[np.ix_(order,order)])
        assert bottleneck_distance(actual,other,dimension=1)==pytest.approx(0.,abs=1e-12)


def test_truncated_filtration_is_censored_not_essential():
    p=rips_persistence(square(),max_radius=1.1)
    assert p.intervals(1)==((1.,None,'right_censored'),)
    with pytest.raises(ValueError,match='censored'):bottleneck_distance(p,p,dimension=1)


def test_bottleneck_uses_diagonal_and_matches_independent_solver():
    a=rips_persistence(square());b=rips_persistence(square()*1.2)
    x=np.array([(u,v)for u,v,_ in a.intervals(1)])
    y=np.array([(u,v)for u,v,_ in b.intervals(1)])
    assert bottleneck_distance(a,b,dimension=1)==pytest.approx(gudhi.bottleneck_distance(x,y,e=0.),abs=1e-12)
    empty=rips_persistence(np.ones((3,3))-np.eye(3))
    assert bottleneck_distance(a,empty,dimension=1)==pytest.approx((math.sqrt(2)-1)/2)


@pytest.mark.parametrize('change',('asymmetry','negative','nan','diagonal','triangle','boolean','oversize'))
def test_bad_metrics_and_resource_overflow_refuse(change):
    d=square()
    if change=='asymmetry':d[0,1]+=.2
    elif change=='negative':d[0,1]=d[1,0]=-1
    elif change=='nan':d[0,1]=np.nan
    elif change=='diagonal':d[0,0]=.1
    elif change=='triangle':d[0,1]=d[1,0]=10
    elif change=='boolean':d=d.astype(bool)
    else:d=np.zeros((65,65))
    with pytest.raises(ValueError):rips_persistence(d)
    with pytest.raises(ValueError):rips_persistence(square(),max_simplices=5)


def test_piecewise_topology_loss_gradient_and_identity():
    # Distinct edge lengths keep the selected filtration chamber fixed for gradcheck.
    x=torch.tensor([[0.,0.],[1.,.11],[1.4,1.2],[.02,1.5]],dtype=torch.float64)
    dx=torch.cdist(x,x)
    assert topological_metric_loss(dx,dx)['loss'].item()==0
    z=(x*torch.tensor([1.1,.9],dtype=torch.float64)).requires_grad_()
    fn=lambda value:topological_metric_loss(dx,torch.cdist(value,value))['loss']
    assert torch.autograd.gradcheck(fn,(z,),eps=1e-6,atol=1e-5,rtol=1e-4)
    result=fn(z);result.backward();assert torch.isfinite(z.grad).all()and z.grad.abs().sum()>0
    assert topological_metric_loss(dx,dx)['differentiation']=='fixed-pairing-piecewise;ties-not-globally-differentiable'
