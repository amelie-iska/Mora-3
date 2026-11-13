"""Vectorization reference checks only: no optimizer or campaign allocation."""
from dataclasses import replace
import math
import numpy as np
import pytest
import torch
from mora3.reasoning.persistence_vectors import (
    landscape_values, sampled_landscape_vectors, exact_landscape_vectors,
    gaussian_persistence_images, pairwise_l2, native_diagram_batch,
    exact_native_landscape_distance, rank_landscape_memories,
)
from mora3.reasoning.persistent_topology import PersistenceBar, PersistenceResult, rips_persistence
from mora3.reasoning.analogy_control import TopologyMemory


def tensor(rows):
    return torch.tensor(rows, dtype=torch.float64).reshape(1, -1, 2)


def reference_landscape(bars, grid, layers):
    out=np.zeros((layers,len(grid)))
    for j,t in enumerate(grid):
        heights=sorted([max(0.,min(t-b,d-t))for b,d in bars],reverse=True)
        out[:min(len(heights),layers),j]=heights[:layers]
    return out


def diagram(rows, dimension=1):
    bars=tuple(PersistenceBar(dimension,float(b),float(d),'finite',(0,1),(0,1,2))for b,d in rows)
    return PersistenceResult(bars,4,3,4.,True)


def test_vectorized_landscapes_match_independent_order_statistics_and_multiplicity():
    bars=tensor([[0,4],[1,3],[1,3],[2,2.5]])
    grid=torch.linspace(0,4,81,dtype=torch.float64)
    got=landscape_values(bars,torch.ones((1,4),dtype=torch.bool),grid)
    np.testing.assert_allclose(got[0],reference_landscape(bars[0].tolist(),grid.tolist(),4),rtol=0,atol=1e-15)
    assert got.shape==(1,4,81)
    assert torch.equal(got,landscape_values(bars[:,[2,0,3,1]],torch.ones((1,4),dtype=torch.bool),grid))


def test_masked_padding_cannot_add_a_bar():
    bars=tensor([[0,2],[0,0]])
    valid=torch.tensor([[True,False]])
    grid=torch.tensor([0.,1.,2.],dtype=torch.float64)
    got=landscape_values(bars,valid,grid)
    assert torch.equal(got,torch.tensor([[[0.,1.,0.],[0.,0.,0.]]],dtype=torch.float64))


def test_sampled_vector_uses_quadrature_and_declares_compression():
    bars=tensor([[0,2]])
    grid=torch.tensor([0.,1.,2.],dtype=torch.float64)
    out=sampled_landscape_vectors(bars,torch.ones((1,1),dtype=torch.bool),grid)
    assert out['metric']=='sampled-landscape-trapezoidal-L2'
    assert out['lossless'] is False
    assert out['vectors'].square().sum().item()==1.
    # True integral is 2/3, not the sampled trapezoidal value.
    exact=exact_landscape_vectors((bars[0],torch.empty((0,2),dtype=torch.float64)))
    assert pairwise_l2(exact['vectors'])[0,1].square().item()==pytest.approx(2/3,abs=1e-14)


def test_coarse_sampling_can_hide_a_positive_bar_but_exact_path_does_not():
    bars=tensor([[.2,.3]])
    grid=torch.tensor([0.,1.],dtype=torch.float64)
    out=sampled_landscape_vectors(bars,torch.ones((1,1),dtype=torch.bool),grid)
    assert out['vectors'].count_nonzero()==0
    exact=exact_landscape_vectors((bars[0],torch.empty((0,2),dtype=torch.float64)))
    assert pairwise_l2(exact['vectors'])[0,1].item()>0


def test_exact_full_landscape_l2_includes_pair_crossings_and_all_layers():
    a=torch.tensor([[0.,4.],[1.,5.],[.3,2.9]],dtype=torch.float64)
    b=torch.tensor([[.2,4.2],[.7,3.1]],dtype=torch.float64)
    exact=exact_landscape_vectors((a,b))
    t=np.linspace(0,5,100001)
    delta=reference_landscape(a.tolist(),t,3)-reference_landscape(b.tolist(),t,3)
    numerical=np.trapezoid((delta**2).sum(axis=0),t)
    actual=pairwise_l2(exact['vectors'])[0,1].square().item()
    assert actual==pytest.approx(numerical,abs=2e-8)
    assert exact['all_layers_retained'] and exact['integration']=='two-point-Gauss-on-all-piecewise-linear-knots'
    assert any(abs(float(k)-2.5)<1e-14 for k in exact['knots'])


def test_exact_vectors_do_not_change_distance_when_another_diagram_is_added():
    a=tensor([[0,4],[1,3]])[0];b=tensor([[.1,4.2]])[0];c=tensor([[.07,.11],[2.13,8.71]])[0]
    x=pairwise_l2(exact_landscape_vectors((a,b))['vectors'])[0,1]
    y=pairwise_l2(exact_landscape_vectors((a,b,c))['vectors'])[0,1]
    assert x.item()==pytest.approx(y.item(),rel=1e-12)


def test_empty_diagrams_have_well_defined_zero_l2():
    empty=torch.empty((0,2),dtype=torch.float64)
    result=exact_landscape_vectors((empty,empty))
    assert torch.equal(pairwise_l2(result['vectors']),torch.zeros((2,2),dtype=torch.float64))


def test_image_integrates_gaussians_over_pixels_not_centers():
    bars=tensor([[0.,2.],[1.,3.]])
    valid=torch.ones((1,2),dtype=torch.bool)
    x=torch.tensor([-1.,.3,2.],dtype=torch.float64);y=torch.tensor([0.,1.2,3.5],dtype=torch.float64)
    result=gaussian_persistence_images(bars,valid,x,y,bandwidth=(.4,.7))
    expected=np.zeros((2,2))
    cdf=lambda z:.5*math.erfc(-z/math.sqrt(2))
    for b,d in bars[0].tolist():
        life=d-b
        for j in range(2):
            for i in range(2):
                expected[j,i]+=life*(cdf((x[i+1]-b)/.4)-cdf((x[i]-b)/.4))*(cdf((y[j+1]-life)/.7)-cdf((y[j]-life)/.7))
    np.testing.assert_allclose(result['pixel_mass'][0],expected,rtol=2e-14,atol=2e-15)
    assert result['lossless'] is False
    assert result['outside_weighted_mass'][0].item()==pytest.approx(4.-expected.sum(),abs=1e-14)
    area=torch.diff(y)[:,None]*torch.diff(x)[None,:]
    assert torch.allclose(result['density_l2_vectors'],(result['pixel_mass']/area.sqrt()).flatten(1))


def test_images_keep_nonzero_gaussian_tail_without_cdf_cancellation():
    bars=tensor([[0.,1.]])
    r=gaussian_persistence_images(bars,torch.ones((1,1),dtype=torch.bool),
        torch.tensor([8.,9.],dtype=torch.float64),torch.tensor([0.,2.],dtype=torch.float64),bandwidth=(1.,1.))
    assert r['pixel_mass'][0,0,0].item()>0


def test_image_one_pixel_can_conflate_distinct_births():
    bars=torch.tensor([[[.25,1.25]],[[.75,1.75]]],dtype=torch.float64)
    r=gaussian_persistence_images(bars,torch.ones((2,1),dtype=torch.bool),
        torch.tensor([0.,1.],dtype=torch.float64),torch.tensor([0.,2.],dtype=torch.float64),bandwidth=(.2,.2))
    assert r['pixel_mass'][0].item()==pytest.approx(r['pixel_mass'][1].item(),abs=1e-15)
    e=exact_landscape_vectors(tuple(bars))
    assert pairwise_l2(e['vectors'])[0,1].item()>0


def test_gradients_of_fixed_grid_landscapes_and_images():
    bars=tensor([[.12,.83],[.31,1.67]]).requires_grad_()
    mask=torch.ones((1,2),dtype=torch.bool)
    grid=torch.tensor([.03,.24,.51,.78,1.14,1.8],dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda z:landscape_values(z,mask,grid),(bars,),eps=1e-6,atol=1e-5)
    x=torch.tensor([-.1,.5,1.2],dtype=torch.float64);y=torch.tensor([0.,.7,1.9],dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda z:gaussian_persistence_images(z,mask,x,y,bandwidth=(.3,.4))['pixel_mass'],(bars,),eps=1e-6,atol=1e-5)


def test_l2_is_numerically_stable_for_large_common_offsets():
    x=torch.tensor([[1e10,1e10],[1e10+1,1e10+2]],dtype=torch.float64)
    assert pairwise_l2(x)[0,1].item()==pytest.approx(math.sqrt(5),rel=1e-15)
    assert torch.equal(pairwise_l2(x).diag(),torch.zeros(2,dtype=torch.float64))


@pytest.mark.parametrize('bad',[[[2,1]],[[0,float('nan')]],[[0,float('inf')]],[[1,1]]])
def test_invalid_finite_intervals_fail(bad):
    with pytest.raises(ValueError):landscape_values(tensor(bad),torch.ones((1,1),dtype=torch.bool),torch.arange(3,dtype=torch.float64))


def test_capacity_rejects_entire_request_and_cropped_landscape_fails():
    bars=tensor([[0,2],[1,3]])
    mask=torch.ones((1,2),dtype=torch.bool)
    with pytest.raises(ValueError,match='budget'):landscape_values(bars,mask,torch.arange(10,dtype=torch.float64),max_elements=3)
    with pytest.raises(ValueError,match='support'):sampled_landscape_vectors(bars,mask,torch.tensor([.5,2.5],dtype=torch.float64))
    with pytest.raises(ValueError,match='budget'):exact_landscape_vectors((bars[0],),max_elements=2)


def test_native_adapter_retains_all_original_bars_including_essential():
    points=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
    d=np.linalg.norm(points[:,None]-points[None,:],axis=-1)
    result=rips_persistence(d)
    packed=native_diagram_batch((result,),dimension=0)
    assert packed['source_results'][0] is result
    assert len(packed['nonfinite'][0])==1
    assert packed['mask'].sum()==3
    assert packed['source_results'][0].bars==result.bars
    assert exact_native_landscape_distance(result,result)==0.


def test_censored_and_mismatched_essential_bars_are_not_silently_dropped():
    a=diagram([[0,2]])
    censored=replace(a,bars=(PersistenceBar(1,0.,None,'right_censored',(0,1),None),),complete_filtration=False)
    with pytest.raises(ValueError,match='censored'):native_diagram_batch((censored,),dimension=1)
    essential=replace(a,bars=a.bars+(PersistenceBar(1,0.,None,'essential',(0,1),None),))
    with pytest.raises(ValueError,match='essential'):exact_native_landscape_distance(a,essential)


def test_native_memory_l2_ranking_preserves_leakage_rules_and_original_certificate_path():
    def mem(name,d,split='train',groups=None):
        return TopologyMemory(name,'condition',split,groups or (name,), 'a'*64,'b'*64,'physical-nm:v1',d)
    q=mem('q',diagram([[0,2]]),'validation')
    close=mem('close',diagram([[.1,2.1]]));far=mem('far',diagram([[3,5]]))
    heldout=mem('test',q.diagram,'test');samegroup=mem('leak',q.diagram,groups=('q',))
    result=rank_landscape_memories(q,(far,heldout,samegroup,close),limit=2)
    assert [row['identity']for row in result]==['close','far']
    assert all(row['distance_metric']=='exact-finite-landscape-L2'for row in result)
    assert all(row['biological_equivalence_proved'] is False for row in result)
    with pytest.raises(ValueError):rank_landscape_memories(q,(replace(close,filtration_tag='angstrom'),),limit=1)


def test_against_installed_gudhi_landscape_oracle():
    from gudhi.representations import Landscape
    a=np.array([[0.,4.],[1.,2.],[3.,8.],[6.,8.]])
    model=Landscape(num_landscapes=4,resolution=101,sample_range=[0.,8.],keep_endpoints=True)
    expected=model.fit_transform([a]).reshape(4,101)/math.sqrt(2)
    actual=landscape_values(torch.from_numpy(a)[None],torch.ones((1,4),dtype=torch.bool),torch.linspace(0,8,101,dtype=torch.float64))[0]
    np.testing.assert_allclose(actual.numpy(),expected,atol=2e-14,rtol=2e-14)
