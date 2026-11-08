import pytest
import torch
from mora3.model.response_quality import response_quality


def test_unequal_groups_and_pythagorean_error_decomposition():
    # Group means (2,4) and (10,8); within SSE=4, total entries=6.
    y=torch.tensor([[1.,3.],[3.,5.],[10.,8.]],dtype=torch.float64)
    p=torch.tensor([[4.,3.],[4.,3.],[9.,6.]],dtype=torch.float64)
    r=response_quality(p,y,['a','a','b'])
    assert r['context_mean_mse']==pytest.approx(4/6)
    assert r['prediction_mse']==pytest.approx(19/6)
    assert r['excess_over_context_mean_mse']==pytest.approx(15/6)
    assert r['gene_mean_mse']==pytest.approx(((y-y.mean(0))**2).mean().item())
    assert r['groups']==2 and r['singleton_groups']==1
    assert r['training_gate_passed']
    order=torch.tensor([2,0,1]);s=response_quality(p[order],y[order],['b','a','a'])
    assert s==r


def test_equal_observed_inputs_require_equal_predictions():
    with pytest.raises(ValueError,match='deterministic'):
        response_quality(torch.tensor([[1.],[2.]]),torch.ones(2,1),['a','a'])


@pytest.mark.parametrize('kind',['empty','shape','nan','keys','dtype'])
def test_invalid_quality_inputs_fail_closed(kind):
    p=torch.ones(2,3);y=p.clone();keys=['a','b']
    if kind=='empty':p=y=torch.empty(0,3);keys=[]
    if kind=='shape':y=torch.ones(2,4)
    if kind=='nan':p[0,0]=float('nan')
    if kind=='keys':keys=['a']
    if kind=='dtype':y=y.long()
    with pytest.raises(ValueError):response_quality(p,y,keys)


def test_constant_prediction_fails_strict_gate_and_cuda_agrees():
    y=torch.tensor([[0.,2.],[2.,0.]],dtype=torch.float64);p=torch.ones_like(y)
    a=response_quality(p,y,['a','a'])
    assert not a['training_gate_passed']
    assert a['excess_over_context_mean_mse']==0
    if torch.cuda.is_available():assert response_quality(p.cuda(),y.cuda(),['a','a'])==a
