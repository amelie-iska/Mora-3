import math
import pytest
import torch
from mora3.model.tox21_assessment import masked_call_metrics, fit_call_prior


def test_ties_auc_unknown_and_reference_equations():
    z=torch.tensor([[0.,float('nan')],[0.,2.],[2.,-2.]],dtype=torch.float64)
    y=torch.tensor([[0.,0.],[1.,1.],[1.,0.]],dtype=torch.float64)
    m=torch.tensor([[True,False],[True,True],[True,True]])
    r=masked_call_metrics(z,y,m)
    assert r['axes'][0]['roc_auc']==.75
    assert r['axes'][1]['roc_auc']==1.
    expected=sum(math.log1p(math.exp(float(z[i,j])))-float(z[i,j]*y[i,j]) for i,j in [(0,0),(1,0),(2,0),(1,1),(2,1)])/5
    assert r['micro']['bce']==pytest.approx(expected)
    assert r['micro']['observed']==5
    assert r['axes'][0]['accuracy']==pytest.approx(2/3)


def test_missing_class_and_train_only_prior():
    y=torch.tensor([[0.,0.],[0.,0.]])
    m=torch.tensor([[True,False],[True,False]])
    prior=fit_call_prior(y,m)
    assert prior==pytest.approx([math.log(.5/2.5),0.])
    r=masked_call_metrics(torch.zeros_like(y),y,m)
    assert r['axes'][0]['roc_auc'] is None
    assert r['axes'][0]['roc_auc_undefined']=='requires_both_classes'
    assert r['axes'][1]['bce'] is None
    assert r['macro_roc_auc'] is None


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_device_reference_and_observed_nonfinite_rejected(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip()
    z=torch.tensor([[-2.],[2.]],device=device)
    y=torch.tensor([[0.],[1.]],device=device);m=torch.ones_like(y,dtype=torch.bool)
    r=masked_call_metrics(z,y,m)
    assert r['micro']['brier']==pytest.approx((1/(1+math.exp(2)))**2)
    assert r['macro_roc_auc']==1.
    z[0,0]=float('nan')
    with pytest.raises(ValueError):masked_call_metrics(z,y,m)


def test_invalid_labels_and_shapes_rejected():
    with pytest.raises(ValueError):fit_call_prior(torch.tensor([[2.]]),torch.tensor([[True]]))
    with pytest.raises(ValueError):masked_call_metrics(torch.zeros(2,1),torch.zeros(1,1),torch.ones(1,1,dtype=torch.bool))
