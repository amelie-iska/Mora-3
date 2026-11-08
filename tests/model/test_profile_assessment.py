import copy
from dataclasses import replace
import pytest
import torch
from test_molecule_profile_model import model, view
import importlib

@pytest.fixture(autouse=True)
def assessment_module():
    global module
    assert importlib.util.find_spec("mora3.model.profile_assessment") is not None, "missing native profile assessment"
    module=importlib.import_module("mora3.model.profile_assessment")


def test_metrics_match_scalar_reference_and_constant_correlation():
    z=torch.tensor([[1.,2.,3.],[3.,3.,3.]],dtype=torch.float64)
    y=torch.tensor([[2.,4.,6.],[1.,2.,3.]],dtype=torch.float64)
    r=module.profile_metrics(z,y)
    assert r['records']==2 and r['coordinates_per_record']==3
    assert r['mse']==pytest.approx(19/6)
    assert r['mae']==pytest.approx(1.5)
    assert r['mean_record_pearson']==pytest.approx(1.)
    assert r['pearson_defined_records']==1
    assert r['zero_baseline_mse']==pytest.approx(70/6)


@pytest.mark.parametrize('bad',[torch.tensor([[float('nan'),0.]]),torch.ones(2),torch.ones(0,2)])
def test_rejects_incomplete_or_nonfinite_metrics(bad):
    with pytest.raises(ValueError):module.profile_metrics(bad,torch.ones(1,2))


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_evaluator_target_exclusion_modes_rng_gradients_and_weights(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    m=model(device);m.train();m.output['L1000'].eval()
    vs=(view(),view(modality='CellPainting'))
    m(vs)[0].sum().backward()
    states=copy.deepcopy(m.state_dict());grads=[None if p.grad is None else p.grad.clone() for p in m.parameters()]
    modes=[x.training for x in m.modules()];rng=torch.get_rng_state().clone()
    cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
    result=module.assess_profiles(m,vs)
    changed=tuple(replace(v,target_bytes=b'not readable targets') for v in vs)
    # Prediction boundary must not access targets; assessment reads them afterward.
    z=module.predict_profiles(m,changed)
    assert all(torch.equal(a,b) for a,b in zip(result['predictions'],z))
    assert set(result['modalities'])=={'CellPainting','L1000'}
    assert result['modalities']['L1000']['coordinates_per_record']==977
    assert result['modalities']['CellPainting']['coordinates_per_record']==1783
    assert [x.training for x in m.modules()]==modes
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all()))
    for k,v in states.items():
        if isinstance(v,torch.Tensor):assert torch.equal(v,m.state_dict()[k])
        else:assert v==m.state_dict()[k]
    for p,g in zip(m.parameters(),grads):
        assert (p.grad is None)==(g is None)
        if g is not None:assert torch.equal(p.grad,g)
    assert all(not p.requires_grad for p in z)
    with pytest.raises(ValueError):module.assess_profiles(m,(replace(vs[0],axis=vs[0].axis[::-1]),))
    assert [x.training for x in m.modules()]==modes
