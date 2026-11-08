import pytest
import torch
from mora3.model import flow_quality

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_energy_score_reference_and_explicit_negative_policy(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('no CUDA')
    samples=torch.tensor([[[-1.,3.]],[[3.,3.]]],device=device,dtype=torch.float64)
    target=torch.tensor([[1.,3.]],device=device,dtype=torch.float64)
    r=flow_quality.response_quality(samples,target)
    assert r['sample_mse']==pytest.approx(2)
    assert r['ensemble_mean_mse']==pytest.approx(0)
    assert r['energy_score']==pytest.approx(1)
    assert r['negative_fraction']==pytest.approx(.25)
    assert r['clamp_zero_sample_mse']==pytest.approx(1.25)
    assert r['coordinates']==2 and r['draws']==2

@pytest.mark.parametrize('fault',['nan','shape','negative_target','empty'])
def test_rejects_invalid_or_missing_coordinates(fault):
    x=torch.ones(2,3,4);y=torch.ones(3,4)
    if fault=='nan':x[0,0,0]=float('nan')
    if fault=='shape':y=y[:1]
    if fault=='negative_target':y[0,0]=-1
    if fault=='empty':x=x[:0]
    with pytest.raises(ValueError):flow_quality.response_quality(x,y)
