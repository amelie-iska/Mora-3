import pytest
import torch
from mora3.model.lincs_objective import native_profile_loss

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_masked_reference_and_gradients(device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    p=torch.arange(978,device=device,dtype=torch.float64).requires_grad_()
    y=torch.zeros_like(p); y[5]=float('nan')
    mask=torch.ones(978,device=device,dtype=torch.bool); mask[5]=False
    loss=native_profile_loss(p,y,mask,modality='L1000')
    ref=p[mask].square().mean()
    assert torch.equal(loss,ref)
    assert torch.equal(torch.autograd.grad(loss,p)[0],torch.autograd.grad(ref,p)[0])

@pytest.mark.parametrize('n,modality',[(977,'L1000'),(1782,'CellPainting'),(978,'unknown')])
def test_reject_wrong_axis(n,modality):
    with pytest.raises(ValueError): native_profile_loss(torch.zeros(n),torch.zeros(n),torch.ones(n,dtype=torch.bool),modality=modality)

def test_reject_empty_and_observed_nonfinite():
    p=torch.zeros(1783); y=p.clone(); m=torch.zeros(1783,dtype=torch.bool)
    with pytest.raises(ValueError): native_profile_loss(p,y,m,modality='CellPainting')
    m[0]=True; y[0]=float('nan')
    with pytest.raises(ValueError): native_profile_loss(p,y,m,modality='CellPainting')
