"""Explicit input extent; never converts ordered instance targets to semantic masks."""
import copy
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from mora3.model.color_input import color_tensor
from mora3.model.color_semantic import ColorNuclearSemantic

POLICY='native-max1000-v1'

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_complete_native_pixels_and_gradient(device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    a=np.random.default_rng(21701).integers(0,256,(1000,1000,3),dtype=np.uint8)
    x=color_tensor(a,alpha_policy='require-opaque',input_policy=POLICY,device=device,dtype=torch.float64)
    expected=torch.tensor(a.copy(),device=device,dtype=torch.float64).permute(2,0,1)[None]/255
    torch.testing.assert_close(x,expected,rtol=0,atol=0)
    x.requires_grad_();x.square().mean().backward()
    torch.testing.assert_close(x.grad,2*expected/expected.numel(),rtol=1e-15,atol=1e-22)
    with pytest.raises(ValueError):color_tensor(a,alpha_policy='require-opaque')

@pytest.mark.parametrize('policy',[None,True,1000,'unbounded','native-max1001-v1'])
def test_unknown_input_policy_refuses(policy):
    with pytest.raises(ValueError,match='policy'):
        color_tensor(np.zeros((1,1,3),dtype=np.uint8),alpha_policy='require-opaque',input_policy=policy)

@pytest.mark.parametrize('shape',[(1001,2,3),(2,1001,3),(0,2,3)])
def test_explicit_policy_remains_bounded(shape):
    with pytest.raises(ValueError):
        color_tensor(np.zeros(shape,dtype=np.uint8),alpha_policy='require-opaque',input_policy=POLICY)

def test_policy_preserves_alpha_refusal():
    with pytest.raises(ValueError,match='opaque'):
        color_tensor(np.zeros((1000,2,4),dtype=np.uint8),alpha_policy='require-opaque',input_policy=POLICY)

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_opt_in_model_equations_gradients_and_strict_state(device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.manual_seed(21701)
    model=ColorNuclearSemantic(4,alpha_policy='require-opaque',input_policy=POLICY).to(device=device,dtype=torch.float64)
    ref=copy.deepcopy(model)
    a=np.random.default_rng(21701).integers(0,256,(17,19,3),dtype=np.uint8)
    logits=model(a)
    x=torch.tensor(a.copy(),device=device,dtype=torch.float64).permute(2,0,1)[None]/255
    local=ref.pixels(x)
    t=dict(features=local.mean((2,3)),endpoints=torch.zeros((1,2),dtype=torch.long,device=device),roles=torch.zeros(1,dtype=torch.long,device=device),identifiers=F.pad(torch.ones((1,1),dtype=torch.float64,device=device),(0,ref.backbone.incidence.in_features//2-1)))
    latent,_=ref.backbone([t])
    expected=F.conv2d(torch.cat((local,latent[:,0,:,None,None].expand(-1,-1,17,19)),1),ref.head.weight,ref.head.bias)[0,0]
    torch.testing.assert_close(logits,expected,rtol=0,atol=0)
    logits.square().mean().backward();expected.square().mean().backward()
    for p,q in zip(model.parameters(),ref.parameters()):
        assert p.grad is not None
        torch.testing.assert_close(p.grad,q.grad,rtol=1e-11,atol=1e-13)
    old=ColorNuclearSemantic(4,alpha_policy='require-opaque')
    assert old.get_extra_state()['version']==1
    assert 'input_policy' not in old.get_extra_state()
    assert model.get_extra_state()['version']==2
    assert model.get_extra_state()['input_policy']==POLICY
    with pytest.raises(ValueError,match='configuration'):model.set_extra_state(old.get_extra_state())
    with pytest.raises(ValueError,match='configuration'):old.set_extra_state(model.get_extra_state())
    assert sum(p.numel() for p in old.parameters())==sum(p.numel() for p in model.parameters())
