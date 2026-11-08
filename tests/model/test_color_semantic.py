import copy
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from mora3.model.color_semantic import ColorNuclearSemantic, nuclear_semantic_objective
from mora3.model.color_input import color_tensor

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_equations_gradients_and_target_exclusion(device):
    torch.manual_seed(16601)
    model=ColorNuclearSemantic(4,alpha_policy='require-opaque').to(device=device,dtype=torch.float64)
    ref=copy.deepcopy(model)
    pixels=np.random.default_rng(16601).integers(0,256,(7,9,3),dtype=np.uint8)
    logits=model(pixels)
    x=color_tensor(pixels,alpha_policy='require-opaque',device=device,dtype=torch.float64)
    local=ref.pixels(x)
    tensor=dict(features=local.mean((2,3)),endpoints=torch.zeros((1,2),dtype=torch.long,device=device),
        roles=torch.zeros(1,dtype=torch.long,device=device),identifiers=F.pad(torch.ones((1,1),device=device,dtype=torch.float64),(0,ref.backbone.incidence.in_features//2-1)))
    latent,_=ref.backbone([tensor])
    expected=F.conv2d(torch.cat((local,latent[:,0,:,None,None].expand(-1,-1,7,9)),1),ref.head.weight,ref.head.bias)[0,0]
    torch.testing.assert_close(logits,expected,rtol=0,atol=0)
    target=torch.arange(63,device=device).reshape(7,9)%3==0
    loss=nuclear_semantic_objective(logits,target)
    oracle=(F.softplus(expected)-target.to(expected.dtype)*expected).mean()
    torch.testing.assert_close(loss,oracle,rtol=1e-14,atol=1e-14)
    loss.backward();oracle.backward()
    for (name,p),(other,q) in zip(model.named_parameters(),ref.named_parameters()):
        assert name==other and p.grad is not None,name
        torch.testing.assert_close(p.grad,q.grad,rtol=1e-11,atol=1e-13)
    assert torch.equal(model(pixels),logits)
    assert not torch.equal(nuclear_semantic_objective(logits,~target),loss)
    assert logits.shape==(7,9)

@pytest.mark.parametrize('bad',[torch.zeros(3,4),torch.zeros(1,3,4,dtype=torch.bool),torch.ones(3,4,dtype=torch.int64)])
def test_reject_non_native_binary_target(bad):
    with pytest.raises(ValueError):nuclear_semantic_objective(torch.zeros(3,4),bad)

def test_nonfinite_and_configuration_reject():
    with pytest.raises(ValueError):nuclear_semantic_objective(torch.full((3,4),float('nan')),torch.zeros(3,4,dtype=torch.bool))
    with pytest.raises(ValueError):ColorNuclearSemantic(3,alpha_policy='require-opaque')
    with pytest.raises(ValueError):ColorNuclearSemantic(4,alpha_policy='composite')
    m=ColorNuclearSemantic(4,alpha_policy='require-opaque')
    state=m.get_extra_state();state['outputs']=['whole-cell']
    with pytest.raises(ValueError):m.set_extra_state(state)
    with pytest.raises(ValueError):m(np.zeros((4,4,4),dtype=np.uint8))
