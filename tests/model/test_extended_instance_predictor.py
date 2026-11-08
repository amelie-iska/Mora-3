"""Extended native predictor: exact affine gradients and strict policy isolation."""
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from mora3.model.color_semantic import ColorInstancePredictor
from mora3.model.color_input import color_tensor

@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('slots',[65,512])
def test_extended_predictor_reference(device,slots):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(23201)
    m=ColorInstancePredictor(4,slots=slots,max_output_bytes=100000,alpha_policy='require-opaque',slot_policy='extended-max512-v2').to(device=device,dtype=torch.float64)
    pixels=np.arange(3*4*3,dtype=np.uint8).reshape(3,4,3)
    actual=m(pixels)
    x=color_tensor(pixels,alpha_policy=m.alpha_policy,device=device,dtype=torch.float64,input_policy=m.input_policy)
    local=m.pixels(x)
    t=dict(features=local.mean((2,3)),endpoints=torch.zeros((1,2),dtype=torch.long,device=device),roles=torch.zeros(1,dtype=torch.long,device=device),identifiers=F.pad(torch.ones((1,1),device=device,dtype=torch.float64),(0,m.backbone.incidence.in_features//2-1)))
    latent,_=m.backbone([t])
    features=torch.cat((local,latent[:,0,:,None,None].expand(-1,-1,3,4)),1)
    expected=torch.einsum('qc,chw->qhw',m.head.projection.weight[:,:,0,0],features[0])+m.head.projection.bias[:,None,None]
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    params=tuple(m.parameters())
    for a,b in zip(torch.autograd.grad(actual.square().mean(),params),torch.autograd.grad(expected.square().mean(),params)):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a,b,rtol=1e-11,atol=1e-11)
    calls=[]
    hook=m.pixels.register_forward_pre_hook(lambda *a:calls.append(True))
    with pytest.raises(ValueError,match='output-byte'):m(np.zeros((20,20,3),dtype=np.uint8))
    assert not calls
    hook.remove()

def test_extended_predictor_state_separation():
    kw=dict(width=4,slots=3,max_output_bytes=4096,alpha_policy='require-opaque')
    old=ColorInstancePredictor(**kw)
    new=ColorInstancePredictor(**kw,slot_policy='extended-max512-v2')
    assert old.get_extra_state()['version']==1
    assert new.get_extra_state()['version']==2
    assert new.get_extra_state()['slot_policy']=='extended-max512-v2'
    for target,source in [(old,new),(new,old)]:
        with pytest.raises(ValueError,match='configuration mismatch'):target.load_state_dict(source.state_dict())
    same=ColorInstancePredictor(**kw,slot_policy='extended-max512-v2')
    same.load_state_dict(new.state_dict())
    for a,b in zip(same.parameters(),new.parameters()):assert torch.equal(a,b)
