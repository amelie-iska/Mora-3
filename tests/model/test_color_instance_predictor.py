import inspect
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from mora3.model import color_semantic

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_native_predictor_explicit_context_affine_and_all_gradients(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(22201)
    m=color_semantic.ColorInstancePredictor(4,slots=3,max_output_bytes=4096,alpha_policy='require-opaque').to(device=device,dtype=torch.float64)
    pixels=np.arange(7*9*3,dtype=np.uint8).reshape(7,9,3)
    actual=m(pixels)
    from mora3.model.color_input import color_tensor
    x=color_tensor(pixels,alpha_policy=m.alpha_policy,device=device,dtype=torch.float64,input_policy=m.input_policy)
    local=m.pixels(x)
    t=dict(features=local.mean((2,3)),endpoints=torch.zeros((1,2),dtype=torch.long,device=device),roles=torch.zeros(1,dtype=torch.long,device=device),identifiers=F.pad(torch.ones((1,1),device=device,dtype=torch.float64),(0,m.backbone.incidence.in_features//2-1)))
    latent,_=m.backbone([t])
    features=torch.cat((local,latent[:,0,:,None,None].expand(-1,-1,7,9)),1)
    ref=torch.einsum('qc,chw->qhw',m.head.projection.weight[:,:,0,0],features[0])+m.head.projection.bias[:,None,None]
    torch.testing.assert_close(actual,ref,rtol=1e-12,atol=1e-12)
    params=tuple(m.parameters())
    for a,b in zip(torch.autograd.grad(actual.square().mean(),params),torch.autograd.grad(ref.square().mean(),params)):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a,b,rtol=1e-11,atol=1e-11)
    assert list(inspect.signature(m.forward).parameters)==['pixels']

@pytest.mark.parametrize('case',['width','policy','alpha','budget','extent','state'])
def test_native_predictor_strict_boundaries(case):
    kw=dict(width=4,slots=3,max_output_bytes=4096,alpha_policy='require-opaque')
    if case=='width':kw['width']=3
    if case=='policy':kw['input_policy']='resize'
    if case=='alpha':kw['alpha_policy']='discard'
    if case in ('width','policy','alpha'):
        with pytest.raises(ValueError):color_semantic.ColorInstancePredictor(**kw)
        return
    m=color_semantic.ColorInstancePredictor(**kw)
    if case=='state':
        n=color_semantic.ColorInstancePredictor(**{**kw,'max_output_bytes':8192})
        with pytest.raises(ValueError):n.load_state_dict(m.state_dict())
    else:
        pixels=np.zeros((20,20,3) if case=='budget' else (513,1,3),dtype=np.uint8)
        with pytest.raises(ValueError):m(pixels)
