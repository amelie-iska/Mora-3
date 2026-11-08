import numpy as np
import pytest
import torch
from mora3.model.color_input import color_tensor

@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('channels',[3,4])
def test_exact_channel_order_and_alpha_policy(device,channels):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    a=np.arange(2*3*channels,dtype=np.uint8).reshape(2,3,channels)
    if channels==4: a[:,:,3]=255
    x=color_tensor(a,alpha_policy='require-opaque',device=device,dtype=torch.float64)
    expected=torch.tensor(a[:,:,:3].copy(),device=device,dtype=torch.float64).permute(2,0,1)[None]/255
    torch.testing.assert_close(x,expected,rtol=0,atol=0)
    assert x.shape==(1,3,2,3)
    a[:]=0
    assert x.sum()>0

@pytest.mark.parametrize('bad',[np.zeros((2,3),dtype=np.uint8),np.zeros((2,3,3),dtype=np.float32),np.zeros((2,3,2),dtype=np.uint8),np.zeros((0,3,3),dtype=np.uint8)])
def test_reject_unsupported_pixels(bad):
    with pytest.raises(ValueError): color_tensor(bad,alpha_policy='require-opaque')

def test_alpha_is_not_silently_dropped_or_composited():
    a=np.zeros((2,3,4),dtype=np.uint8);a[:,:,3]=254
    with pytest.raises(ValueError,match='opaque'): color_tensor(a,alpha_policy='require-opaque')
    with pytest.raises(ValueError,match='policy'): color_tensor(a,alpha_policy='guess')
    a[:,:,3]=255
    with pytest.raises(ValueError): color_tensor(a,alpha_policy='require-opaque',dtype=torch.int64)
