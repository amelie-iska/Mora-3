"""Native image convolution metadata; never a scaling qualification."""
import pytest
import torch
from torch import nn
from mora3.model.morphology import MultiscalePixels
from mora3.model.parameterization import parameter_ledger, width_family


def roles(m, channels=('intensity',)):
    from mora3.model.parameterization import image_encoder_parameter_roles
    return image_encoder_parameter_roles(m, channel_axis=channels)


@pytest.mark.parametrize('device', ['cpu','cuda'])
@pytest.mark.parametrize('channels', [('intensity',),('red','green','blue')])
def test_complete_native_count_axes_and_readonly(device, channels):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.manual_seed(27901)
    for w in (8,12,16):
        m=MultiscalePixels(w,input_channels=len(channels)).to(device)
        before={n:p.clone() for n,p in m.named_parameters()}
        rng=torch.random.get_rng_state().clone()
        gpu=torch.cuda.get_rng_state().clone() if device=='cuda' else None
        r=roles(m,channels); ledger=parameter_ledger(m,r)
        # Sum all ten 3x3 convolution weights and their output-channel biases.
        assert ledger.total_numel==9*(len(channels)*w+51*w*w)+20*w
        assert len(r)==20
        assert r['enc0.0.weight'].role=='custom'
        assert [a.kind for a in r['enc0.0.weight'].axes]==['width','finite','finite','finite']
        assert r['dec1.0.weight'].axes[1].identity=='image.concat-skip2-bottom4'
        assert r['dec0.0.weight'].axes[1].identity=='image.concat-skip1-decoder2'
        assert all(r[n].role=='vector' for n in r if n.endswith('bias'))
        assert torch.equal(rng,torch.random.get_rng_state())
        if gpu is not None: assert torch.equal(gpu,torch.cuda.get_rng_state())
        for n,p in m.named_parameters(): assert torch.equal(p,before[n]) and p.grad is None


def test_channel_identity_order_and_scaling_refusal():
    m=MultiscalePixels(8,input_channels=3)
    a=roles(m,('red','green','blue')); b=roles(m,('blue','green','red'))
    assert a['enc0.0.weight']!=b['enc0.0.weight']
    assert a['enc1.0.weight']==b['enc1.0.weight']
    with pytest.raises(ValueError,match='custom'):
        width_family(*(parameter_ledger(m,roles(m)) for m in
                       [MultiscalePixels(w) for w in (8,12,16)]))


@pytest.mark.parametrize('drift',['extra','module','stride','padding','dilation','groups','mode',
    'transposed','output_padding','bias','shape','activation','subclass','channels','duplicate','empty','list'])
def test_drift_refuses(drift):
    m=MultiscalePixels(8); channels=('intensity',)
    if drift=='extra': m.register_parameter('extra',nn.Parameter(torch.ones(1)))
    if drift=='module': m.extra=nn.Identity()
    for attr,val in [('stride',(2,2)),('padding',(0,0)),('dilation',(2,2)),('groups',2),
                     ('mode','reflect'),('transposed',True),('output_padding',(1,1))]:
        if drift==attr: setattr(m.enc0[0], 'padding_mode' if attr=='mode' else attr,val)
    if drift=='bias': m.enc0[0].bias=None
    if drift=='shape': m.dec0[0]=nn.Conv2d(24,8,1)
    if drift=='activation': m.enc0[1]=nn.ReLU()
    if drift=='subclass':
        class Other(MultiscalePixels): pass
        m=Other(8)
    if drift=='channels': channels=('red','green')
    if drift=='duplicate': channels=('x','x')
    if drift=='empty': channels=(' ',)
    if drift=='list': channels=['intensity']
    with pytest.raises(ValueError): roles(m,channels)
