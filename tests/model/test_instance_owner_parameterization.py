import pytest
import torch
from torch import nn
from mora3.model.color_semantic import ColorInstancePredictor
from mora3.model import parameterization as p


def owner(w=8, slots=3):
    return ColorInstancePredictor(w, slots=slots, max_output_bytes=24000000,
                                  alpha_policy='require-opaque')


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('w', [8,12,16])
def test_complete_instance_owner_counts_aliases_and_state(device,w):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.manual_seed(28101)
    a=owner(w).to(device); b=owner(w,7).to(device)
    b.backbone=a.backbone; b.pixels=a.pixels
    model=nn.ModuleDict({'a':a,'b':b})
    before={n:t.clone() for n,t in model.named_parameters()}
    rng=torch.random.get_rng_state().clone()
    ledger=p.instance_parameter_ledger(model,channel_axis=('R','G','B'),identifier_axis=('image',))
    encoder=9*(3*w+51*w*w)+20*w
    backbone=w*(w+7)+(6*w*w+10*w)
    assert ledger.total_numel==encoder+backbone+10*(2*w+1)
    assert any('a.pixels.enc0.0.weight' in e.aliases and 'b.pixels.enc0.0.weight' in e.aliases for e in ledger.parameters)
    assert torch.equal(rng,torch.random.get_rng_state())
    assert all(torch.equal(t,before[n]) and t.grad is None for n,t in model.named_parameters())


@pytest.mark.parametrize('change',['width','channels','backbone','extra','head','alpha','policy','empty','type','tie'])
def test_instance_owner_refuses_drift(change):
    a=owner(); b=owner(); b.backbone=a.backbone
    model=nn.ModuleDict({'a':a,'b':b})
    channels=('R','G','B')
    if change=='width': a.width=10
    elif change=='channels': channels=('R','B')
    elif change=='backbone': b.backbone=owner().backbone
    elif change=='extra': a.extra=nn.Linear(8,8)
    elif change=='head': a.head=nn.Linear(16,3)
    elif change=='alpha': a.alpha_policy='unknown'
    elif change=='policy': a.input_policy='unknown'
    elif change=='empty': model=nn.ModuleDict()
    elif change=='type': model=nn.Sequential(a)
    else: a.pixels.enc0[2].weight=a.pixels.enc0[0].weight
    with pytest.raises(ValueError):
        p.instance_parameter_ledger(model,channel_axis=channels,identifier_axis=('image',))


def test_single_owner_complete_and_custom_scaling_refuses():
    a=owner()
    ledger=p.instance_parameter_ledger(a,channel_axis=('R','G','B'),identifier_axis=('image',))
    assert ledger.total_numel==sum(t.numel() for t in a.parameters())
    assert set(alias for e in ledger.parameters for alias in e.aliases)==set(dict(a.named_parameters()))
    with pytest.raises(ValueError): p.width_family(ledger,ledger,ledger)
