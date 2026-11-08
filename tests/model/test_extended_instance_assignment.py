"""Synthetic equation probes; no full-source execution or training coverage."""
import pytest
import torch
from torch.nn import functional as F
from mora3.model.instance_affinity import overlapping_set_objective

DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
POLICY = 'extended-max512-v2'

@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('slots', [65, 512])
@pytest.mark.parametrize('block', [None, 1, 3])
def test_extended_complete_overlapping_set_equation_and_all_logits(device, slots, block):
    # Unique optimum: every slot strongly predicts its own unique nonempty mask.
    # Sixteen pixels admit 65535 distinct nonempty masks, including overlaps.
    bits = torch.arange(16, device=device)
    y = ((torch.arange(1, slots+1, device=device)[:, None] >> bits) & 1).bool().reshape(slots,4,4)
    x = torch.where(y, 9., -9.).double().requires_grad_()
    reference = F.binary_cross_entropy_with_logits(x,y.double())
    actual = overlapping_set_objective(x,y,slot_policy=POLICY,pixel_block_size=block)
    torch.testing.assert_close(actual,reference,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(torch.autograd.grad(actual,x)[0],torch.autograd.grad(reference,x)[0],rtol=1e-12,atol=1e-12)
    with pytest.raises(ValueError):
        overlapping_set_objective(x,y,pixel_block_size=block)

@pytest.mark.parametrize('policy', [None, True, 512, 'future', 'extended-max513-v3'])
def test_unknown_policy_refused(policy):
    with pytest.raises(ValueError,match='policy'):
        overlapping_set_objective(torch.zeros(1,1,1),torch.ones(1,1,1,dtype=torch.bool),slot_policy=policy)

@pytest.mark.parametrize('slots,n', [(513,1),(65,66),(65,1)])
def test_extended_invalid_capacity_or_empty_plane_refused(slots,n):
    y=torch.ones(n,1,1,dtype=torch.bool)
    if slots==65 and n==1: y.zero_()
    with pytest.raises(ValueError):
        overlapping_set_objective(torch.zeros(slots,1,1),y,slot_policy=POLICY)

@pytest.mark.parametrize('device',DEVICES)
def test_legacy_explicit_default_is_identical(device):
    torch.manual_seed(23101)
    x=torch.randn(4,3,3,device=device,dtype=torch.float64,requires_grad=True)
    y=torch.rand(2,3,3,device=device)>.4
    a=overlapping_set_objective(x,y)
    b=overlapping_set_objective(x,y,slot_policy='bounded-max64-v1')
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    torch.testing.assert_close(torch.autograd.grad(a,x)[0],torch.autograd.grad(b,x)[0],rtol=0,atol=0)

@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('n',[0,2])
def test_extended_padding_and_permuted_overlap_reference(device,n):
    slots=65
    y=torch.tensor([[[1,1],[0,0]],[[0,1],[1,0]]],device=device,dtype=torch.bool)[:n]
    padded=torch.zeros(slots,2,2,device=device,dtype=torch.float64)
    if n: padded[7]=y[1]; padded[53]=y[0]
    x=(padded*16-8).requires_grad_()
    actual=overlapping_set_objective(x,y,slot_policy=POLICY,pixel_block_size=3)
    reference=F.binary_cross_entropy_with_logits(x,padded)
    torch.testing.assert_close(actual,reference,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(torch.autograd.grad(actual,x)[0],torch.autograd.grad(reference,x)[0],rtol=1e-12,atol=1e-12)
