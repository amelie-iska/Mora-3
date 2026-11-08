"""Explicit head-only capacity policy; loss/task admission remains separate."""
import pytest
import torch
from mora3.model.instance_affinity import OverlappingInstanceHead, overlapping_set_objective

POLICY = 'extended-max512-v2'

@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('slots', [65, 512])
def test_extended_affine_all_gradients(device, slots):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.manual_seed(23001)
    head = OverlappingInstanceHead(4, slots=slots, max_output_bytes=100000,
                                   slot_policy=POLICY).to(device=device, dtype=torch.float64)
    x = torch.randn(1, 4, 2, 3, device=device, dtype=torch.float64, requires_grad=True)
    actual = head(x)
    expected = torch.einsum('qc,chw->qhw', head.projection.weight[:, :, 0, 0], x[0]) + head.projection.bias[:, None, None]
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
    parameters = (x, *head.parameters())
    a = torch.autograd.grad(actual.square().sum(), parameters, retain_graph=True)
    b = torch.autograd.grad(expected.square().sum(), parameters)
    for left, right in zip(a, b):
        assert torch.isfinite(left).all() and left.abs().sum() > 0
        torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-12)
    assert sum(p.numel() for p in head.parameters()) == slots * 5
    # Explicit head policy cannot silently widen the existing assignment loss.
    with pytest.raises(ValueError, match='1..64'):
        overlapping_set_objective(actual, torch.ones(1, 2, 3, device=device, dtype=torch.bool))

@pytest.mark.parametrize('slots,policy', [(513,POLICY),(0,POLICY),(True,POLICY),(65,'bounded-max64-v1'),(3,'unknown'),(3,None)])
def test_policy_refuses_before_parameter_allocation(monkeypatch, slots, policy):
    def forbidden(*args, **kwargs):
        raise AssertionError('projection allocated before policy validation')
    monkeypatch.setattr(torch.nn, 'Conv2d', forbidden)
    with pytest.raises(ValueError):
        OverlappingInstanceHead(4, slots=slots, max_output_bytes=1000, slot_policy=policy)

@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_exact_output_budget_before_projection(device):
    if device == 'cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    head = OverlappingInstanceHead(4, slots=512, max_output_bytes=512*6*8,
                                  slot_policy=POLICY).to(device=device, dtype=torch.float64)
    calls=[]
    hook=head.projection.register_forward_pre_hook(lambda *args:calls.append(True))
    assert head(torch.zeros(1,4,2,3,device=device,dtype=torch.float64)).shape == (512,2,3)
    calls.clear()
    with pytest.raises(ValueError,match='output-byte'):
        head(torch.zeros(1,4,2,4,device=device,dtype=torch.float64))
    assert not calls
    hook.remove()

def test_versioned_state_and_legacy_identity():
    legacy=OverlappingInstanceHead(4,slots=3,max_output_bytes=4096)
    assert legacy.get_extra_state() == dict(version=1,architecture='independent-affine-mask-slots-v1',channels=4,slots=3,max_output_bytes=4096,max_extent=1000,training_admission=False)
    extended=OverlappingInstanceHead(4,slots=3,max_output_bytes=4096,slot_policy=POLICY)
    assert extended.get_extra_state() == dict(version=2,architecture='independent-affine-mask-slots-v2',channels=4,slots=3,max_output_bytes=4096,max_extent=1000,training_admission=False,slot_policy=POLICY,max_slots=512)
    for target,source in [(legacy,extended),(extended,legacy)]:
        with pytest.raises(ValueError,match='configuration mismatch'):target.load_state_dict(source.state_dict())
    same=OverlappingInstanceHead(4,slots=3,max_output_bytes=4096,slot_policy=POLICY)
    same.load_state_dict(extended.state_dict())
    for a,b in zip(same.parameters(),extended.parameters()):assert torch.equal(a,b)
