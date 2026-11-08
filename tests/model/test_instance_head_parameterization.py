import pytest
import torch
from mora3.model.instance_affinity import OverlappingInstanceHead
from mora3.model import parameterization as p

@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('width,slots', [(8,3),(12,7),(16,65)])
def test_instance_head_independent_count_axes_and_state(device,width,slots):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.manual_seed(28001)
    head=OverlappingInstanceHead(width,slots=slots,max_output_bytes=1000000,
        slot_policy='extended-max512-v2' if slots>64 else 'bounded-max64-v1').to(device)
    before={n:t.detach().clone() for n,t in head.named_parameters()}
    rng=torch.random.get_rng_state().clone()
    roles=p.instance_head_parameter_roles(head)
    ledger=p.parameter_ledger(head,roles)
    assert sum(e.numel for e in ledger.parameters)==slots*(width+1)
    assert roles['projection.weight'].role=='custom'
    assert roles['projection.weight'].axes[0].kind=='finite'
    assert roles['projection.weight'].axes[1]==p.Axis('width','hidden')
    assert roles['projection.bias'].axes==(roles['projection.weight'].axes[0],)
    assert torch.equal(rng,torch.random.get_rng_state())
    assert all(torch.equal(t,before[n]) and t.grad is None for n,t in head.named_parameters())

@pytest.mark.parametrize('change', ['slots','channels','policy','budget','padding','stride','extra','shape'])
def test_instance_head_refuses_configuration_drift(change):
    head=OverlappingInstanceHead(8,slots=3,max_output_bytes=1000000)
    if change=='slots': head.slots=4
    elif change=='channels': head.channels=9
    elif change=='policy': head.slot_policy='unknown'
    elif change=='budget': head.max_output_bytes=0
    elif change=='padding': head.projection.padding=(1,1)
    elif change=='stride': head.projection.stride=(2,2)
    elif change=='extra': head.extra=torch.nn.Linear(8,8)
    else: head.projection.weight=torch.nn.Parameter(torch.zeros(3,8,2,1))
    with pytest.raises(ValueError): p.instance_head_parameter_roles(head)

def test_slot_order_identity_depends_on_capacity():
    def roles(slots):
        return p.instance_head_parameter_roles(OverlappingInstanceHead(8,slots=slots,max_output_bytes=10000))
    assert roles(3)['projection.bias'].axes != roles(4)['projection.bias'].axes
