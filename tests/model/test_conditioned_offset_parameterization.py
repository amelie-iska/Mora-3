import pytest
import torch
from torch import nn
from mora3.model import parameterization as p
from mora3.model.color_conditioning import ConditionedColorInstance
from mora3.model.color_semantic import ColorInstancePredictor


def owner(w=8,s=3):
    return ConditionedColorInstance(ColorInstancePredictor(w,slots=s,max_output_bytes=24000000,alpha_policy='require-opaque'),id_width=1)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('w,s',[(8,3),(12,7),(16,5)])
def test_local_offset_equation_and_state(device,w,s):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.manual_seed(28201)
    m=owner(w,s).to(device)
    before={n:t.clone() for n,t in m.named_parameters()}
    rng=torch.random.get_rng_state().clone()
    roles=p.conditioned_instance_offset_parameter_roles(m)
    ledger=p.parameter_ledger(m.offset,roles)
    assert ledger.total_numel==s*(w+1)
    assert roles['weight'].role=='output'
    assert roles['weight'].axes[0]==p.instance_head_parameter_roles(m.color.head)['projection.weight'].axes[0]
    assert roles['weight'].axes[1]==p.Axis('width','hidden')
    assert roles['bias'].axes==(roles['weight'].axes[0],)
    assert torch.equal(rng,torch.random.get_rng_state())
    assert all(torch.equal(t,before[n]) and t.grad is None for n,t in m.named_parameters())


@pytest.mark.parametrize('change',['width','slots','offset','bias','extra','config','type'])
def test_local_offset_refuses_drift(change):
    m=owner()
    if change=='width': m.color.width=12
    elif change=='slots': m.color.head.slots=5
    elif change=='offset': m.offset=nn.Linear(7,3)
    elif change=='bias': m.offset.register_parameter('bias',None)
    elif change=='extra': m.offset.extra=nn.Linear(8,3)
    elif change=='config': m.config['slots']=5
    else: m=nn.Sequential(m)
    with pytest.raises(ValueError): p.conditioned_instance_offset_parameter_roles(m)
