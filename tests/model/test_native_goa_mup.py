"""Native reader-shaped GOA fixtures; source pool completeness is not inferred."""
import copy
import math
import pytest
import torch
from torch.nn import functional as F
from mora3.model.uniprot_go import NativeGOAHead
from mora3.model.goa_mup import goa_mup_parameter_roles,goa_mup_width_family
from test_goa_objective import view


def make(width,packet):
    return NativeGOAHead(width,(packet,),parameterization_policy='mup-goa-v1',base_width=8,output_mult=1.5)


def test_native_readout_initialization_exact_rng_and_nonunit_equation_all_gradients(tmp_path):
    packet=view(tmp_path)
    torch.manual_seed(2026110678);old=NativeGOAHead(16,(packet,));rng=torch.get_rng_state().clone()
    torch.manual_seed(2026110678);head=make(16,packet)
    assert torch.equal(rng,torch.get_rng_state())
    for name,p in old.named_parameters():assert torch.equal(dict(head.named_parameters())[name],p*math.sqrt(16/8))
    head=head.double();x=torch.linspace(-1,1,16,dtype=torch.float64)[None].requires_grad_()
    expected=F.linear(x*(1.5/(16/8)),head.output.weight,head.output.bias);actual=head(x)
    assert torch.equal(actual,expected)
    for a,b in zip(torch.autograd.grad(actual.sum(),(x,*head.parameters()),retain_graph=True),
                   torch.autograd.grad(expected.sum(),(x,*head.parameters()))):assert torch.equal(a,b)


def test_complete_source_pool_axis_family_and_strict_policy_checkpoint(tmp_path):
    packet=view(tmp_path);ms=[make(w,packet) for w in (8,12,16)]
    family=goa_mup_width_family(*ms)
    assert family.total_numel==tuple(sum(p.numel() for p in m.parameters()) for m in ms)
    assert not any(r.role=='custom' for r in goa_mup_parameter_roles(ms[-1]).values())
    state=copy.deepcopy(ms[-1].state_dict());state['_extra_state']['mup']['output_mult']=True
    state['output.weight'].fill_(99)
    before={n:p.clone() for n,p in ms[-1].named_parameters()}
    with pytest.raises(ValueError,match='configuration|policy'):ms[-1].load_state_dict(state)
    for n,p in ms[-1].named_parameters():assert torch.equal(p,before[n])


def test_implicit_policy_refuses_before_nn_initialization(tmp_path):
    packet=view(tmp_path);rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):NativeGOAHead(16,(packet,),base_width=8)
    assert torch.equal(rng,torch.get_rng_state())
