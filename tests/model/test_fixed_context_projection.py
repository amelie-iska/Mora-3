"""Finite latent/time columns plus scalable context: independent tiny equations."""
import copy
import math
import pytest
import torch
from torch.nn import functional as F
from mora3.model.fixed_context_projection import (FixedContextProjection,
    fixed_context_parameter_roles,fixed_context_width_family)


def make(width=8):
    return FixedContextProjection(width,('state0','state1','layer/horizon'),tuple('fixed-hidden:'+str(i) for i in range(4)),
        source_namespace='synthetic-GFN-fixed-context',base_context_width=8,context_mult=1.5)


def test_native_matrix_draws_split_without_shared_storage_or_extra_rng():
    torch.manual_seed(2026110676);native=torch.nn.Linear(19,4);rng=torch.get_rng_state().clone()
    torch.manual_seed(2026110676);p=make(16)
    scale=math.sqrt(19/11)
    assert torch.equal(rng,torch.get_rng_state())
    assert torch.equal(p.fixed_weight,native.weight[:,:3]*scale)
    assert torch.equal(p.context_weight,native.weight[:,3:]*scale)
    assert torch.equal(p.bias,native.bias*scale)
    assert len({q.untyped_storage().data_ptr() for q in p.parameters()})==3


def test_actual_fixed_prefix_context_suffix_equation_and_all_gradients():
    p=make(16).double();x=torch.linspace(-1,1,38,dtype=torch.float64).reshape(2,19).requires_grad_()
    expected=F.linear(x[:,:3],p.fixed_weight,p.bias)+F.linear(x[:,3:]*(1.5/(16/8)),p.context_weight)
    actual=p(x)
    assert torch.equal(actual,expected)
    params=(x,*p.parameters())
    for a,b in zip(torch.autograd.grad(actual.sum(),params,retain_graph=True),torch.autograd.grad(expected.sum(),params)):
        assert torch.equal(a,b)


def test_finite_output_axes_are_preserved_across_main_width_family():
    ms=[make(w) for w in (8,12,16)]
    family=fixed_context_width_family(*ms)
    assert family.total_numel==tuple(sum(p.numel() for p in m.parameters()) for m in ms)
    r=fixed_context_parameter_roles(ms[-1])
    assert tuple(a.kind for a in r['fixed_weight'].axes)==('finite','finite')
    assert tuple(a.kind for a in r['context_weight'].axes)==('finite','width')
    assert r['context_weight'].role=='output'
    state=copy.deepcopy(ms[-1].state_dict());state['_extra_state']['fixed_axis']=('wrong',)
    state['context_weight'].fill_(99)
    before={n:p.clone() for n,p in ms[-1].named_parameters()}
    with pytest.raises(ValueError,match='configuration|policy'):ms[-1].load_state_dict(state)
    for n,p in ms[-1].named_parameters():assert torch.equal(p,before[n])
