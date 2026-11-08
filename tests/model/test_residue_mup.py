"""Native residue μP equations; synthetic tensors, no source training."""
import copy
import math
import pytest
import torch
from torch.nn import functional as F
from mora3.model.residue_objective import ContextualResidueHead,SharedResidueModel
from mora3.model.residue_mup import residue_mup_parameter_roles,residue_mup_width_family
from mora3.model.mu_optimizer import MuAdamW
from test_annotation_mup import make as annotation


def make(width=8):
    return SharedResidueModel(annotation(width),('site','bond'),
        parameterization_policy='mup-residue-v1')


def test_fresh_head_initialization_exact_native_draws_then_bias_and_readout_rescale():
    torch.manual_seed(2026110661);old=ContextualResidueHead(12,('site','bond'))
    rng=torch.get_rng_state().clone()
    torch.manual_seed(2026110661);new=ContextualResidueHead(12,('site','bond'),
        parameterization_policy='mup-residue-v1',base_width=8,output_mult=1.5)
    assert torch.equal(rng,torch.get_rng_state())
    for name,p in old.named_parameters():
        factor=math.sqrt(12/8) if name in ('context.bias','local.bias','output.weight','output.bias') else 1
        assert torch.equal(dict(new.named_parameters())[name],p*factor)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_complete_residue_convolution_readout_equations_and_all_gradients(dtype):
    h=make(16).head.to(dtype=dtype)
    x=torch.linspace(-1,1,80,dtype=dtype).reshape(5,16).requires_grad_()
    c=torch.linspace(-.5,.5,16,dtype=dtype).requires_grad_()
    p=torch.arange(5,dtype=dtype)[:,None]
    f=torch.exp(-torch.arange(0,16,2,dtype=dtype)*math.log(10000.)/16)
    pos=torch.stack((torch.sin(p*f),torch.cos(p*f)),dim=-1).flatten(1)
    a=x+pos+F.linear(c,h.context.weight,h.context.bias)
    features=a+F.gelu(F.conv1d(a.T[None],h.local.weight,h.local.bias,padding=1)[0].T)
    expected=F.linear(features*(1.5/(16/8)),h.output.weight,h.output.bias)
    actual=h(x,c)
    tol=2e-5 if dtype==torch.float32 else 1e-12
    torch.testing.assert_close(actual,expected,rtol=tol,atol=tol)
    torch.testing.assert_close(h.encode_features(x,c),features,rtol=0,atol=0)
    params=(x,c,*h.parameters())
    for ga,gb in zip(torch.autograd.grad(actual.sum(),params,retain_graph=True),
                     torch.autograd.grad(expected.sum(),params)):
        torch.testing.assert_close(ga,gb,rtol=tol,atol=tol)


def test_complete_family_native_conv_axis_and_optimizer_no_updates():
    ms=[make(w) for w in (8,12,16)]
    family=residue_mup_width_family(*ms,identifier_axis=('protein-coordinate',))
    for member,m in zip(('base','delta','target'),ms):
        roles=residue_mup_parameter_roles(m,identifier_axis=('protein-coordinate',))
        r=roles['head.local.weight']
        assert r.role=='hidden' and tuple(a.kind for a in r.axes)==('width','width','finite')
        opt=MuAdamW(m,family,roles,base_lr=.002,weight_decay=.01,decay_aliases=(),member=member)
        assert not opt.state
    assert family.total_numel==tuple(sum(p.numel() for p in m.parameters()) for m in ms)


def test_legacy_owner_refuses_new_shared_policy_before_rng_or_shared_mutation():
    shared=annotation(12);rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match='explicit'):
        SharedResidueModel(shared,('site',))
    assert torch.equal(rng,torch.get_rng_state())


def test_configuration_drift_and_checkpoint_preflight_before_any_tensor_copy():
    m=make(12);sd=copy.deepcopy(m.state_dict());before=copy.deepcopy(m.state_dict())
    sd['head._extra_state']['mup']['base_width']=4
    sd['shared.payload.embedding.weight'].fill_(99)
    with pytest.raises(ValueError,match='configuration|policy'):
        m.load_state_dict(sd)
    for k,v in before.items():
        if isinstance(v,torch.Tensor):assert torch.equal(v,m.state_dict()[k])
    m.head.base_width=4
    with pytest.raises(ValueError,match='drift'):
        residue_mup_parameter_roles(m,identifier_axis=('protein-coordinate',))


@pytest.mark.parametrize('change',['axis','conv','policy'])
def test_family_refuses_source_or_operator_drift(change):
    ms=[make(w) for w in (8,12,16)]
    if change=='axis':ms[-1].head.feature_axis=('bond','site')
    elif change=='conv':ms[-1].head.local.padding=(0,)
    else:ms[-1].head.parameterization_policy='legacy'
    with pytest.raises(ValueError):residue_mup_width_family(*ms,identifier_axis=('protein-coordinate',))
