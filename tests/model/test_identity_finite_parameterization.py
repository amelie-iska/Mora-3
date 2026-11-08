"""Fixed atom identity channels must never be declared scalable hidden widths."""
import copy
import hashlib
import json
import pytest
import torch
from torch import nn
from mora3.model.current_frame_velocity import SplitContextualCoefficients
from mora3.model.identity_noise_vector import IDENTITY_CONTEXT_AXIS
from mora3.model.parameterization import parameter_ledger, ParameterRole, Axis


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_fixed_inventory_preserves_equation_state_and_rng(device,dtype):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.manual_seed(35501)
    m=SplitContextualCoefficients(IDENTITY_CONTEXT_AXIS,max_atoms=16).to(device=device,dtype=dtype)
    state=copy.deepcopy(m.state_dict());rng=torch.get_rng_state().clone()
    roles=m.identity_parameter_declarations()
    assert torch.equal(rng,torch.get_rng_state())
    ledger=parameter_ledger(m,roles)
    assert ledger.total_numel==248
    expected=hashlib.sha256(json.dumps(['current-vector-atom-identity-context.v1',IDENTITY_CONTEXT_AXIS],separators=(',', ':')).encode()).hexdigest()
    for name in ('receiver.weight','neighbor.weight'):
        assert roles[name].role=='custom'
        assert all(a.kind=='finite' for a in roles[name].axes)
        assert roles[name].axes[1].identity==expected
    assert roles['geometry.weight']==m.parameter_declarations(width_identity='hidden')['geometry.weight']
    g=torch.arange(18,device=device,dtype=dtype).reshape(3,3,2)/18
    h=torch.arange(369,device=device,dtype=dtype).reshape(3,123)/369
    actual=m(g,h,origin='synthetic_literal',context_axis=IDENTITY_CONTEXT_AXIS)
    reference=g@m.geometry.weight.T+(h@m.receiver.weight.T)[:,None,:]+(h@m.neighbor.weight.T)[None,:,:]
    torch.testing.assert_close(actual,reference)
    for k,v in state.items():
        assert torch.equal(v,m.state_dict()[k]) if torch.is_tensor(v) else v==m.state_dict()[k]
    assert all(p.grad is None for p in m.parameters())


@pytest.mark.parametrize('axis',[tuple(reversed(IDENTITY_CONTEXT_AXIS)),IDENTITY_CONTEXT_AXIS[:-1],tuple('x'+x for x in IDENTITY_CONTEXT_AXIS)])
def test_noncanonical_axis_refuses(axis):
    m=SplitContextualCoefficients(axis,max_atoms=16)
    with pytest.raises(ValueError):m.identity_parameter_declarations()


def test_alias_count_and_conflicting_semantics():
    m=SplitContextualCoefficients(IDENTITY_CONTEXT_AXIS,max_atoms=16)
    roles=m.identity_parameter_declarations()
    m.neighbor.weight=m.receiver.weight
    assert parameter_ledger(m,roles).total_numel==125
    root=nn.ModuleDict({'first':m,'alias':m})
    declarations={f'{prefix}.{k}':v for prefix in ('first','alias') for k,v in roles.items()}
    assert parameter_ledger(root,declarations).total_numel==125
    declarations['alias.receiver.weight']=ParameterRole('output',(roles['receiver.weight'].axes[0],Axis('width','hidden')))
    with pytest.raises(ValueError):parameter_ledger(root,declarations)


@pytest.mark.parametrize('drift',['buffer','bias','shape','module','subclass'])
def test_owner_drift_refuses(drift):
    m=SplitContextualCoefficients(IDENTITY_CONTEXT_AXIS,max_atoms=16)
    if drift=='buffer':m.receiver.register_buffer('extra',torch.ones(1))
    elif drift=='bias':m.receiver.bias=nn.Parameter(torch.ones(1))
    elif drift=='shape':m.receiver.weight=nn.Parameter(torch.ones(1,122))
    elif drift=='module':m.extra=nn.Identity()
    else:
        class Other(SplitContextualCoefficients):pass
        m.__class__=Other
    with pytest.raises(ValueError):m.identity_parameter_declarations()
