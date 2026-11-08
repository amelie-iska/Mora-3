"""Packed mixed finite/hidden coefficient inventory does not qualify scaling."""
import copy
import pytest
import torch
from torch import nn
from mora3.model.current_frame_velocity import ContextualCurrentFrameVelocityReadout as Readout
from mora3.model.invariant_atom_projection import InvariantAtomProjection
from mora3.model.parameterization import parameter_ledger


def make(w=8):
    return Readout(tuple(f'h:{i}' for i in range(w)),max_atoms=16)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('width',[8,12,16])
def test_complete_composition(width,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(33201)
    m=make(width).to(device); state=copy.deepcopy(m.state_dict())
    rng=torch.get_rng_state().clone()
    r=m.parameter_declarations(width_identity='hidden')
    assert torch.equal(rng,torch.get_rng_state())
    assert set(r)=={'coefficients.weight'}
    assert r['coefficients.weight'].role=='custom'
    assert all(a.kind=='finite' for a in r['coefficients.weight'].axes)
    assert parameter_ledger(m,r).total_numel==2+2*width
    p=InvariantAtomProjection(('d','d2','g2'),width,width_identity='hidden').to(device)
    root=nn.ModuleDict({'projection':p,'readout':m,'alias':m})
    roles={f'projection.{k}':v for k,v in p.parameter_declarations().items()}
    for prefix in ('readout','alias'):roles.update({f'{prefix}.{k}':v for k,v in r.items()})
    ledger=parameter_ledger(root,roles)
    assert ledger.total_numel==width*width+6*width+2
    assert len(ledger.parameters)==4
    for k,v in m.state_dict().items():
        assert torch.equal(v,state[k]) if torch.is_tensor(v) else v==state[k]
    assert all(t.grad is None for t in root.parameters())


@pytest.mark.parametrize('change',['root','child','buffer','child-buffer','child-module','extra','bias','shape','metadata','subclass','identity'])
def test_drift_refuses(change):
    m=make();identity='hidden'
    if change=='root':m.extra=nn.Identity()
    elif change=='child':m.coefficients=nn.Identity()
    elif change=='buffer':m.register_buffer('extra',torch.ones(1))
    elif change=='child-buffer':m.coefficients.register_buffer('extra',torch.ones(1))
    elif change=='child-module':m.coefficients.extra=nn.Identity()
    elif change=='extra':m.extra=nn.Parameter(torch.ones(1))
    elif change=='bias':m.coefficients.bias=nn.Parameter(torch.ones(1))
    elif change=='shape':m.coefficients.weight=nn.Parameter(torch.ones(1,17))
    elif change=='metadata':m.coefficients.in_features=17
    elif change=='subclass':
        class Other(nn.Linear):pass
        m.coefficients.__class__=Other
    elif change=='identity':identity=''
    with pytest.raises(ValueError):m.parameter_declarations(width_identity=identity)


def test_axis_identity_binds_order_and_width_identity():
    m=make();r=m.parameter_declarations(width_identity='hidden')
    other=Readout(tuple(reversed(m._context_axis)),max_atoms=16)
    assert r!=other.parameter_declarations(width_identity='hidden')
    assert r!=m.parameter_declarations(width_identity='different')
    assert r!=make(12).parameter_declarations(width_identity='hidden')
