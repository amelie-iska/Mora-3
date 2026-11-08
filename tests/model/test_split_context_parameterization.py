"""Split finite/hidden ownership inventory; no scaling qualification."""
import copy
import pytest
import torch
from torch import nn
from mora3.model.current_frame_velocity import SplitContextualCoefficients
from mora3.model.invariant_atom_projection import InvariantAtomProjection
from mora3.model.parameterization import parameter_ledger


def make(w=8):
    return SplitContextualCoefficients(tuple(f'h:{i}' for i in range(w)),max_atoms=16)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('width',[8,12,16])
def test_complete_split_composition(width,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(33401)
    m=make(width).to(device);state=copy.deepcopy(m.state_dict())
    rng=torch.get_rng_state().clone()
    roles=m.parameter_declarations(width_identity='hidden')
    assert torch.equal(rng,torch.get_rng_state())
    assert set(roles)=={'geometry.weight','receiver.weight','neighbor.weight'}
    assert roles['geometry.weight'].role=='custom'
    assert [a.kind for a in roles['geometry.weight'].axes]==['finite','finite']
    for name in ('receiver.weight','neighbor.weight'):
        assert roles[name].role=='output'
        assert [a.kind for a in roles[name].axes]==['finite','width']
        assert roles[name].axes[1].identity=='hidden'
    assert roles['receiver.weight']==roles['neighbor.weight']
    assert parameter_ledger(m,roles).total_numel==2+2*width
    p=InvariantAtomProjection(('d','d2','g2'),width,width_identity='hidden').to(device)
    root=nn.ModuleDict({'projection':p,'readout':m,'alias':m})
    declarations={f'projection.{k}':v for k,v in p.parameter_declarations().items()}
    for prefix in ('readout','alias'):
        declarations.update({f'{prefix}.{k}':v for k,v in roles.items()})
    ledger=parameter_ledger(root,declarations)
    assert ledger.total_numel==width*width+6*width+2
    assert len(ledger.parameters)==6
    assert sum(len(e.aliases)==2 for e in ledger.parameters)==3
    for k,v in m.state_dict().items():
        assert torch.equal(v,state[k]) if torch.is_tensor(v) else v==state[k]
    assert all(t.grad is None for t in root.parameters())


@pytest.mark.parametrize('child',['geometry','receiver','neighbor'])
@pytest.mark.parametrize('change',['replace','buffer','module','extra','bias','shape','metadata','subclass'])
def test_child_drift_refuses(child,change):
    m=make();c=getattr(m,child)
    if change=='replace':setattr(m,child,nn.Identity())
    elif change=='buffer':c.register_buffer('extra',torch.ones(1))
    elif change=='module':c.extra=nn.Identity()
    elif change=='extra':c.extra=nn.Parameter(torch.ones(1))
    elif change=='bias':c.bias=nn.Parameter(torch.ones(1))
    elif change=='shape':c.weight=nn.Parameter(torch.ones(1,c.in_features+1))
    elif change=='metadata':c.in_features+=1
    elif change=='subclass':
        class Other(nn.Linear):pass
        c.__class__=Other
    with pytest.raises(ValueError):m.parameter_declarations(width_identity='hidden')


@pytest.mark.parametrize('change',['module','buffer','parameter','identity','subclass'])
def test_root_drift_refuses(change):
    m=make();identity='hidden'
    if change=='module':m.extra=nn.Identity()
    elif change=='buffer':m.register_buffer('extra',torch.ones(1))
    elif change=='parameter':m.extra=nn.Parameter(torch.ones(1))
    elif change=='identity':identity=' '
    else:
        class Other(SplitContextualCoefficients):pass
        m.__class__=Other
    with pytest.raises(ValueError):m.parameter_declarations(width_identity=identity)


def test_axis_order_and_independent_alias_conflicts():
    m=make();r=m.parameter_declarations(width_identity='hidden')
    other=SplitContextualCoefficients(tuple(reversed(m._context_axis)),max_atoms=16)
    assert r!=other.parameter_declarations(width_identity='hidden')
    assert r!=m.parameter_declarations(width_identity='different')
    m.neighbor.weight=m.receiver.weight
    assert parameter_ledger(m,r).total_numel==10
    # Tying finite geometry and hidden axes must be refused even at width two.
    n=make(2);n.geometry.weight=n.receiver.weight
    with pytest.raises(ValueError):n.parameter_declarations(width_identity='hidden')
