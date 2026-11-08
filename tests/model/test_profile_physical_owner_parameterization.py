"""Physical annotation/profile composition preserves exact ownership and axes."""
import pytest
import torch
from torch import nn
from test_physical_owner_parameterization import model as physical
from test_profile_owner_parameterization import inventory
from mora3.model.molecule_profile import SharedMoleculeProfileModel


def model(width=8):
    return SharedMoleculeProfileModel(physical(width), {
        name:tuple(f'{name}:{i}' for i in range(n))
        for name,n in [('CellPainting',1783),('L1000',977)]})


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('width',[8,12,16])
def test_complete_physical_profile_count_and_readonly(device,width):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(29001)
    m=model(width).to(device).eval()
    before={n:p.clone() for n,p in m.named_parameters()}
    rng=torch.random.get_rng_state().clone()
    gpu=torch.cuda.get_rng_state().clone() if device=='cuda' else None
    ledger=inventory(m);w=width
    expected=w*(w+7)+2*(6*w*w+10*w)+519*w+w*(w+1)+w+4*(w+1)+3*w+7*(w+1)+5520*w+2*w+2760
    assert ledger.total_numel==expected==sum(p.numel() for p in m.parameters())
    rows={a:e for e in ledger.parameters for a in e.aliases}
    assert set(rows)==set(dict(m.named_parameters(remove_duplicate=False)))
    assert rows['shared.physical.query.weight'].axes[1]==rows['profile_input.L1000.weight'].axes[0]
    assert torch.equal(rng,torch.random.get_rng_state()) and not m.training
    if gpu is not None:assert torch.equal(gpu,torch.cuda.get_rng_state())
    for n,p in m.named_parameters():assert torch.equal(before[n],p) and p.grad is None
    from mora3.model.parameterization import width_family
    with pytest.raises(ValueError,match='custom'):width_family(ledger,ledger,ledger)


def test_compatible_cross_owner_tie_is_counted_once():
    m=model()
    m.output['CellPainting'].bias=m.shared.payload.projection.bias
    # Deliberately incompatible shape/role must not become an accepted alias.
    with pytest.raises(ValueError):inventory(m)
    m=model()
    m.profile_input['L1000'].bias=m.shared.payload.projection.bias
    ledger=inventory(m)
    rows={a:e for e in ledger.parameters for a in e.aliases}
    assert rows['shared.payload.projection.bias'].aliases==('profile_input.L1000.bias','shared.payload.projection.bias')
    assert ledger.total_numel==sum(p.numel() for p in m.parameters())


@pytest.mark.parametrize('drift',['extra','width','query-shape','parameter-order','method-order','subclass'])
def test_physical_drift_and_conflicting_alias_refuse(drift):
    m=model()
    if drift=='extra':m.shared.physical.extra=nn.Parameter(torch.ones(1))
    if drift=='width':m.shared.physical.query=nn.Embedding(3,12)
    if drift=='query-shape':m.shared.physical.query.weight=nn.Parameter(torch.ones(4,8))
    if drift=='parameter-order':m.shared.physical.parameter_axis=('pH','mass','temperature')
    if drift=='method-order':m.shared.physical.method_axis=('ESI','MALDI','other')
    if drift=='subclass':
        class Other(type(m.shared)):pass
        m.shared.__class__=Other
    with pytest.raises(ValueError):inventory(m)
