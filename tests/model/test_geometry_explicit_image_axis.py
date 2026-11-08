"""Explicit parent coordinates must replace only image incidence declarations."""
import pytest
import torch
from test_geometry_owner_parameterization import model, ledger
from mora3.model.parameterization import geometry_fusion_parameter_ledger

@pytest.mark.parametrize('device', ['cpu','cuda'])
@pytest.mark.parametrize('width', [8,12,16])
def test_explicit_image_axis_changes_only_incidence_metadata(device,width):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('no CUDA')
    torch.manual_seed(30701)
    m=model(width).to(device).eval()
    old=ledger(m)
    before={n:t.clone() for n,t in m.state_dict().items() if torch.is_tensor(t)}
    rng=torch.get_rng_state().clone()
    gpu=torch.cuda.get_rng_state().clone() if device=='cuda' else None
    result=geometry_fusion_parameter_ledger(m,identifier_axis=('object:0','object:1','object:2'),image_identifier_axis=('parent:0','parent:1'))
    assert result.total_numel==old.total_numel
    rows=lambda l:{a:e for e in l.parameters for a in e.aliases}
    a,b=rows(old),rows(result)
    assert set(a)==set(b)
    assert [n for n in a if a[n]!=b[n]]==['image.backbone.incidence.weight']
    reverse=geometry_fusion_parameter_ledger(m,identifier_axis=('object:0','object:1','object:2'),image_identifier_axis=('parent:1','parent:0'))
    assert reverse!=result
    assert ledger(m)==old
    for n,t in m.state_dict().items():
        if torch.is_tensor(t): assert torch.equal(t,before[n])
    assert all(p.grad is None for p in m.parameters()) and not m.training
    assert torch.equal(rng,torch.get_rng_state())
    if gpu is not None: assert torch.equal(gpu,torch.cuda.get_rng_state())

@pytest.mark.parametrize('axis',[(),('a',),('a','b','c'),('a','a'),('a',''),['a','b'],('a',1)])
def test_invalid_explicit_image_axis_refuses(axis):
    with pytest.raises(ValueError):
        geometry_fusion_parameter_ledger(model(),identifier_axis=('x','y','z'),image_identifier_axis=axis)
