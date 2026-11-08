"""Native TokenGT roles bind ordered inputs without guessing custom scaling."""
import dataclasses
import pytest
import torch
from torch import nn
from mora3.model.tokengt import TokenGT
from mora3.model.parameterization import width_family


def mapped(model, features=('mass','charge','type'), identifiers=('qr0','qr1')):
    from mora3.model.parameterization import tokengt_parameter_ledger
    return tokengt_parameter_ledger(model, feature_axis=features, identifier_axis=identifiers)


@pytest.mark.parametrize('device', ['cpu','cuda'])
@pytest.mark.parametrize('attention', ['vanilla','tropical-code','tropical-blockwise'])
def test_complete_native_alias_map_is_read_only(device, attention):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('no CUDA')
    torch.manual_seed(27001)
    m=TokenGT(3,2,width=8,heads=2,layers=2,ffn_width=12,attention=attention).to(device)
    m.eval();before={n:p.clone() for n,p in m.named_parameters()}
    rng=torch.random.get_rng_state().clone()
    ledger=mapped(m); rows={a:r for r in ledger.parameters for a in r.aliases}
    assert set(rows)==set(dict(m.named_parameters(remove_duplicate=False)))
    assert ledger.total_numel==sum(p.numel() for p in m.parameters())
    assert rows['feature.weight'].role=='input'
    assert tuple(a.kind for a in rows['feature.weight'].axes)==('width','finite')
    assert rows['role.weight'].role=='embedding'
    assert rows['incidence.weight'].role=='custom'
    assert rows['layers.0.ffn.0.weight'].shape==(12,8)
    assert [a.identity for a in rows['layers.0.ffn.0.weight'].axes]==['ffn','hidden']
    assert rows['layers.1.norm2.weight'].role=='normalization'
    custom='layers.0.attention.qkv.weight' if attention=='vanilla' else 'layers.0.attention.query'
    assert rows[custom].role=='custom'
    assert torch.equal(rng,torch.random.get_rng_state()) and not m.training
    for n,p in m.named_parameters(): assert torch.equal(before[n],p) and p.grad is None
    with pytest.raises(dataclasses.FrozenInstanceError): ledger.total_numel=0


def test_ordered_input_identity_and_widths_are_distinct():
    m=TokenGT(3,2,width=8,heads=2,layers=1,ffn_width=12)
    a=mapped(m);b=mapped(m,features=('charge','mass','type'))
    get=lambda ledger,name:next(r for r in ledger.parameters if name in r.aliases)
    assert get(a,'feature.weight').axes[1]!=get(b,'feature.weight').axes[1]
    c=mapped(m,identifiers=('qr1','qr0'))
    assert get(a,'incidence.weight').axes[1]!=get(c,'incidence.weight').axes[1]
    with pytest.raises(ValueError,match='custom'):width_family(a,a,a)


@pytest.mark.parametrize('bad', ['feature-size','id-size','duplicate','list','empty','extra','subclass','module','shape'])
def test_unreviewed_native_structures_or_axis_claims_refuse(bad):
    m=TokenGT(3,2,width=8,heads=2,layers=1,ffn_width=12)
    kwargs={}
    if bad=='feature-size':kwargs['features']=('a','b')
    if bad=='id-size':kwargs['identifiers']=('a',)
    if bad=='duplicate':kwargs['features']=('a','a','b')
    if bad=='list':kwargs['features']=['a','b','c']
    if bad=='empty':kwargs['features']=('a','','c')
    if bad=='extra':m.extra=nn.Parameter(torch.zeros(1))
    if bad=='subclass':
        class Altered(TokenGT): pass
        m=Altered(3,2,width=8,heads=2,layers=1)
    if bad=='module':m.layers[0].ffn[1]=nn.ReLU()
    if bad=='shape':m.layers[0].ffn[2]=nn.Linear(11,8)
    with pytest.raises(ValueError): mapped(m,**kwargs)
