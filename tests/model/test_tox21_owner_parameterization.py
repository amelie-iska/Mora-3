"""Exact native assay value/mask ownership without scaling or computation changes."""
import hashlib
import json
import pytest
import torch
from torch import nn
from test_annotation_parameterization import model as annotation
from test_physical_owner_parameterization import model as physical
from mora3.model.tox21 import SharedTox21Model
from mora3.data.tox21 import ASSAYS


def inventory(m):
    from mora3.model.parameterization import tox21_parameter_ledger
    return tox21_parameter_ledger(m, identifier_axis=('protein',))


@pytest.mark.parametrize('factory',[annotation,physical])
@pytest.mark.parametrize('width',[8,12,16])
@pytest.mark.parametrize('device',['cpu','cuda'])
def test_independent_counts_axes_and_readonly(factory,width,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(29401)
    m=SharedTox21Model(factory(width)).to(device).eval()
    before={n:t.clone() for n,t in m.named_parameters()}
    rng=torch.get_rng_state().clone()
    gpu=torch.cuda.get_rng_state().clone() if device=='cuda' else None
    ledger=inventory(m);w=width
    shared=w*(w+7)+2*(6*w*w+10*w)+519*w+w*(w+1)+w+4*(w+1)
    if factory is physical:shared+=3*w+7*(w+1)
    assert ledger.total_numel==shared+37*w+12==sum(t.numel() for t in m.parameters())
    rows={a:e for e in ledger.parameters for a in e.aliases}
    assert set(rows)==set(dict(m.named_parameters(remove_duplicate=False)))
    digest=lambda x:hashlib.sha256(json.dumps(x,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
    assert rows['call_input.weight'].axes[1].identity==digest(['tox21.condition-value-mask.v1',tuple(('value',a) for a in ASSAYS)+tuple(('known',a) for a in ASSAYS)])
    assert rows['output.weight'].axes[0].identity==digest(['tox21.source-call.v1',ASSAYS])
    assert rows['call_input.weight'].axes[1]!=rows['output.weight'].axes[0]
    assert rows['call_input.weight'].axes[0]==rows['output.weight'].axes[1]
    assert rows['call_input.weight'].role=='input' and rows['output.weight'].role=='output'
    assert torch.equal(rng,torch.get_rng_state()) and not m.training
    if gpu is not None:assert torch.equal(gpu,torch.cuda.get_rng_state())
    assert all(torch.equal(t,before[n]) and t.grad is None for n,t in m.named_parameters())
    from mora3.model.parameterization import width_family
    with pytest.raises(ValueError,match='custom'):width_family(ledger,ledger,ledger)


def test_compatible_bias_tie_counts_once():
    m=SharedTox21Model(annotation())
    m.call_input.bias=m.shared.payload.projection.bias
    ledger=inventory(m)
    rows={a:e for e in ledger.parameters for a in e.aliases}
    assert rows['call_input.bias'].aliases==('call_input.bias','shared.payload.projection.bias')
    assert ledger.total_numel==sum(t.numel() for t in m.parameters())


@pytest.mark.parametrize('drift',['extra-module','extra-parameter','buffer','child-buffer','subclass','shared',
    'input-size','output-size','shape','no-bias','linear-subclass','child-module','parameter-alias-conflict'])
def test_configuration_and_ownership_drift_refuses(drift):
    m=SharedTox21Model(annotation(12))
    if drift=='extra-module':m.extra=nn.Identity()
    if drift=='extra-parameter':m.extra=nn.Parameter(torch.ones(1))
    if drift=='buffer':m.register_buffer('extra',torch.ones(1))
    if drift=='child-buffer':m.call_input.register_buffer('extra',torch.ones(1))
    if drift=='subclass':
        class Other(SharedTox21Model):pass
        m.__class__=Other
    if drift=='shared':m.shared=nn.Identity()
    if drift=='input-size':m.call_input.in_features=23
    if drift=='output-size':m.output.out_features=11
    if drift=='shape':m.call_input.weight=nn.Parameter(torch.ones(12,23))
    if drift=='no-bias':m.output.bias=None
    if drift=='linear-subclass':
        class Other(nn.Linear):pass
        m.call_input.__class__=Other
    if drift=='child-module':m.output.extra=nn.Identity()
    if drift=='parameter-alias-conflict':m.output.weight=m.call_input.weight
    with pytest.raises(ValueError):inventory(m)
