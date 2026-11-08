"""Explicit physical metadata axes; no numeric or generation qualification."""
import hashlib
import json
import pytest
import torch
from torch import nn
from mora3.model.physical_head import StructuredPhysicalHead
from mora3.model.parameterization import Axis, parameter_ledger, width_family


def roles(m):
    from mora3.model.parameterization import physical_head_parameter_roles
    return physical_head_parameter_roles(m)


def head(w=8, parameters=('mass','pH','temperature'), methods=('MALDI','ESI','other')):
    return StructuredPhysicalHead(w, parameters, methods)


def finite(namespace, names):
    return Axis('finite',hashlib.sha256(json.dumps([namespace,names],ensure_ascii=True,
        separators=(',',':')).encode()).hexdigest())


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_semantic_orientation_count_and_readonly(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(27501)
    m=head().to(device)
    before={n:p.clone() for n,p in m.named_parameters()}
    rng=torch.random.get_rng_state().clone()
    gpu=torch.cuda.get_rng_state().clone() if device=='cuda' else None
    r=roles(m); ledger=parameter_ledger(m,r)
    h=Axis('width','hidden')
    assert r['query.weight'].role=='embedding'
    assert r['query.weight'].axes==(finite('physical.parameter-types.v1',m.parameter_axis),h)
    for name,labels in [('unit',('unknown','explicit_source_unit')),
                        ('scope',('unscoped','molecule-scoped')),
                        ('method',m.method_axis)]:
        assert r[name+'.weight'].role=='output'
        assert r[name+'.weight'].axes==(finite('physical.'+name+'-labels.v1',labels),h)
        assert r[name+'.bias'].role=='vector'
        assert r[name+'.bias'].axes==r[name+'.weight'].axes[:1]
    assert ledger.total_numel==3*8+(2+2+3)*(8+1)==sum(p.numel() for p in m.parameters())
    assert torch.equal(rng,torch.random.get_rng_state())
    if gpu is not None:assert torch.equal(gpu,torch.cuda.get_rng_state())
    for n,p in m.named_parameters():assert torch.equal(p,before[n]) and p.grad is None


def test_width_family_and_ordered_finite_identity():
    maps=[roles(head(w)) for w in (8,12,16)]
    assert maps[0]==maps[1]==maps[2]
    family=width_family(*(parameter_ledger(head(w),r) for w,r in zip((8,12,16),maps)))
    assert family.widths==( ('hidden',8,12,16), )
    assert roles(head(parameters=('temperature','pH','mass')))['query.weight']!=maps[0]['query.weight']
    assert roles(head(methods=('ESI','MALDI','other')))['method.weight']!=maps[0]['method.weight']
    assert roles(head(methods=('ESI','MALDI','other')))['unit.weight']==maps[0]['unit.weight']


@pytest.mark.parametrize('drift',['extra','module','linear-subclass','root-subclass','query-shape',
    'unit-shape','scope-bias','method-shape','parameters-list','duplicate','empty-method',
    'config-width','config-axis','config-objective','extra-config'])
def test_drift_refused(drift):
    m=head()
    if drift=='extra':m.register_parameter('extra',nn.Parameter(torch.ones(1)))
    if drift=='module':m.extra=nn.Identity()
    if drift=='linear-subclass':
        class Linear(nn.Linear):pass
        m.unit=Linear(8,2)
    if drift=='root-subclass':
        class Head(StructuredPhysicalHead):pass
        m=Head(8,('mass',),('ESI',))
    if drift=='query-shape':m.query=nn.Embedding(4,8)
    if drift=='unit-shape':m.unit=nn.Linear(9,2)
    if drift=='scope-bias':m.scope.bias=None
    if drift=='method-shape':m.method=nn.Linear(8,2)
    if drift=='parameters-list':m.parameter_axis=list(m.parameter_axis)
    if drift=='duplicate':m.parameter_axis=('mass',)*3
    if drift=='empty-method':m.method_axis=('MALDI','','other')
    if drift=='config-width':m.configuration['width']=9
    if drift=='config-axis':m.configuration['methods']=tuple(reversed(m.method_axis))
    if drift=='config-objective':m.configuration['objective']='numeric'
    if drift=='extra-config':m.configuration['extra']=True
    with pytest.raises(ValueError):roles(m)
