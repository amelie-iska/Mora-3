"""ADMET finite partitions are metadata, never source admission or scaling."""
import copy
import hashlib
import json
from dataclasses import replace
import pytest
import torch
from torch import nn
from test_physical_owner_parameterization import model as physical
from test_admet_model import setup
from mora3.data.admet import AssayContext
from mora3.model.admet_model import SharedAdmetModel


def make(width=8):
    _,v,c,_,_=setup('cpu','regression')
    _,b,d,_,_=setup('cpu','binary')
    return SharedAdmetModel(physical(width),(c,d)),(v,b),(c,d)


def inventory(m):
    from mora3.model.parameterization import admet_parameter_ledger
    return admet_parameter_ledger(m,identifier_axis=('protein',))


@pytest.mark.parametrize('width',[8,12])
@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_counts_finite_partition_order_and_numerical_nonmutation(width,device,dtype):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(36001)
    m,views,contracts=make(width);m=m.to(device=device,dtype=dtype).eval()
    before=copy.deepcopy(m.state_dict());out=m(views,contracts)
    params=tuple(m.parameters());loss=sum(x.square() for p in out for x in p.values())
    gradients=torch.autograd.grad(loss,params,allow_unused=True)
    rng=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone() if device=='cuda' else None
    ledger=inventory(m)
    w=width
    shared=w*(w+7)+2*(6*w*w+10*w)+519*w+w*(w+1)+w+4*(w+1)
    assert ledger.total_numel==shared+3*w+7*(w+1)+3*(w+1)==sum(p.numel() for p in params)
    rows={a:e for e in ledger.parameters for a in e.aliases}
    assert set(rows)==set(dict(m.named_parameters(remove_duplicate=False)))
    for c in contracts:
        names=('logit',) if c.definition.objective_kind=='binary' else ('mean','raw_scale')
        digest=hashlib.sha256(json.dumps(['admet.partition-output.v1',c.partition_key,names],separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
        e=rows['heads.'+c.partition_key+'.weight']
        assert e.role=='output' and e.axes[0].kind=='finite' and e.axes[0].identity==digest
        assert e.axes[1]==rows['shared.go.output.weight'].axes[1]
        assert rows['heads.'+c.partition_key+'.bias'].axes==(e.axes[0],)
    assert torch.equal(rng,torch.get_rng_state())
    if gpu is not None:assert torch.equal(gpu,torch.cuda.get_rng_state())
    after=m(views,contracts)
    for x,y in zip(out,after):
        for key in x:torch.testing.assert_close(x[key],y[key],rtol=0,atol=0)
    grad2=torch.autograd.grad(sum(x.square() for p in after for x in p.values()),params,allow_unused=True)
    for a,b in zip(gradients,grad2):
        assert (a is None)==(b is None)
        if a is not None:torch.testing.assert_close(a,b,rtol=0,atol=0)
    for n,t in m.state_dict().items():
        if isinstance(t,torch.Tensor):assert torch.equal(t,before[n])
        else:assert t==before[n]
    assert not m.training and all(p.grad is None for p in params)
    for v,c,p in zip(views,contracts,out):
        with pytest.raises(ValueError,match='admission'):m.observed_loss(p,v,c)


def test_partition_context_binding_and_compatible_shared_alias():
    m,_,contracts=make();c=contracts[0]
    other=replace(c,assay_context=AssayContext.from_mapping({'species':'distinct'}))
    n=SharedAdmetModel(m.shared,(other,contracts[1]))
    assert inventory(n)!=inventory(m)
    # Two declared ADMET owners can share one annotation owner without double counting.
    from mora3.model.parameterization import compose_owner_parameter_ledgers
    root=nn.ModuleDict({'a':m,'b':n})
    ledger=compose_owner_parameter_ledgers(root,{'a':inventory(m),'b':inventory(n)})
    assert ledger.total_numel==sum(p.numel() for p in root.parameters())
    row=next(e for e in ledger.parameters if 'a.shared.go.output.weight' in e.aliases)
    assert row.aliases==('a.shared.go.output.weight','b.shared.go.output.weight')


@pytest.mark.parametrize('drift',['extra-module','extra-param','buffer','heads-buffer','heads-param',
    'empty','missing','foreign-key','bad-contract','unresolved','class','shape','size','bias',
    'child','child-buffer','child-subclass','root-subclass','shared-type','conflicting-tie'])
def test_configuration_drift_refuses(drift):
    m,_,contracts=make();c=contracts[0];h=m.heads[c.partition_key]
    if drift=='extra-module':m.extra=nn.Identity()
    elif drift=='extra-param':m.extra=nn.Parameter(torch.ones(1))
    elif drift=='buffer':m.register_buffer('x',torch.ones(1))
    elif drift=='heads-buffer':m.heads.register_buffer('x',torch.ones(1))
    elif drift=='heads-param':m.heads.register_parameter('x',nn.Parameter(torch.ones(1)))
    elif drift=='empty':m.contracts={}
    elif drift=='missing':del m.heads[c.partition_key]
    elif drift=='foreign-key':m.contracts['foreign']=m.contracts.pop(c.partition_key)
    elif drift=='bad-contract':m.contracts[c.partition_key]=object()
    elif drift=='unresolved':m.contracts[c.partition_key]=replace(c,definition=replace(c.definition,objective_kind='unresolved'))
    elif drift=='class':m.contracts[c.partition_key]=replace(c,molecule_class=None)
    elif drift=='shape':h.weight=nn.Parameter(torch.zeros(2,9))
    elif drift=='size':h.out_features=1
    elif drift=='bias':h.bias=None
    elif drift=='child':h.extra=nn.Identity()
    elif drift=='child-buffer':h.register_buffer('x',torch.ones(1))
    elif drift=='child-subclass':
        class Other(nn.Linear):pass
        h.__class__=Other
    elif drift=='root-subclass':
        class Other(SharedAdmetModel):pass
        m.__class__=Other
    elif drift=='shared-type':m.shared=nn.Identity()
    else:h.weight=m.shared.go.output.weight
    with pytest.raises(ValueError):inventory(m)


@pytest.mark.parametrize('drift',['extra','width','query-shape','parameter-order','method-order','subclass'])
def test_physical_owner_drift_refuses(drift):
    m,_,_=make()
    if drift=='extra':m.shared.physical.extra=nn.Parameter(torch.ones(1))
    if drift=='width':m.shared.physical.query=nn.Embedding(3,12)
    if drift=='query-shape':m.shared.physical.query.weight=nn.Parameter(torch.ones(4,8))
    if drift=='parameter-order':m.shared.physical.parameter_axis=('pH','mass','temperature')
    if drift=='method-order':m.shared.physical.method_axis=('ESI','MALDI','other')
    if drift=='subclass':
        class Other(type(m.shared)):pass
        m.shared.__class__=Other
    with pytest.raises(ValueError):inventory(m)


def test_physical_alias_axes_and_conflicting_physical_tie():
    from mora3.model.parameterization import compose_owner_parameter_ledgers, width_family
    a,_,c=make();b=SharedAdmetModel(a.shared,c)
    root=nn.ModuleDict({'a':a,'b':b})
    ledger=compose_owner_parameter_ledgers(root,{'a':inventory(a),'b':inventory(b)})
    rows={alias:e for e in ledger.parameters for alias in e.aliases}
    assert rows['a.shared.physical.query.weight'].aliases==('a.shared.physical.query.weight','b.shared.physical.query.weight')
    assert rows['a.shared.physical.query.weight'].axes[1]==rows['a.heads.'+c[0].partition_key+'.weight'].axes[1]
    assert ledger.total_numel==sum(p.numel() for p in root.parameters())
    with pytest.raises(ValueError,match='custom'):width_family(ledger,ledger,ledger)
    a.heads[c[0].partition_key].weight=a.shared.physical.unit.weight
    with pytest.raises(ValueError):inventory(a)
