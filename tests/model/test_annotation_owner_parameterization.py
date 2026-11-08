"""Complete annotation ownership: shared native backbone, exact finite axes."""
import pytest
import torch
from torch import nn
from test_annotation_parameterization import model
from mora3.model.tokengt import TokenGT


def inventory(m, identifiers=('protein',)):
    from mora3.model.parameterization import annotation_parameter_ledger
    return annotation_parameter_ledger(m, identifier_axis=identifiers)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('width',[8,12,16])
def test_complete_shared_count_equation_and_readonly(device,width):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(27401)
    a,b=model(width),model(width)
    b.backbone=a.backbone
    m=nn.ModuleDict(dict(a=a,b=b)).to(device).eval()
    before={n:p.clone() for n,p in m.named_parameters()}
    rng=torch.random.get_rng_state().clone()
    gpu=torch.cuda.get_rng_state().clone() if device=='cuda' else None
    ledger=inventory(m)
    w=width
    # F=w,I=1,2 vanilla layers with FFN=w: native count w*(w+7)+2*(6w²+10w).
    backbone=w*(w+7)+2*(6*w*w+10*w)
    local=519*w+w*(w+1)+w+4*(w+1)
    assert ledger.total_numel==backbone+2*local==sum(p.numel() for p in m.parameters())
    rows={a:e for e in ledger.parameters for a in e.aliases}
    assert set(rows)==set(dict(m.named_parameters(remove_duplicate=False)))
    assert rows['a.backbone.graph_token'].aliases==('a.backbone.graph_token','b.backbone.graph_token')
    assert rows['a.backbone.feature.weight'].role=='hidden'
    assert rows['a.backbone.feature.weight'].axes[0]==rows['a.payload.embedding.weight'].axes[1]
    assert rows['a.backbone.feature.weight'].axes[0]==rows['a.backbone.feature.weight'].axes[1]
    assert rows['a.go.output.weight'].aliases==('a.go.output.weight',)
    assert not m.training and torch.equal(rng,torch.random.get_rng_state())
    if gpu is not None:assert torch.equal(gpu,torch.cuda.get_rng_state())
    for n,p in m.named_parameters():assert torch.equal(before[n],p) and p.grad is None
    from mora3.model.parameterization import width_family
    with pytest.raises(ValueError,match='custom'):width_family(ledger,ledger,ledger)


def test_standalone_identity_and_shared_local_tie():
    a,b=model(),model();b.backbone=a.backbone;b.payload=a.payload
    ledger=inventory(nn.ModuleDict(dict(a=a,b=b)))
    rows={a:e for e in ledger.parameters for a in e.aliases}
    assert rows['a.payload.embedding.weight'].aliases==('a.payload.embedding.weight','b.payload.embedding.weight')
    plain=inventory(a)
    assert plain.total_numel==sum(p.numel() for p in a.parameters())
    other=inventory(a,('other-protein',))
    assert plain.total_numel==other.total_numel and plain!=other


@pytest.mark.parametrize('drift',['different-backbone','wrong-root','empty','module','parameter',
    'backbone-shape','width','identifier-list','identifier-rank','identifier-repeat','conflicting-tie'])
def test_unreviewed_composition_refused(drift):
    a,b=model(),model();b.backbone=a.backbone
    m=nn.ModuleDict(dict(a=a,b=b));ids=('protein',)
    if drift=='different-backbone':b.backbone=model().backbone
    if drift=='wrong-root':m=nn.Sequential(a)
    if drift=='empty':m=nn.ModuleDict()
    if drift=='module':a.backbone.extra=nn.Identity()
    if drift=='parameter':m.extra=nn.Parameter(torch.ones(1))
    if drift=='backbone-shape':a.backbone.feature=nn.Linear(7,8,bias=False)
    if drift=='width':a.backbone=b.backbone=TokenGT(12,1,width=12,heads=2)
    if drift=='identifier-list':ids=['protein']
    if drift=='identifier-rank':ids=('protein','other')
    if drift=='identifier-repeat':ids=('protein','protein')
    if drift=='conflicting-tie':
        b.go.axis_ids=('GO:0000002','GO:0000001');b.go.output.weight=a.go.output.weight
    with pytest.raises(ValueError):inventory(m,ids)
