"""Full native ownership and explicit shared response aliases."""
import pytest
import torch
from torch import nn
from mora3.model.sciplex import SciPlexSharedModel
from mora3.model.response_flow import ResponseFlowHead, ResponseFlowDenoiserV2
from mora3.model.response_desired_flow import DesiredResponseFlow


def inventory(m):
    from mora3.model.parameterization import response_parameter_ledger
    return response_parameter_ledger(m)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('attention',['vanilla','tropical-code','tropical-blockwise'])
def test_shared_owner_exact_equation_aliases_and_readonly(device,attention):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('no CUDA')
    torch.manual_seed(27201)
    w,g=8,3
    native=SciPlexSharedModel(('A','B','C'),width=w,attention=attention)
    m=nn.ModuleDict(dict(native=native,v1=ResponseFlowHead(native),v2=ResponseFlowDenoiserV2(native),desired=DesiredResponseFlow(native))).to(device)
    m.eval(); before={n:p.clone() for n,p in m.named_parameters()};rng=torch.random.get_rng_state().clone()
    ledger=inventory(m)
    # Native nonbackbone: two gene matrices, two six-coordinate matrices,
    # scalar exposure projection, edge and two direction vectors.
    nonbackbone=2*w*g+19*w+g+6
    # Native TokenGT F=w,I=2,L=2,ff=w.
    per_layer=(6*w*w+10*w if attention=='vanilla' else 3*w*w+3*(w//2)**2+6*w)
    backbone=w*(w+9)+2*per_layer
    local=2*w*g+3*w+g
    assert ledger.total_numel==nonbackbone+backbone+3*local+4*w*g
    assert ledger.total_numel==sum(p.numel() for p in m.parameters())
    rows={a:e for e in ledger.parameters for a in e.aliases}
    assert set(rows)==set(dict(m.named_parameters(remove_duplicate=False)))
    assert rows['native.edge'].aliases==('desired.flow.native.edge','native.edge','v1.native.edge','v2.native.edge')
    assert rows['native.expression.weight'].axes==tuple(reversed(rows['native.response_head.weight'].axes))
    assert rows['native.backbone.feature.weight'].axes[0]==rows['native.backbone.feature.weight'].axes[1]
    assert rows['native.backbone.feature.weight'].role=='hidden'
    assert torch.equal(rng,torch.random.get_rng_state()) and not m.training
    for n,p in m.named_parameters():assert torch.equal(p,before[n]) and p.grad is None


@pytest.mark.parametrize('kind',[SciPlexSharedModel,ResponseFlowHead,ResponseFlowDenoiserV2,DesiredResponseFlow])
def test_standalone_and_ordered_genes(kind):
    def make(names):
        n=SciPlexSharedModel(names,width=8)
        return n if kind is SciPlexSharedModel else kind(n)
    a,b=inventory(make(('A','B','C'))),inventory(make(('B','A','C')))
    assert a.total_numel==b.total_numel
    assert [e.axes for e in a.parameters]!=[e.axes for e in b.parameters]


@pytest.mark.parametrize('drift',['extra-module','extra-parameter','shape','type','different-owner','conflicting-tie','empty','wrong-root'])
def test_unreviewed_ownership_refuses(drift):
    n=SciPlexSharedModel(('A','B','C'),width=8)
    m=nn.ModuleDict(dict(native=n,flow=ResponseFlowHead(n)))
    if drift=='extra-module':n.extra=nn.Identity()
    if drift=='extra-parameter':n.extra=nn.Parameter(torch.ones(1))
    if drift=='shape':n.intervention=nn.Linear(7,8)
    if drift=='type':n.context=nn.Sequential(n.context)
    if drift=='different-owner':m['other']=ResponseFlowHead(SciPlexSharedModel(('A','B','C'),width=8))
    if drift=='conflicting-tie':n.context.bias=n.response_head.bias # mismatched dimensions
    if drift=='empty':m=nn.ModuleDict()
    if drift=='wrong-root':m=nn.Sequential(n)
    with pytest.raises(ValueError):inventory(m)
