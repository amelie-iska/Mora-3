"""Exact residue composition, complete aliases and independent counts."""
import pytest
import torch
from torch import nn
from test_physical_owner_parameterization import model as physical
from mora3.model.annotation_multihead import SharedAnnotationModel
from mora3.model.residue_objective import SharedResidueModel


def model(width=8):
    owner=physical(width)
    return SharedResidueModel(owner,('site','domain'))


def inventory(m,ids=('protein',)):
    from mora3.model.parameterization import residue_parameter_ledger
    return residue_parameter_ledger(m,identifier_axis=ids)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('width',[8,12,16])
def test_shared_physical_owner_counts_and_readonly(device,width):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(27801)
    a,b=model(width),model(width);b.shared=a.shared
    m=nn.ModuleDict(dict(a=a,b=b)).to(device).eval()
    before={n:p.clone() for n,p in m.named_parameters()}
    rng=torch.random.get_rng_state().clone()
    gpu=torch.cuda.get_rng_state().clone() if device=='cuda' else None
    ledger=inventory(m);w=width
    backbone=w*(w+7)+2*(6*w*w+10*w)
    physical=519*w+w*(w+1)+w+4*(w+1)+3*w+7*(w+1)
    residue=4*w*w+2*w+2*(w+1)
    assert ledger.total_numel==backbone+physical+2*residue==sum(p.numel() for p in m.parameters())
    rows={n:e for e in ledger.parameters for n in e.aliases}
    assert set(rows)==set(dict(m.named_parameters(remove_duplicate=False)))
    assert rows['a.shared.physical.query.weight'].aliases==('a.shared.physical.query.weight','b.shared.physical.query.weight')
    assert rows['a.head.local.weight'].role=='custom'
    assert rows['a.head.context.weight'].axes[0]==rows['a.shared.backbone.feature.weight'].axes[0]
    assert torch.equal(rng,torch.random.get_rng_state()) and not m.training
    if gpu is not None:assert torch.equal(gpu,torch.cuda.get_rng_state())
    for n,p in m.named_parameters():assert torch.equal(p,before[n]) and p.grad is None
    from mora3.model.parameterization import width_family
    with pytest.raises(ValueError,match='custom'):width_family(ledger,ledger,ledger)


def test_annotation_and_physical_shared_backbone_and_tied_head():
    a=model(); owner=a.shared
    from mora3.model.annotation_vocabulary import AnnotationVocabulary
    import json
    vocab=AnnotationVocabulary(json.dumps(dict(learned_hex=['4142','4344'],controls=['pad','node','edge'])))
    b=SharedResidueModel(SharedAnnotationModel(vocab,('GO:0000001',),('1.1.1.1',),width=8),('site','domain'))
    b.shared.backbone=owner.backbone;b.head=a.head
    m=nn.ModuleDict(dict(a=a,b=b));ledger=inventory(m)
    assert ledger.total_numel==sum(p.numel() for p in m.parameters())
    rows={n:e for e in ledger.parameters for n in e.aliases}
    assert rows['a.head.output.weight'].aliases==('a.head.output.weight','b.head.output.weight')
    assert inventory(a).total_numel==sum(p.numel() for p in a.parameters())


@pytest.mark.parametrize('drift',['empty','root','subclass','owner','extra-module','extra-parameter','width','backbone','tie','axis','ids'])
def test_invalid_composition_refuses(drift):
    a,b=model(),model();b.shared.backbone=a.shared.backbone
    m=nn.ModuleDict(dict(a=a,b=b));ids=('protein',)
    if drift=='empty':m=nn.ModuleDict()
    if drift=='root':m=nn.Sequential(a)
    if drift=='subclass':
        class Derived(SharedResidueModel):pass
        m=Derived(a.shared,('site','domain'))
    if drift=='owner':a.shared=nn.Identity()
    if drift=='extra-module':a.extra=nn.Identity()
    if drift=='extra-parameter':m.extra=nn.Parameter(torch.ones(1))
    if drift=='width':a.head=type(a.head)(12,('site','domain'))
    if drift=='backbone':b.shared.backbone=model().shared.backbone
    if drift=='tie':b.head.feature_axis=('domain','site');b.head.output.weight=a.head.output.weight
    if drift=='axis':a.head.feature_axis=('site','site')
    if drift=='ids':ids=['protein']
    with pytest.raises(ValueError):inventory(m,ids)
