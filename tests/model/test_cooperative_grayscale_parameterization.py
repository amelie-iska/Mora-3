"""Explicit shared rank-two image/response construction ownership."""
import pytest
import torch
from torch import nn
from mora3.model.cooperative_training import CooperativeModel


def ledger(m):
    from mora3.model.parameterization import cooperative_grayscale_parameter_ledger
    return cooperative_grayscale_parameter_ledger(m)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_shared_exact_count_aliases_readonly(device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('no CUDA')
    torch.manual_seed(28601)
    for w in (8,12,16):
        g=3;m=CooperativeModel(('A','B','C'),width=w).to(device).eval()
        before={n:p.clone() for n,p in m.named_parameters()}
        rng=torch.random.get_rng_state().clone()
        gpu=torch.cuda.get_rng_state().clone() if device=='cuda' else None
        x=ledger(m); rows={a:e for e in x.parameters for a in e.aliases}
        pixels=9*(w+51*w*w)+20*w
        backbone=w*(w+9)+2*(6*w*w+10*w)
        response=2*w*g+19*w+g+6
        assert x.total_numel==pixels+backbone+response+3*(2*w+1)
        assert set(rows)==set(dict(m.named_parameters(remove_duplicate=False)))
        assert rows['image.backbone.feature.weight'].aliases==('image.backbone.feature.weight','response.backbone.feature.weight')
        assert rows['image.backbone.feature.weight'].role=='hidden'
        assert rows['image.head.weight'].axes[1].identity=='image.concat-local-context'
        assert torch.equal(rng,torch.random.get_rng_state())
        if gpu is not None: assert torch.equal(gpu,torch.cuda.get_rng_state())
        for n,p in m.named_parameters(): assert torch.equal(before[n],p) and p.grad is None
        assert not m.training


@pytest.mark.parametrize('drift',['root','extra','unshared','config','response_config','normalization','width','head','ids'])
def test_drift_refuses(drift):
    m=CooperativeModel(('A','B','C'),width=8)
    if drift=='root':m=nn.ModuleDict({'owner':m})
    if drift=='extra':m.extra=nn.Identity()
    if drift=='unshared':
        import copy
        m.image.backbone=copy.deepcopy(m.response.backbone)
    if drift=='config':m.config['identifier_policy']='invented'
    if drift=='response_config':m.response.config['attention']='invented'
    if drift=='normalization':m.image.normalization='unsigned-dtype-max'
    if drift=='width':m.image.width=12
    if drift=='head':m.image.head=nn.Conv2d(16,4,1)
    if drift=='ids':m.image.backbone.incidence=nn.Linear(2,8,bias=False)
    with pytest.raises(ValueError):ledger(m)


def test_standalone_rank_one_refusal_preserved():
    from mora3.model.parameterization import morphology_parameter_ledger,width_family
    m=CooperativeModel(('A','B','C'),width=8)
    with pytest.raises(ValueError):morphology_parameter_ledger(m.image)
    with pytest.raises(ValueError,match='custom'):
        width_family(*(ledger(CooperativeModel(('A','B','C'),width=w)) for w in (8,12,16)))
