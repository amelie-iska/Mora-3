"""Complete geometry fusion ownership preserves shared transforms exactly."""
import copy
import pytest
import torch
from torch import nn
from mora3.model.cooperative_training import CooperativeModel
from mora3.model.geometry_pixel_fusion import GeometryPixelFusion


def model(w=8):
    return GeometryPixelFusion(CooperativeModel(('A','B'),width=w).image,id_width=3)


def ledger(m, ids=('object:0','object:1','object:2')):
    from mora3.model.parameterization import geometry_fusion_parameter_ledger
    return geometry_fusion_parameter_ledger(m,identifier_axis=ids)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_complete_counts_aliases_and_readonly(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('no CUDA')
    torch.manual_seed(28801)
    for w in (8,12,16):
        m=model(w).to(device).eval()
        before={n:p.clone() for n,p in m.named_parameters()}
        rng=torch.random.get_rng_state().clone()
        gpu=torch.cuda.get_rng_state().clone() if device=='cuda' else None
        x=ledger(m); rows={a:e for e in x.parameters for a in e.aliases}
        pixels=9*(w+51*w*w)+20*w
        backbone=w*(w+9)+2*(6*w*w+10*w)
        local=w*w+525*w
        assert x.total_numel==pixels+backbone+3*(2*w+1)+local
        assert set(rows)==set(dict(m.named_parameters(remove_duplicate=False)))
        assert rows['image.backbone.feature.weight'].aliases==('geometry.graph.feature.weight','image.backbone.feature.weight')
        assert rows['geometry.graph.feature.weight'].role=='hidden'
        assert rows['geometry.graph.incidence.weight'].axes!=rows['image.backbone.incidence.weight'].axes
        assert ledger(m,('object:1','object:0','object:2'))!=x
        assert torch.equal(rng,torch.random.get_rng_state())
        if gpu is not None:assert torch.equal(gpu,torch.cuda.get_rng_state())
        for n,p in m.named_parameters():assert torch.equal(p,before[n]) and p.grad is None
        assert not m.training


@pytest.mark.parametrize('drift',['root','extra','unshared','config','image','incidence','rootparam','payload'])
def test_drift_refuses(drift):
    m=model()
    if drift=='root':m=nn.ModuleDict({'owner':m})
    if drift=='extra':m.extra=nn.Identity()
    if drift=='unshared':m.geometry.graph.feature=copy.deepcopy(m.image.backbone.feature)
    if drift=='config':m.config['training_admission']=True
    if drift=='image':m.image.width=12
    if drift=='incidence':m.geometry.graph.incidence=nn.Linear(6,8,bias=True)
    if drift=='rootparam':m.extra=nn.Parameter(torch.zeros(1))
    if drift=='payload':m.geometry.payload.pooling='mean'
    with pytest.raises(ValueError):ledger(m)


def test_custom_scaling_refuses():
    from mora3.model.parameterization import width_family
    with pytest.raises(ValueError,match='custom'):
        width_family(*(ledger(model(w)) for w in (8,12,16)))
