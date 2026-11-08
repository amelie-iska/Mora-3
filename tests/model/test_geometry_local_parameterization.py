"""Native geometry local ownership; independently checked finite axes/counts."""
import hashlib
import json
import pytest
import torch
from torch import nn
from mora3.model.cooperative_training import CooperativeModel
from mora3.model.geometry_pixel_fusion import GeometryPixelFusion


def roles(m, ids=('object:0','object:1','object:2')):
    from mora3.model.parameterization import geometry_fusion_local_parameter_roles
    return geometry_fusion_local_parameter_roles(m, identifier_axis=ids)


def model(w=8):
    return GeometryPixelFusion(CooperativeModel(('A','B'),width=w).image,id_width=3)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_local_count_axes_and_readonly(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('no CUDA')
    from mora3.model.parameterization import parameter_ledger
    torch.manual_seed(28701)
    for w in (8,12,16):
        m=model(w).to(device).eval()
        before={n:p.clone() for n,p in m.named_parameters()}
        rng=torch.random.get_rng_state().clone()
        gpu=torch.cuda.get_rng_state().clone() if device=='cuda' else None
        r=roles(m)
        assert set(r)=={'geometry.payload.embedding.weight','geometry.payload.projection.weight',
                       'geometry.payload.projection.bias','geometry.graph.incidence.weight'}
        params=dict(m.named_parameters(remove_duplicate=False))
        assert sum(params[n].numel() for n in r)==w*w+519*w+6*w
        assert r['geometry.payload.projection.weight'].role=='custom'
        assert r['geometry.payload.embedding.weight'].axes[0].kind=='finite'
        raw=json.dumps(['tokengt.ordered-incidence.v1',tuple(s+':'+i for s in ('left','right')
            for i in ('object:0','object:1','object:2'))],ensure_ascii=True,separators=(',',':'))
        assert r['geometry.graph.incidence.weight'].axes[1].identity==hashlib.sha256(raw.encode()).hexdigest()
        assert m.image.backbone.feature is m.geometry.graph.feature
        assert roles(m,('object:1','object:0','object:2'))!=r
        assert torch.equal(rng,torch.random.get_rng_state())
        if gpu is not None:assert torch.equal(gpu,torch.cuda.get_rng_state())
        for n,p in m.named_parameters():assert torch.equal(p,before[n]) and p.grad is None
        assert not m.training


@pytest.mark.parametrize('drift',['root','extra','shared','config','fusionconfig','rank','projection','pooling','tokenizer','graph','image','incidence'])
def test_unreviewed_drift_refuses(drift):
    import copy
    m=model()
    if drift=='root':m=nn.ModuleDict({'x':m})
    if drift=='extra':m.extra=nn.Identity()
    if drift=='shared':m.geometry.graph.feature=copy.deepcopy(m.image.backbone.feature)
    if drift=='config':m.geometry.config['pooling']='invented'
    if drift=='fusionconfig':m.config['training_admission']=True
    if drift=='rank':m.geometry.config['id_width']=4
    if drift=='projection':m.geometry.payload.projection=nn.Linear(8,8)
    if drift=='pooling':m.geometry.payload.pooling='mean'
    if drift=='tokenizer':m.geometry.payload.codec.tokenizer._controls=('invented',)
    if drift=='graph':m.geometry.graph.extra=nn.Identity()
    if drift=='image':m.image.width=12
    if drift=='incidence':m.geometry.graph.incidence=nn.Linear(6,8,bias=True)
    with pytest.raises(ValueError):roles(m)


@pytest.mark.parametrize('ids',[(),['a','b','c'],('a','a','c'),('a','','c'),('a','b')])
def test_invalid_identifier_namespace_refuses(ids):
    with pytest.raises(ValueError):roles(model(),ids)
