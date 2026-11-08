"""Strict reconstruction roots preserve complete shared ownership without execution."""
import copy
import pytest
import torch
from test_reconstruction_cooperative import build
from test_geometry_reconstruction import build as geometry_build
from test_goa_owner_parameterization import owners
from mora3.model import parameterization as p


def inherited(m):
    return dict(owners(m),goa=p.parameter_ledger(m.goa,p.goa_local_parameter_roles(
        m.goa,axis_ids=m.goa.axis_ids,source_binding=m.goa.get_extra_state())))


def inventory(m,o=None,**kw):
    args=dict(owners=inherited(m) if o is None else o,latent_axis=('x','y'),
              configuration=m.reconstruction.get_extra_state())
    if hasattr(m,'geometry_fusion'):
        args.update(geometry_identifier_axis=tuple(f'object:{i}' for i in range(16)),
                    image_identifier_axis=('context','observation'))
    args.update(kw)
    return p.reconstruction_cooperative_parameter_ledger(m,**args)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('geometry',[False,True])
def test_complete_counts_aliases_and_preservation(tmp_path,device,geometry):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('no CUDA')
    torch.manual_seed(30801)
    m,_=(geometry_build if geometry else build)(tmp_path,device)
    m.eval();m.annotation.train()
    for t in m.parameters():t.grad=torch.ones_like(t)
    state=copy.deepcopy(m.state_dict());modes={n:c.training for n,c in m.named_modules()}
    grads={n:t.grad.clone() for n,t in m.named_parameters()}
    rng=torch.get_rng_state().clone()
    gpu=torch.cuda.get_rng_state().clone() if device=='cuda' else None
    o=inherited(m)
    base={id(t):t for name,c in m.named_children() if name!='reconstruction' for t in c.parameters()}
    result=inventory(m,o)
    assert result.total_numel==sum(t.numel() for t in base.values())+2*4
    assert result.total_numel==sum(t.numel() for t in m.parameters())
    rows={a:e for e in result.parameters for a in e.aliases}
    assert set(rows)==set(dict(m.named_parameters(remove_duplicate=False)))
    assert rows['decoder.profiles.shared.backbone.feature.weight'] is rows['native.response.backbone.feature.weight']
    assert rows['reconstruction.latent_projection.weight'].axes[0]==p.Axis('width','hidden')
    assert inventory(m,latent_axis=('y','x'))!=result
    if geometry:
        assert rows['geometry_fusion.image.backbone.incidence.weight'] is rows['native.response.backbone.incidence.weight']
        with pytest.raises(ValueError):inventory(m,image_identifier_axis=('observation','context'))
    assert torch.equal(rng,torch.get_rng_state())
    if gpu is not None:assert torch.equal(gpu,torch.cuda.get_rng_state())
    assert modes=={n:c.training for n,c in m.named_modules()}
    for n,t in m.state_dict().items():
        if torch.is_tensor(t):assert torch.equal(t,state[n])
        else:assert t==state[n]
    for n,t in m.named_parameters():assert torch.equal(t.grad,grads[n])


@pytest.mark.parametrize('drift',['missing','extra-owner','stale','extra-module','parameter','buffer',
    'decoder','config','axis','tie','subclass','architecture','ownership','geometry-args'])
def test_strict_refusals(tmp_path,drift):
    m,_=build(tmp_path,'cpu');o=inherited(m);kw={}
    if drift=='missing':o.pop('goa')
    elif drift=='extra-owner':o['reconstruction']=o['goa']
    elif drift=='stale':m.residue.head.output.weight.requires_grad_(False)
    elif drift=='extra-module':m.extra=torch.nn.Identity()
    elif drift=='parameter':m.extra=torch.nn.Parameter(torch.ones(1))
    elif drift=='buffer':m.register_buffer('extra',torch.ones(1))
    elif drift=='decoder':object.__setattr__(m.reconstruction,'decoder',copy.deepcopy(m.decoder))
    elif drift=='config':kw['configuration']={}
    elif drift=='axis':kw['latent_axis']=('x','x')
    elif drift=='tie':m.reconstruction.latent_projection.weight=m.decoder.embedding.weight
    elif drift=='subclass':
        class Other(type(m)):pass
        m.__class__=Other
    elif drift=='architecture':m.configuration['architecture']='other'
    elif drift=='ownership':m.configuration['reconstruction_ownership']='copied'
    elif drift=='geometry-args':kw['image_identifier_axis']=('context','observation')
    with pytest.raises(ValueError):inventory(m,o,**kw)


@pytest.mark.parametrize('drift',['missing-object-axis','missing-image-axis','shared-image','schedule','pairing','geometry-config'])
def test_geometry_refusals(tmp_path,drift):
    m,_=geometry_build(tmp_path,'cpu');kw={}
    if drift=='missing-object-axis':kw['geometry_identifier_axis']=None
    elif drift=='missing-image-axis':kw['image_identifier_axis']=None
    elif drift=='shared-image':m.geometry_fusion.image=copy.deepcopy(m.native.image)
    elif drift=='schedule':m.configuration['geometry_schedule']='new'
    elif drift=='pairing':m.configuration['geometry_pairing']='invented'
    elif drift=='geometry-config':m.configuration['geometry']={}
    with pytest.raises(ValueError):inventory(m,**kw)
