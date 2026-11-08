"""Complete native cooperative ownership; no numerical or admission changes."""
import copy
import pytest
import torch
from test_lincs_native import cases, project, api
from test_lincs_cooperative import build
from test_reconstruction_owner_parameterization import inherited
from mora3.model import parameterization as p


def inventory(m,o=None,**kw):
    assert hasattr(p,'lincs_cooperative_parameter_ledger'), 'missing cooperative LINCS inventory'
    args=dict(owners=inherited(m) if o is None else o,identifier_axis=('context','observation'),
        latent_axis=('x','y'),configuration=m.reconstruction.get_extra_state(),
        geometry_identifier_axis=tuple(f'object:{i}' for i in range(16)),
        image_identifier_axis=('context','observation'))
    args.update(kw)
    return p.lincs_cooperative_parameter_ledger(m,**args)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_complete_count_aliases_outputs_gradients_and_state(tmp_path,cases,device,dtype):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(36201)
    trainer=build(tmp_path,cases,device)
    m=trainer.model.to(dtype=dtype).eval();m.annotation.train()
    # Fresh synthetic FP32 fixture, not historical checkpoint conversion. The
    # reconstruction precision declaration is explicit before the snapshot.
    m.configuration['reconstruction']=m.reconstruction.get_extra_state()
    forward=tuple(project(api(),c) for c in cases.values() if project(api(),c).direction=='molecule_to_profile')
    reverse=tuple(project(api(),c) for c in cases.values() if project(api(),c).direction=='profile_to_molecule')
    def values():return (*m.lincs(forward),m.lincs.condition(reverse))
    before=copy.deepcopy(m.state_dict());modes={n:c.training for n,c in m.named_modules()}
    params=tuple(m.parameters());out=values()
    grad=torch.autograd.grad(sum(x.square().sum() for x in out),params,allow_unused=True)
    rng=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone() if device=='cuda' else None
    result=inventory(m)
    # Count inherited unique objects independently; new three matrices/feature,
    # two input biases and one output bias per modality add no shared-owner copy.
    parent={id(t):t for n,c in m.named_children() if n!='lincs' for t in c.parameters()}
    w=m.annotation.payload.embedding.embedding_dim;n=1783+978
    assert result.total_numel==sum(t.numel() for t in parent.values())+3*n*w+2*w+n
    assert result.total_numel==sum(t.numel() for t in params)
    rows={a:e for e in result.parameters for a in e.aliases}
    assert set(rows)==set(dict(m.named_parameters(remove_duplicate=False)))
    assert rows['lincs.shared.payload.embedding.weight'] is rows['annotation.payload.embedding.weight']
    assert rows['lincs.shared.backbone.feature.weight'] is rows['native.response.backbone.feature.weight']
    assert rows['lincs.shared.physical.query.weight'] is rows['annotation.physical.query.weight']
    assert rows['lincs.output.L1000.weight'].axes[0]!=rows['profile.output.L1000.weight'].axes[0]
    assert torch.equal(rng,torch.get_rng_state())
    if gpu is not None:assert torch.equal(gpu,torch.cuda.get_rng_state())
    assert modes=={n:c.training for n,c in m.named_modules()}
    after=values();g2=torch.autograd.grad(sum(x.square().sum() for x in after),params,allow_unused=True)
    for a,b in zip(out,after):torch.testing.assert_close(a,b,rtol=0,atol=0)
    for a,b in zip(grad,g2):
        assert (a is None)==(b is None)
        if a is not None:torch.testing.assert_close(a,b,rtol=0,atol=0)
    for key,t in m.state_dict().items():
        if torch.is_tensor(t):assert torch.equal(t,before[key])
        else:assert t==before[key]
    assert all(t.grad is None for t in params)
    assert trainer.steps==0 and not trainer.optimizer.state
    with pytest.raises(ValueError,match='custom'):p.width_family(result,result,result)


@pytest.mark.parametrize('drift',['missing','extra-owner','stale','extra-module','parameter','buffer',
    'architecture','alias','lincs-config','pairing','reconstruction','geometry-config','geometry-image',
    'geometry-schedule','geometry-pairing','decoder','source','axis','shape','subclass','tie','identifier'])
def test_root_and_native_ownership_refusals(tmp_path,cases,drift):
    m=build(tmp_path,cases).model;o=inherited(m);kw={}
    if drift=='missing':o.pop('annotation')
    elif drift=='extra-owner':o['lincs']=o['annotation']
    elif drift=='stale':m.residue.head.output.weight.requires_grad_(False)
    elif drift=='extra-module':m.extra=torch.nn.Identity()
    elif drift=='parameter':m.extra=torch.nn.Parameter(torch.ones(1))
    elif drift=='buffer':m.register_buffer('extra',torch.ones(1))
    elif drift=='architecture':m.configuration['architecture']='other'
    elif drift=='alias':m.lincs.shared=copy.deepcopy(m.annotation)
    elif drift=='lincs-config':m.configuration['lincs']={}
    elif drift=='pairing':m.configuration['lincs_pairing']='invented'
    elif drift=='reconstruction':m.configuration['reconstruction_ownership']='copied'
    elif drift=='geometry-config':m.configuration['geometry']={}
    elif drift=='geometry-image':m.geometry_fusion.image=copy.deepcopy(m.native.image)
    elif drift=='geometry-schedule':m.configuration['geometry_schedule']='other'
    elif drift=='geometry-pairing':m.configuration['geometry_pairing']='invented'
    elif drift=='decoder':object.__setattr__(m.reconstruction,'decoder',copy.deepcopy(m.decoder))
    elif drift=='source':m.lincs.source_axes['L1000']['extra']='drift'
    elif drift=='axis':m.lincs.axes['L1000']=m.lincs.axes['L1000'][::-1]
    elif drift=='shape':m.lincs.output['L1000'].weight=torch.nn.Parameter(torch.zeros(978,9))
    elif drift=='subclass':
        class Other(type(m)):pass
        m.__class__=Other
    elif drift=='tie':m.lincs.output['L1000'].weight=m.lincs.profile_input['L1000'].weight
    elif drift=='identifier':kw['identifier_axis']=('observation','context')
    assert hasattr(p,'lincs_cooperative_parameter_ledger'), 'missing cooperative LINCS inventory'
    with pytest.raises(ValueError):inventory(m,o,**kw)


def test_compatible_native_input_mask_tie_counts_once(tmp_path,cases):
    m=build(tmp_path,cases).model
    initial=inventory(m)
    m.lincs.profile_mask['L1000'].weight=m.lincs.profile_input['L1000'].weight
    result=inventory(m)
    assert result.total_numel==initial.total_numel-978*m.annotation.payload.embedding.embedding_dim
    row=next(e for e in result.parameters if 'lincs.profile_mask.L1000.weight' in e.aliases)
    assert row.aliases==('lincs.profile_input.L1000.weight','lincs.profile_mask.L1000.weight')
