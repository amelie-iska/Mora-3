"""Exact cooperative semantic owner composition; no admission or scaling."""
import copy
import pytest
import torch
from mora3.model import parameterization as p
from mora3.model.color_cooperative import ColorCooperativeModel
from test_lincs_cooperative import build, cases
from test_reconstruction_owner_parameterization import inherited
from test_color_supervised_task import source
from mora3.model.color_supervised_task import ColorSemanticTask


def make(tmp_path,cases,device='cpu',policy='native-max512-v1'):
    return ColorCooperativeModel(build(tmp_path,cases,device).model,
                                 color_id_width=2,input_policy=policy)


def inventory(m,owners=None,**kw):
    assert hasattr(p,'color_cooperative_parameter_ledger'),'missing cooperative semantic inventory'
    args=dict(owners=inherited(m.parent) if owners is None else owners,
        identifier_axis=('context','observation'),latent_axis=('x','y'),
        configuration=m.parent.reconstruction.get_extra_state(),
        geometry_identifier_axis=tuple(f'object:{i}' for i in range(16)),
        image_identifier_axis=('context','observation'),
        channel_axis=('R','G','B'),color_identifier_axis=('context','observation'))
    args.update(kw)
    return p.color_cooperative_parameter_ledger(m,**args)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('policy',['native-max512-v1','native-max1000-v1'])
def test_count_shared_aliases_and_numerical_nonmutation(tmp_path,cases,device,dtype,policy):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(36401)
    m=make(tmp_path,cases,device,policy).to(dtype=dtype).eval()
    # Explicit fresh-fixture precision declaration, never checkpoint migration.
    m.parent.configuration['reconstruction']=m.parent.reconstruction.get_extra_state()
    m.configuration['parent']=copy.deepcopy(m.parent.configuration)
    task=ColorSemanticTask(source(),source().binding_sha256)
    params=tuple(m.parameters());before=copy.deepcopy(m.state_dict())
    modes=[c.training for c in m.modules()]
    loss=m.color_objective(task)['loss']
    grads=torch.autograd.grad(loss,params,allow_unused=True)
    rng=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone() if device=='cuda' else None
    result=inventory(m)
    # Independent object-set addition and native color equations: shared feature,
    # role, graph token, attention and final norm are already parent owned.
    parent_ids={id(t) for t in m.parent.parameters()}
    new=sum(t.numel() for t in m.color.parameters() if id(t) not in parent_ids)
    w=m.color.color.width
    expected=9*(3*w+51*w*w)+20*w+2*w+1+6*w+517*w+w*(w+1)+w+w+1
    assert new==expected
    assert result.total_numel==sum(t.numel() for t in m.parent.parameters())+new
    assert result.total_numel==sum(t.numel() for t in params)
    rows={a:e for e in result.parameters for a in e.aliases}
    assert set(rows)==set(dict(m.named_parameters(remove_duplicate=False)))
    assert rows['color.color.backbone.feature.weight'] is rows['parent.annotation.backbone.feature.weight']
    assert rows['color.context.graph.feature.weight'] is rows['parent.native.response.backbone.feature.weight']
    assert rows['color.color.backbone.incidence.weight'] is not rows['parent.annotation.backbone.incidence.weight']
    assert torch.equal(rng,torch.get_rng_state())
    if gpu is not None:assert torch.equal(gpu,torch.cuda.get_rng_state())
    after=m.color_objective(task)['loss'];torch.testing.assert_close(loss,after,rtol=0,atol=0)
    g2=torch.autograd.grad(after,params,allow_unused=True)
    rtol,atol=(0,0) if device=='cpu' else ((1e-5,1e-8) if dtype==torch.float32 else (1e-10,1e-12))
    for a,b in zip(grads,g2):
        assert (a is None)==(b is None)
        if a is not None:torch.testing.assert_close(a,b,rtol=rtol,atol=atol)
    for k,v in m.state_dict().items():
        if torch.is_tensor(v):assert torch.equal(v,before[k])
        else:assert v==before[k]
    assert modes==[c.training for c in m.modules()]
    assert all(t.grad is None for t in params)
    with pytest.raises(ValueError,match='custom'):p.width_family(result,result,result)


@pytest.mark.parametrize('drift',['architecture','version','pairing','admission','parent-config',
    'color-config','native-color','extra-config','extra-module','parameter','buffer','subclass',
    'feature','role','graph_token','layers','final_norm','parent-source','parent-pairing',
    'owners-missing','owners-extra','owners-stale','color-axis','rgb','parent-axis','tie'])
def test_configuration_ownership_and_axis_refusals(tmp_path,cases,drift):
    m=make(tmp_path,cases);o=inherited(m.parent);kw={}
    if drift=='architecture':m.configuration['architecture']='other'
    elif drift=='version':m.configuration['version']=2
    elif drift=='pairing':m.configuration['pairing']='invented'
    elif drift=='admission':m.configuration['training_admission']=True
    elif drift=='parent-config':m.configuration['parent']={}
    elif drift=='color-config':m.configuration['color']={}
    elif drift=='native-color':m.configuration['native_color']={}
    elif drift=='extra-config':m.configuration['extra']='unknown'
    elif drift=='extra-module':m.extra=torch.nn.Identity()
    elif drift=='parameter':m.extra=torch.nn.Parameter(torch.ones(1))
    elif drift=='buffer':m.register_buffer('extra',torch.ones(1))
    elif drift=='subclass':
        class Other(type(m)):pass
        m.__class__=Other
    elif drift in ('feature','role','graph_token','layers','final_norm'):
        value=getattr(m.color.color.backbone,drift)
        setattr(m.color.color.backbone,drift,torch.nn.Parameter(value.detach().clone()) if isinstance(value,torch.nn.Parameter) else copy.deepcopy(value))
    elif drift=='parent-source':m.parent.lincs.source_axes['L1000']['extra']='drift'
    elif drift=='parent-pairing':m.parent.configuration['lincs_pairing']='invented'
    elif drift=='owners-missing':o.pop('goa')
    elif drift=='owners-extra':o['color']=o['goa']
    elif drift=='owners-stale':m.parent.residue.head.output.weight.requires_grad_(False)
    elif drift=='color-axis':kw['color_identifier_axis']=('context',)
    elif drift=='rgb':kw['channel_axis']=('B','G','R')
    elif drift=='parent-axis':kw['identifier_axis']=('observation','context')
    elif drift=='tie':m.color.offset.weight=m.parent.annotation.backbone.feature.weight
    with pytest.raises(ValueError):inventory(m,o,**kw)
