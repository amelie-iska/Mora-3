"""Current input metadata cannot authorize source or target geometry."""
from dataclasses import replace
import importlib
import pytest
import torch
from mora3.data import spatial_design_constraints as q
from mora3.data import geometry_frame_policy as p
from mora3.model.current_frame_velocity import CurrentFrameVelocityReadout


def fixture():
    req=q.DesignRequest(q.Direction.DE_NOVO,
        (q.InputSpec('ts',q.InputKind.TS_GEOMETRY,q.Role.USER_SUPPLIED),),
        q.Budget(1,8,8,32,1,1048576,1))
    src=p.SourceDeclaration('synthetic:current','literal','reader','a'*64,'b'*64,
        None,None,p.Origin.SYNTHETIC_LITERAL,p.Purpose.CONDITION_INPUT)
    axis=tuple(q.AtomKey('ts',i,'C',f'C{i}') for i in range(3))
    return dict(request=req,compiled=q.compile_design_request(req),
        current=p.CurrentFrame('ts','current','angstrom',src),role=q.Role.USER_SUPPLIED,
        native_axis=axis,native_axis_sha256=p.axis_digest(axis),
        xyz_available=(True,True,True),free_atoms=(True,False,True))


def bind(c):
    return importlib.import_module('mora3.data.current_frame_request_binding').bind_current_frame_request(**c)


def test_complete_binding_retains_identity_and_refuses_authority():
    c=fixture(); b=bind(c)
    assert b.native_axis==c['native_axis'] and b.free_atoms==(True,False,True)
    assert b.request_sha256==c['compiled'].request_sha256
    assert b.synthetic_diagnostic and not b.native_model_authorized
    for name in ('require_source_use','require_model_use','require_training_use'):
        with pytest.raises(ValueError):getattr(b,name)()


@pytest.mark.parametrize('change',[
    'reorder','duplicate','namespace','mask_partial','mask_components','mask_int',
    'mask_short','free_int','free_short','empty','compiled','purpose','role','key','budget'])
def test_invalid_binding_refused(change):
    c=fixture()
    if change=='reorder':c['native_axis']=c['native_axis'][::-1]
    if change=='duplicate':c['native_axis']=(c['native_axis'][0],)*3
    if change=='namespace':c['native_axis']=tuple(replace(a,entity='gold') for a in c['native_axis']);c['native_axis_sha256']=p.axis_digest(c['native_axis'])
    if change=='mask_partial':c['xyz_available']=(True,False,True)
    if change=='mask_components':c['xyz_available']=((True,True,True),)*3
    if change=='mask_int':c['xyz_available']=(True,1,True)
    if change=='mask_short':c['xyz_available']=(True,)
    if change=='free_int':c['free_atoms']=(True,0,True)
    if change=='free_short':c['free_atoms']=(True,)
    if change=='empty':c['native_axis']=();c['native_axis_sha256']=p.axis_digest(())
    if change=='compiled':c['compiled']=replace(c['compiled'],predictor_input_keys=('gold',))
    if change=='purpose':c['current']=replace(c['current'],source=replace(c['current'].source,purpose=p.Purpose.WITHHELD_TARGET))
    if change=='role':c['role']=q.Role.DESIRED
    if change=='key':c['current']=replace(c['current'],input_key='gold')
    if change=='budget':c['request']=replace(c['request'],budget=replace(c['request'].budget,max_atoms=2));c['compiled']=q.compile_design_request(c['request'])
    with pytest.raises(ValueError):bind(c)


def test_unqualified_source_metadata_cannot_drive_diagnostic():
    c=fixture();c['current']=replace(c['current'],source=replace(c['current'].source,
        origin=p.Origin.UNQUALIFIED_SOURCE,source_id='external:unreviewed'))
    b=bind(c);assert not b.synthetic_diagnostic
    with pytest.raises(ValueError):b.require_synthetic_diagnostic()


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_bound_diagnostic_matches_vector_equation_and_gradients(device,dtype):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(32501)
    c=fixture();b=bind(c)
    m=CurrentFrameVelocityReadout(('scalar',),max_atoms=8).to(device=device,dtype=dtype)
    x=torch.tensor([[0.,1.,2.],[2.,0.,1.],[-1.,3.,0.]],device=device,dtype=dtype,requires_grad=True)
    f=torch.arange(9,device=device,dtype=dtype).reshape(3,3,1).div(10).requires_grad_()
    v=m.forward_bound_diagnostic(x,f,binding=b,atom_axis=c['native_axis'],feature_axis=('scalar',))
    w=m.coefficients.weight[0,0]
    ref=torch.stack([sum(w*f[i,j,0]*(x[j]-x[i]) for j in range(3))/3
        if c['free_atoms'][i] else x[i]*0 for i in range(3)])
    torch.testing.assert_close(v,ref)
    for a,r in zip(torch.autograd.grad(v.square().sum(),(x,f,m.coefficients.weight),retain_graph=True),
                   torch.autograd.grad(ref.square().sum(),(x,f,m.coefficients.weight))):
        torch.testing.assert_close(a,r)


@pytest.mark.parametrize('bad',['axis','features','source','forged_mask'])
def test_refusals_precede_learned_work(bad):
    c=fixture();b=bind(c);axis=c['native_axis'];features=('scalar',)
    if bad=='axis':axis=axis[::-1]
    if bad=='features':features=('unknown',)
    if bad=='source':
        c['current']=replace(c['current'],source=replace(c['current'].source,origin=p.Origin.UNQUALIFIED_SOURCE));b=bind(c)
    if bad=='forged_mask':object.__setattr__(b,'xyz_available',(True,False,True))
    m=CurrentFrameVelocityReadout(('scalar',),max_atoms=8)
    called=[];h=m.coefficients.register_forward_pre_hook(lambda *args:called.append(1))
    with pytest.raises(ValueError):
        m.forward_bound_diagnostic(torch.zeros(3,3),torch.ones(3,3,1),binding=b,atom_axis=axis,feature_axis=features)
    h.remove();assert called==[]
