"""Ragged diagnostic inventories preserve finite identity and exact aliases."""
import copy
import pytest
import torch
from mora3.model.current_frame_velocity import SplitContextualCoefficients
from mora3.model.identity_noise_vector import IDENTITY_CONTEXT_AXIS
import mora3.model.identity_noise_vector as mod
from mora3.model.parameterization import width_family


def case(owner):
    # Numerical payloads are deliberately opaque to this metadata-only API.
    return dict(coefficients=owner,current_xyz=object(),width_identity='diagnostic-identity-123')


def inventory(cases,ready):
    assert hasattr(mod,'identity_noise_parameter_ledger'),'missing ragged identity inventory'
    return mod.identity_noise_parameter_ledger(cases,ready)


@pytest.mark.parametrize('tying',['independent','owner','weight','receiver_neighbor'])
def test_complete_alias_counts_and_finite_axes(tying):
    a=SplitContextualCoefficients(IDENTITY_CONTEXT_AXIS,max_atoms=8)
    b=SplitContextualCoefficients(IDENTITY_CONTEXT_AXIS,max_atoms=8)
    expected=496
    if tying=='owner':b=a;expected=248
    if tying=='weight':b.receiver.weight=a.receiver.weight;expected=373
    if tying=='receiver_neighbor':a.neighbor.weight=a.receiver.weight;expected=373
    ledger=inventory([case(a),object(),case(b)],[True,False,True])
    assert ledger.total_numel==ledger.trainable_numel==expected
    aliases={n for p in ledger.parameters for n in p.aliases}
    assert aliases=={f'case_{i}.{name}.weight' for i in (0,2) for name in ('geometry','receiver','neighbor')}
    assert all(p.role=='custom' and all(ax.kind=='finite' for ax in p.axes) for p in ledger.parameters)
    with pytest.raises(ValueError,match='custom'):width_family(ledger,ledger,ledger)


@pytest.mark.parametrize('bad',['empty','length','bool','none_active','container','extra','missing','width','owner','axis','drift','limit'])
def test_invalid_active_input_refuses_without_forward(bad):
    a=SplitContextualCoefficients(IDENTITY_CONTEXT_AXIS,max_atoms=8)
    b=SplitContextualCoefficients(IDENTITY_CONTEXT_AXIS,max_atoms=8)
    cases=[case(a),case(b)];ready=[True,True];calls=[]
    handles=[m.register_forward_pre_hook(lambda *args:calls.append(1)) for owner in (a,b) for m in owner.modules()]
    if bad=='empty':cases=[];ready=[]
    if bad=='length':ready.pop()
    if bad=='bool':ready[1]=1
    if bad=='none_active':ready=[False,False]
    if bad=='container':cases[1]=object()
    if bad=='extra':cases[1]['target']=object()
    if bad=='missing':cases[1].pop('current_xyz')
    if bad=='width':cases[1]['width_identity']=' '
    if bad=='owner':cases[1]['coefficients']=object()
    if bad=='axis':cases[1]['coefficients']=SplitContextualCoefficients(tuple(reversed(IDENTITY_CONTEXT_AXIS)),max_atoms=8)
    if bad=='drift':b.register_buffer('unexpected',torch.ones(1))
    if bad=='limit':cases=[cases[0]]*65;ready=[True]*65
    with pytest.raises(ValueError):inventory(cases,ready)
    for h in handles:h.remove()
    assert not calls


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_retained_ragged_equations_gradients_state_rng(device,dtype):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from test_retained_identity_noise_ragged import inputs, test_equal_case_independent_equations_and_gradients
    torch.manual_seed(35601)
    selections,cases,targets=inputs(device,dtype)
    states=[copy.deepcopy(c['coefficients'].state_dict()) for c in cases]
    rng=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state_all()
    before=mod.retained_identity_noise_vector_loss(selections,cases,targets,[True,True],batch_identity=True)
    ledger=inventory(cases,[True,True])
    after=mod.retained_identity_noise_vector_loss(selections,cases,targets,[True,True],batch_identity=True)
    assert ledger.total_numel==496
    torch.testing.assert_close(before.loss,after.loss,rtol=0,atol=0)
    params=[v for c in cases for v in (c['current_xyz'],*c['coefficients'].parameters())]
    for x,y in zip(torch.autograd.grad(before.loss,params),torch.autograd.grad(after.loss,params)):
        torch.testing.assert_close(x,y,rtol=0,atol=0)
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(gpu,torch.cuda.get_rng_state_all()))
    for c,state in zip(cases,states):
        for k,v in c['coefficients'].state_dict().items():
            assert torch.equal(v,state[k]) if torch.is_tensor(v) else v==state[k]
        assert all(p.grad is None for p in c['coefficients'].parameters())
        for name in ('require_source_use','require_model_use','require_training_use'):
            with pytest.raises(ValueError):getattr(c['coefficients'],name)()
    test_equal_case_independent_equations_and_gradients(device,dtype)
