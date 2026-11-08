"""Source-bound on-policy updates must use declared fixed latent rewards."""
import copy
import pytest
import torch
from test_molecule_profile_model import model, view
from mora3.model.profile_gfn import ProfileGFNPolicy, conditional_trajectory_balance
from mora3.reasoning.gfn_sampling import sample_paths, score_paths
import mora3.model.profile_gfn_training as training

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_real_update_matches_independent_tb_adam_reference(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(11801)
    policy=ProfileGFNPolicy(model(device).double(),dimension=2,horizon=3,hidden=5,source_binding={'pool':'fixture'})
    views=(view('profile_to_molecule'),view('profile_to_molecule','L1000'))
    reference=copy.deepcopy(policy)
    reward=training.LatentGaussianReward(scale=1.5)
    trainer=training.ProfileGFNTrainer(policy,views,reward=reward,lr=.001,seed=11801)
    context=reference.context(views)
    paths=sample_paths(reference,len(views),context=context,generator=torch.Generator(device=device).manual_seed(11801))
    terminal=paths['states'][torch.arange(len(views),device=device),paths['lengths']-1]
    log_reward=torch.distributions.Normal(torch.zeros_like(terminal),torch.full_like(terminal,1.5)).log_prob(terminal).sum(1)-torch.tensor(3.,device=device,dtype=torch.float64).log()
    loss=conditional_trajectory_balance(reference.log_partition(context),**score_paths(reference,paths,context=context),log_reward=log_reward,mask=paths['mask'])['loss']
    optimizer=torch.optim.Adam(reference.parameters(),lr=.001)
    optimizer.zero_grad();loss.backward();optimizer.step()
    result=trainer.step()
    assert result['step']==1
    assert result['loss']==pytest.approx(float(loss.detach()),abs=1e-11)
    for actual,expected in zip(policy.parameters(),reference.parameters()):torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
    config=trainer.binding();config['reward']['scale']=99
    assert trainer.binding()['reward']['scale']==1.5
    assert trainer.binding()['policy']['source_binding']=={'pool':'fixture'}

@pytest.mark.parametrize('scale',[0,-1,float('nan'),float('inf'),True])
def test_bad_reward_config_rejected(scale):
    with pytest.raises(ValueError):training.LatentGaussianReward(scale=scale)

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_reward_detached_normalized_and_layers_explicit(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    reward=training.LatentGaussianReward(scale=2.)
    terminal=torch.tensor([[0.,1.],[2.,3.]],device=device,dtype=torch.float64,requires_grad=True)
    actual=reward(terminal,horizon=4)
    expected=torch.distributions.Normal(0.,2.).log_prob(terminal).sum(1)-torch.tensor(4.).log().double()
    torch.testing.assert_close(actual,expected)
    assert not actual.requires_grad
    with pytest.raises(ValueError):reward(terminal,horizon=0)


def test_missing_reward_and_empty_batch_rejected():
    p=ProfileGFNPolicy(model(),dimension=2,horizon=2,source_binding={'pool':'fixture'})
    with pytest.raises(ValueError):training.ProfileGFNTrainer(p,(),reward=training.LatentGaussianReward())
    with pytest.raises(ValueError):training.ProfileGFNTrainer(p,(view('profile_to_molecule'),),reward=None)


def test_optimizer_configuration_drift_rejected_before_sampling():
    p=ProfileGFNPolicy(model(),dimension=2,horizon=2,source_binding={'pool':'fixture'})
    trainer=training.ProfileGFNTrainer(p,(view('profile_to_molecule'),),reward=training.LatentGaussianReward())
    trainer.optimizer.param_groups[0]['lr']=.5
    before=trainer.generator.get_state().clone()
    with pytest.raises(ValueError,match='binding'):trainer.step()
    assert torch.equal(before,trainer.generator.get_state())
    assert trainer.step_number==0


def recovery_trainer(device='cpu'):
    p=ProfileGFNPolicy(model(device).double(),dimension=2,horizon=3,hidden=5,source_binding={'pool':'fixture'})
    return training.ProfileGFNTrainer(p,(view('profile_to_molecule'),view('profile_to_molecule','CellPainting')),reward=training.LatentGaussianReward(1.5),seed=11901)


def same(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_checkpoint_replays_model_adam_global_and_local_rng_exactly(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    t=recovery_trainer(device);t.step()
    state=t.state_dict();expected=t.step();end=t.state_dict()
    fresh=recovery_trainer(device);fresh.load_state_dict(state)
    assert fresh.step()==expected
    assert same(fresh.state_dict(),end)
    state['model']['normalizer.weight'].zero_()
    assert not torch.equal(state['model']['normalizer.weight'],fresh.policy.normalizer.weight)


@pytest.mark.parametrize('damage',['binding','steps','model','moment','missing_slot','generator','cpu_rng','mode','optimizer','runtime'])
def test_malformed_recovery_is_transactional(damage):
    t=recovery_trainer();t.step();state=t.state_dict();before=t.state_dict()
    if damage=='binding':state['binding']['reward']['scale']=9
    elif damage=='steps':state['steps']=True
    elif damage=='model':state['model']['normalizer.weight'].fill_(float('nan'))
    elif damage=='moment':next(iter(state['optimizer']['state'].values()))['exp_avg_sq'].fill_(-1)
    elif damage=='missing_slot':state['optimizer']['state'].pop(next(iter(state['optimizer']['state'])))
    elif damage=='generator':state['generator_rng']=torch.zeros(3,dtype=torch.uint8)
    elif damage=='cpu_rng':state['cpu_rng']=torch.zeros(3,dtype=torch.uint8)
    elif damage=='mode':state['modes'].pop('')
    elif damage=='optimizer':state['optimizer']['param_groups'][0]['betas']=(.5,.5)
    elif damage=='runtime':state['runtime']['threads']=-1
    with pytest.raises((ValueError,RuntimeError,KeyError)):t.load_state_dict(state)
    assert same(before,t.state_dict())


def test_model_precision_rejected_before_any_live_mutation():
    t=recovery_trainer();t.step();before=t.state_dict();bad=t.state_dict()
    bad['model']['normalizer.weight']=bad['model']['normalizer.weight'].float()
    with pytest.raises(ValueError):t.load_state_dict(bad)
    assert same(before,t.state_dict())


def test_runtime_drift_rejected_before_update_or_snapshot():
    t=recovery_trainer();previous=torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32=not previous
        with pytest.raises(ValueError):t.state_dict()
        with pytest.raises(ValueError):t.step()
    finally:torch.backends.cudnn.allow_tf32=previous


def test_adam_betas_drift_rejected_before_sampling():
    t=recovery_trainer();before=t.generator.get_state().clone()
    t.optimizer.param_groups[0]['betas']=(.5,.5)
    with pytest.raises(ValueError):t.step()
    assert torch.equal(before,t.generator.get_state())


def test_active_gradient_cannot_restore_without_its_adam_history():
    t=recovery_trainer();t.step();bad=t.state_dict();before=t.state_dict()
    pid=next(iter(bad['optimizer']['state']))
    bad['optimizer']['state'].pop(pid);bad['initialized_slots'].remove(pid)
    with pytest.raises(ValueError):t.load_state_dict(bad)
    assert same(before,t.state_dict())


def test_replaced_reward_rejected_before_restore():
    t=recovery_trainer();t.step();state=t.state_dict();before=t.policy.normalizer.weight.clone()
    t.reward=training.LatentGaussianReward(2.)
    with pytest.raises(ValueError):t.load_state_dict(state)
    assert torch.equal(before,t.policy.normalizer.weight)
