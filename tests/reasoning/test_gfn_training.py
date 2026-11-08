"""On-policy learning, analytic target moments and complete continuation state."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import pytest
import torch


def trainer(device='cpu', horizon=1, seed=19):
    from mora3.reasoning.gfn_training import OnPolicyTrainer
    from mora3.reasoning.gfn_sampling import LayeredGaussianPolicy
    torch.manual_seed(seed)
    policy=LayeredGaussianPolicy(1,horizon,8).to(device=device,dtype=torch.float64)
    return OnPolicyTrainer(policy,torch.tensor([[1.2]],device=device,dtype=torch.float64),batch_size=128,lr=.01,seed=seed)


def test_gaussian_learning_and_terminal_moments():
    t=trainer();initial=t.evaluate(5000,seed=31)
    for _ in range(400):t.step()
    result=t.evaluate(12000,seed=31)
    assert result['tb_loss'] < initial['tb_loss']*.02
    assert abs(result['mean'][0]-1.2)<.04
    assert abs(result['variance'][0]-1)<.06
    assert abs(float(t.policy.log_z.detach()))<.05


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_exact_resume_and_exclusive_checkpoint(tmp_path,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    a=trainer(device,3)
    for _ in range(4):a.step()
    path=tmp_path/'mid.pt';a.save(path)
    with pytest.raises(FileExistsError):a.save(path)
    b=trainer(device,3,seed=3);b.load(path)
    assert a.step_number==b.step_number==4
    for _ in range(3):assert a.step()==b.step()
    for x,y in zip(a.policy.parameters(),b.policy.parameters()):assert torch.equal(x,y)
    assert torch.equal(a.generator.get_state(),b.generator.get_state())
    # Evaluation must leave optimizer's rollout RNG untouched.
    state=b.generator.get_state().clone();b.evaluate(100,seed=42)
    assert torch.equal(state,b.generator.get_state())
    wrong=trainer(device,3);wrong.centers.add_(1)
    with pytest.raises(ValueError,match='configuration'):wrong.load(path)


def test_mixture_reward_independent_oracle_and_contracts():
    from mora3.reasoning.gfn_training import mixture_log_reward
    x=torch.tensor([[0.],[2.]],dtype=torch.float64)
    centers=torch.tensor([[-1.],[1.]],dtype=torch.float64)
    got=mixture_log_reward(x,centers,3)
    expected=torch.distributions.Normal(centers.T,1).log_prob(x).logsumexp(1)-torch.tensor(6.,dtype=x.dtype).log()
    torch.testing.assert_close(got,expected)
    with pytest.raises(ValueError):mixture_log_reward(x,centers,0)
    t=trainer();t.centers.fill_(float('nan'))
    with pytest.raises(ValueError):t.step()
