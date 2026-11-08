"""Layered sampler analytic joint-density and gradient oracles."""
import sys
import math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import pytest
import torch
from mora3.reasoning import continuous_gfn as core


def api():
    from mora3.reasoning.gfn_sampling import LayeredGaussianPolicy, sample_paths, score_paths
    return LayeredGaussianPolicy,sample_paths,score_paths


def constant(device='cpu'):
    cls,_,_=api()
    p=cls(2,3,hidden=8).to(device=device,dtype=torch.float64)
    with torch.no_grad():
        for v in p.parameters():v.zero_()
        p.forward_head.bias[:2]=.7
        p.forward_head.bias[2:]=math.log(.8)
    return p


def test_analytic_gaussian_terminal_and_geometric_lengths():
    _,sample,score=api();p=constant()
    paths=sample(p,30000,generator=torch.Generator().manual_seed(19))
    terminal=paths['states'][torch.arange(30000),paths['lengths']-1]
    torch.testing.assert_close(terminal.mean(0),torch.full((2,),.7,dtype=terminal.dtype),atol=.02,rtol=0)
    torch.testing.assert_close(terminal.var(0),torch.full((2,),.64,dtype=terminal.dtype),atol=.025,rtol=0)
    freq=torch.bincount(paths['lengths'],minlength=4)[1:]/30000
    torch.testing.assert_close(freq,torch.tensor([.5,.25,.25]),atol=.012,rtol=0)
    s=score(p,paths)
    mask=paths['mask'];x=paths['states'];prev=torch.cat([torch.zeros_like(x[:,:1]),x[:,:-1]],1)
    normal=torch.distributions.Normal(.7,.8).log_prob(x).sum(-1)
    expected_pf=normal.clone();expected_pf[:,1:]-=torch.log(torch.tensor(2.))
    expected_pb=torch.distributions.Normal(0.,1.).log_prob(prev).sum(-1);expected_pb[:,0]=0
    torch.testing.assert_close(s['log_forward'][mask],expected_pf[mask])
    torch.testing.assert_close(s['log_backward'][mask],expected_pb[mask])
    torch.testing.assert_close(s['log_stop'],torch.where(paths['lengths']==3,0.,-torch.log(torch.tensor(2.))).double())
    assert not x.requires_grad
    assert (x[~mask]==0).all()


def test_seed_forced_horizon_and_single_layer():
    cls,sample,score=api();p=constant()
    a=sample(p,12,generator=torch.Generator().manual_seed(3),optional_stop=False)
    b=sample(p,12,generator=torch.Generator().manual_seed(3),optional_stop=False)
    assert torch.equal(a['states'],b['states']) and (a['lengths']==3).all()
    assert (score(p,a)['log_stop']==0).all()
    one=cls(2,1,8).double();paths=sample(one,5)
    assert (paths['lengths']==1).all() and (score(one,paths)['log_stop']==0).all()
    assert (score(one,paths)['log_backward']==0).all()


def test_contracts():
    cls,sample,score=api()
    for args in [(0,3,8),(2,0,8),(2,3,0),(True,3,8)]:
        with pytest.raises(ValueError):cls(*args)
    p=constant()
    for n in [0,-1,True]:
        with pytest.raises(ValueError):sample(p,n)
    paths=sample(p,4)
    bad=dict(paths,mask=~paths['mask'])
    with pytest.raises(ValueError):score(p,bad)
    bad=dict(paths,states=paths['states']*float('nan'))
    with pytest.raises(ValueError):score(p,bad)
    bad=dict(paths,lengths=paths['lengths']+1)
    with pytest.raises(ValueError):score(p,bad)
    with pytest.raises(ValueError):sample(p,2,optional_stop='yes')


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_score_balance_finite_gradients_and_gpu_parity(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    _,sample,score=api();torch.manual_seed(19)
    cpu=constant();paths=sample(cpu,32,generator=torch.Generator().manual_seed(19))
    def run(p,paths):
        s=score(p,paths);terminal=paths['states'][torch.arange(32,device=paths['states'].device),paths['lengths']-1]
        reward=core.diagonal_gaussian_log_density(terminal,torch.zeros_like(terminal),torch.zeros_like(terminal))-torch.log(terminal.new_tensor(3.))
        loss=core.trajectory_balance(p.log_z,**s,log_reward=reward,mask=paths['mask'])['loss']
        loss.backward()
        return loss.detach().cpu(),[v.grad.detach().cpu() for v in p.parameters()]
    baseline,grads=run(cpu,paths)
    other=constant(device);local={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in paths.items()}
    loss,g=run(other,local)
    torch.testing.assert_close(loss,baseline)
    for a,b in zip(g,grads):
        assert torch.isfinite(a).all();torch.testing.assert_close(a,b,atol=1e-9,rtol=1e-9)


def test_exact_joint_target_has_zero_tb_residual():
    _,sample,score=api();p=constant()
    with torch.no_grad():
        p.backward_head.bias.copy_(p.forward_head.bias)
    paths=sample(p,1000,generator=torch.Generator().manual_seed(7))
    terminal=paths['states'][torch.arange(1000),paths['lengths']-1]
    weights=terminal.new_tensor([.5,.25,.25])
    reward=torch.distributions.Normal(terminal.new_tensor(.7),terminal.new_tensor(.8)).log_prob(terminal).sum(-1)+weights[paths['lengths']-1].log()
    result=core.trajectory_balance(p.log_z,**score(p,paths),log_reward=reward,mask=paths['mask'])
    torch.testing.assert_close(result['residual'],torch.zeros_like(reward),atol=2e-15,rtol=0)


def test_time_conditioning_and_detach_contract():
    cls,sample,score=api();torch.manual_seed(29);p=cls(2,3,8).double()
    x=torch.zeros(4,2,dtype=torch.float64)
    assert not torch.equal(p.parameters_at(x,0)[0],p.parameters_at(x,1)[0])
    for layer in [-1,4,True]:
        with pytest.raises(ValueError):p.parameters_at(x,layer)
    paths=sample(p,8)
    with pytest.raises(ValueError):score(p,dict(paths,states=paths['states'].requires_grad_()))
