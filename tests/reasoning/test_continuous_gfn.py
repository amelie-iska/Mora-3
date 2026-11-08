"""Density normalization, mixed stop mass and TB sign/masking/gradient oracles."""
import importlib
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import pytest
import torch


def api():
    return importlib.import_module('mora3.reasoning.continuous_gfn')


def test_gaussian_equation_normalization_and_gradients():
    x=torch.tensor([[.2,-.7],[1.,2.]],dtype=torch.float64,requires_grad=True)
    mu=torch.tensor([[.1,.3],[-.2,.8]],dtype=torch.float64,requires_grad=True)
    ls=torch.tensor([[.2,-.3],[.4,.1]],dtype=torch.float64,requires_grad=True)
    actual=api().diagonal_gaussian_log_density(x,mu,ls)
    oracle=torch.stack([sum(-.5*((x[b,k]-mu[b,k])/ls[b,k].exp())**2-ls[b,k]-.5*math.log(2*math.pi) for k in range(2)) for b in range(2)])
    torch.testing.assert_close(actual,oracle)
    for a,b in zip(torch.autograd.grad(actual.sum(),(x,mu,ls),retain_graph=True),torch.autograd.grad(oracle.sum(),(x,mu,ls))):torch.testing.assert_close(a,b)
    grid=torch.linspace(-9,9,20001,dtype=torch.float64)[:,None]
    density=api().diagonal_gaussian_log_density(grid,torch.zeros_like(grid),torch.zeros_like(grid)).exp()
    assert torch.trapezoid(density,grid[:,0]).item()==pytest.approx(1,abs=1e-10)
    assert torch.autograd.gradcheck(api().diagonal_gaussian_log_density,(x,mu,ls))


def test_stop_and_continue_normalize_including_extreme_logits():
    logits=torch.tensor([-1000.,-.7,0.,.9,1000.],dtype=torch.float64,requires_grad=True)
    stop,cont=api().stop_log_probabilities(logits)
    torch.testing.assert_close(stop.exp()+cont.exp(),torch.ones_like(logits))
    assert torch.isfinite(stop).all() and torch.isfinite(cont).all()
    (stop.sum()+cont.sum()).backward()
    torch.testing.assert_close(logits.grad,1-2*logits.detach().sigmoid())


def test_tb_ragged_scalar_oracle_gradients_and_padding_exclusion():
    pf=torch.tensor([[-.2,-1.,999.],[-.4,-.7,-.8]],dtype=torch.float64,requires_grad=True)
    pb=torch.tensor([[0.,-.3,-999.],[0.,-.6,-.9]],dtype=torch.float64,requires_grad=True)
    mask=torch.tensor([[True,True,False],[True,True,True]])
    stop=torch.tensor([-.5,-.2],dtype=torch.float64,requires_grad=True)
    reward=torch.tensor([1.2,-.7],dtype=torch.float64,requires_grad=True)
    z=torch.tensor(.4,dtype=torch.float64,requires_grad=True)
    result=api().trajectory_balance(z,pf,pb,stop,reward,mask)
    expected=torch.stack([z+pf[0,0]+pf[0,1]+stop[0]-reward[0]-pb[0,0]-pb[0,1],z+pf[1].sum()+stop[1]-reward[1]-pb[1].sum()])
    torch.testing.assert_close(result['residual'],expected)
    torch.testing.assert_close(result['loss'],expected.square().mean())
    args=(z,pf,pb,stop,reward)
    ga=torch.autograd.grad(result['loss'],args,retain_graph=True)
    gb=torch.autograd.grad(expected.square().mean(),args)
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b)
    assert ga[1][0,2]==0 and ga[2][0,2]==0
    balanced_reward=(z+torch.where(mask,pf-pb,0).sum(-1)+stop).detach()
    assert api().trajectory_balance(z,pf,pb,stop,balanced_reward,mask)['loss']==0
    assert torch.autograd.gradcheck(lambda z,pf,pbi,s,r:api().trajectory_balance(z,pf,torch.cat((torch.zeros_like(pbi[:,:1]),pbi),-1),s,r,mask)['loss'],(z,pf,pb[:,1:].detach().requires_grad_(),stop,reward))


def test_invalid_inputs_fail_closed():
    m=api();x=torch.ones(2,3)
    for bad in [x.long(),x[:0],x*float('nan')]:
        with pytest.raises(ValueError):m.diagonal_gaussian_log_density(bad,bad,bad)
    for mu,ls in [(x[:1],x),(x,x.double()),(x,x*1000)]:
        with pytest.raises(ValueError):m.diagonal_gaussian_log_density(x,mu,ls)
    with pytest.raises(ValueError):m.stop_log_probabilities(torch.tensor([float('inf')]))
    pf=torch.zeros(2,3);pb=pf.clone();s=torch.zeros(2);r=s.clone();z=torch.tensor(0.);mask=torch.ones(2,3,dtype=torch.bool)
    for badmask in [mask.float(),torch.tensor([[True,False,True],[True,True,True]]),mask & False]:
        with pytest.raises(ValueError):m.trajectory_balance(z,pf,pb,s,r,badmask)
    for zz,ff,bb,ss,rr in [(z,pf,pb,s+1,r),(z,pf,pb+1,s,r),(z,pf[:1],pb,s,r),(z.expand(2),pf,pb,s,r),(z,pf,pb,s,r*float('nan'))]:
        with pytest.raises(ValueError):m.trajectory_balance(zz,ff,bb,ss,rr,mask)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_gpu_full_density_balance_gradient_parity():
    def run(device):
        x=torch.tensor([[[.2,-.1],[.4,.7]],[[.1,.8],[-.2,.3]]],dtype=torch.float64,device=device)
        mu=torch.full_like(x,.1,requires_grad=True);ls=torch.full_like(x,-.3,requires_grad=True)
        logits=torch.zeros(2,2,dtype=x.dtype,device=device,requires_grad=True)
        z=torch.tensor(.7,dtype=x.dtype,device=device,requires_grad=True)
        stop,cont=api().stop_log_probabilities(logits)
        pf=api().diagonal_gaussian_log_density(x,mu,ls)+cont
        pb=torch.stack([torch.zeros(2,dtype=x.dtype,device=device),api().diagonal_gaussian_log_density(x[:,0],mu[:,1],ls[:,1])],-1)
        out=api().trajectory_balance(z,pf,pb,stop[:,-1],-x[:,-1].square().sum(-1),torch.ones(2,2,dtype=torch.bool,device=device))
        return out['loss'].detach().cpu(),[g.cpu() for g in torch.autograd.grad(out['loss'],(mu,ls,logits,z))]
    a,ga=run('cpu');b,gb=run('cuda');torch.testing.assert_close(a,b)
    for x,y in zip(ga,gb):torch.testing.assert_close(x,y)
