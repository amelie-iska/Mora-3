import pytest
import torch
from mora3.model.flow_matching import gaussian_path, velocity_loss, euler_flow

@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_path_derivative_endpoints_and_masked_gradient(device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('no CUDA')
    x0=torch.linspace(-1,1,10000,device=device,dtype=torch.float64).reshape(2,5000)
    x1=x0.square(); t=torch.tensor([0.,1.],device=device,dtype=torch.float64)
    x,u=gaussian_path(x0,x1,t,sigma_min=.01)
    torch.testing.assert_close(x[0],x0[0]); torch.testing.assert_close(x[1],x1[1]+.01*x0[1])
    a,_=gaussian_path(x0,x1,torch.full_like(t,.3),sigma_min=.01)
    b,_=gaussian_path(x0,x1,torch.full_like(t,.30001),sigma_min=.01)
    torch.testing.assert_close((b-a)/.00001,u)
    p=(u+2).detach().requires_grad_(); mask=torch.ones_like(p,dtype=torch.bool);mask[:,7]=False
    p.data[:,7]=float('nan')
    loss=velocity_loss(p,u,mask); assert loss.item()==pytest.approx(4)
    loss.backward(); assert torch.all(p.grad[:,7]==0)
    assert velocity_loss(p,u,torch.zeros_like(mask)) is None

@pytest.mark.parametrize('sigma', [0.,.01])
def test_integrator_constant_velocity(sigma):
    a=torch.tensor([[2.,-1.]]); b=torch.tensor([[4.,3.]])
    _,u=gaussian_path(a,b,torch.zeros(1),sigma_min=sigma)
    out=euler_flow(lambda x,t:u,a,steps=7)
    torch.testing.assert_close(out,b+sigma*a)

@pytest.mark.parametrize('fault',['time','shape','sigma','dtype','nonfinite'])
def test_path_rejects_invalid(fault):
    a=torch.ones(2,3);b=a.clone();t=torch.zeros(2);s=.01
    if fault=='time': t[0]=1.1
    if fault=='shape': b=b[:1]
    if fault=='sigma': s=-1
    if fault=='dtype': b=b.double()
    if fault=='nonfinite': b[0,0]=float('nan')
    with pytest.raises(ValueError):gaussian_path(a,b,t,sigma_min=s)


def test_loss_rejects_broadcast_and_available_nan():
    a=torch.ones(2,3)
    with pytest.raises(ValueError):velocity_loss(a,a,torch.ones(1,3,dtype=torch.bool))
    with pytest.raises(ValueError):velocity_loss(a,a*float('nan'),torch.ones_like(a,dtype=torch.bool))
    with pytest.raises(ValueError):euler_flow(lambda x,t:x,a,steps=0)
    with pytest.raises(ValueError):euler_flow(lambda x,t:x[:1],a,steps=2)
