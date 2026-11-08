"""Exact packed/split scalar and relative-vector equivalence; synthetic only."""
import pytest
import torch
from mora3.model import current_frame_velocity as cv

@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('width',[8,12,16])
def test_packed_vector_all_gradients(device,dtype,width):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(33301)
    cls=getattr(cv,'SplitContextualCoefficients',None)
    assert cls is not None, 'separate split diagnostic operator missing'
    axis=tuple(f'h:{i}' for i in range(width))
    m=cls(axis,max_atoms=8).to(device=device,dtype=dtype)
    packed=torch.randn(1,2+2*width,device=device,dtype=dtype,requires_grad=True)
    with torch.no_grad():
        m.geometry.weight.copy_(packed[:,:2]);m.receiver.weight.copy_(packed[:,2:2+width]);m.neighbor.weight.copy_(packed[:,2+width:])
    x=torch.randn(5,3,device=device,dtype=dtype,requires_grad=True)
    h=torch.randn(5,width,device=device,dtype=dtype,requires_grad=True)
    # Independent current-only distance and centered-dot construction.
    rel=x[None,:,:]-x[:,None,:];z=x-x.mean(0)
    g=torch.stack((torch.linalg.vector_norm(rel,dim=-1),z@z.T),-1)
    f=torch.cat((g,h[:,None,:].expand(5,5,-1),h[None,:,:].expand(5,5,-1)),-1)
    c=m(g,h,origin='synthetic_literal',context_axis=axis)
    reference=(f*packed).sum(-1,keepdim=True)
    mask=torch.tensor([True,False,True,True,False],device=device)
    v=torch.where(mask[:,None],(c*rel).mean(1),0.)
    ref=torch.where(mask[:,None],(reference*rel).mean(1),0.)
    tol=2e-4 if dtype==torch.float32 else 1e-10
    torch.testing.assert_close(v,ref,rtol=tol,atol=tol)
    probe=torch.randn_like(v)
    a=torch.autograd.grad((v*probe).sum(),(x,h,*m.parameters()),retain_graph=True)
    b=torch.autograd.grad((ref*probe).sum(),(x,h,packed))
    for actual,expected in zip(a[:2],b[:2]):torch.testing.assert_close(actual,expected,rtol=tol,atol=tol)
    torch.testing.assert_close(torch.cat(a[2:],-1),b[2],rtol=tol,atol=tol)
    assert sum(p.numel() for p in m.parameters())==2+2*width
    assert not torch.any(v[~mask])
    perm=torch.tensor([3,1,4,0,2],device=device)
    torch.testing.assert_close(m(g[perm][:,perm],h[perm],origin='synthetic_literal',context_axis=axis),c[perm][:,perm],rtol=tol,atol=tol)
    for name in ['require_source_use','require_model_use','require_training_use']:
        with pytest.raises(ValueError):getattr(m,name)()
    with pytest.raises(RuntimeError):m.load_state_dict({'coefficients.weight':packed.detach()})

@pytest.mark.parametrize('bad',['origin','axis','nan','shape','empty','bound','dtype'])
def test_invalid_inputs_preflight(bad):
    cls=getattr(cv,'SplitContextualCoefficients',None)
    assert cls is not None,'separate split diagnostic operator missing'
    m=cls(('a','b'),max_atoms=3);g=torch.ones(2,2,2);h=torch.ones(2,2)
    kw=dict(origin='synthetic_literal',context_axis=('a','b'))
    if bad=='origin':kw['origin']='source_observed'
    if bad=='axis':kw['context_axis']=('b','a')
    if bad=='nan':h[0,0]=float('nan')
    if bad=='shape':g=g[:,:1]
    if bad=='empty':g=g[:0,:0];h=h[:0]
    if bad=='bound':g=torch.ones(4,4,2);h=torch.ones(4,2)
    if bad=='dtype':h=h.double()
    calls=[]
    hooks=[child.register_forward_hook(lambda *args:calls.append(1)) for child in m.children()]
    with pytest.raises(ValueError):m(g,h,**kw)
    assert calls==[]
    for hook in hooks:hook.remove()

@pytest.mark.parametrize('axis,bound',[((),8),(('a','a'),8),(['a'],8),(('a',),True),(('a',),257)])
def test_constructor_refuses(axis,bound):
    cls=getattr(cv,'SplitContextualCoefficients',None)
    assert cls is not None,'separate split diagnostic operator missing'
    with pytest.raises(ValueError):cls(axis,max_atoms=bound)
