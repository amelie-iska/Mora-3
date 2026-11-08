"""All-active contextual noise preflight and equal-case gradient reference."""
import pytest
import torch
from test_noise_context_vector import inputs
from test_independent_noise_ragged import inputs as ragged
import mora3.model.current_frame_velocity as mod


def cases(device='cpu',dtype=torch.float64,width=8):
    native,targets=ragged(device,dtype)
    out=[]
    for n in native:
        a=inputs(device,dtype,width)
        for k,v in n.items():
            if k!='readout':a[k]=v
        a['context']=a['context'][:len(n['atom_axis'])].detach().requires_grad_()
        out.append(a)
    return out,targets


def run(c,t,r=(True,True)):
    assert hasattr(mod,'independent_noise_context_vector_loss'),'missing contextual ragged loss'
    return mod.independent_noise_context_vector_loss(c,t,r)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('width',[8,12,16])
def test_reference_and_gradients(device,dtype,width):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    c,t=cases(device,dtype,width);actual=run(c,t);terms=[];operands=[]
    for a,u in zip(c,t):
        x=a['current_xyz'];h=a['context'];p=a['coefficients'];z=x-x.mean(0);n=len(x)
        v=torch.stack([sum(((x[j]-x[i]).norm()*p.geometry.weight[0,0]+(z[i]*z[j]).sum()*p.geometry.weight[0,1]+(h[i]*p.receiver.weight[0]).sum()+(h[j]*p.neighbor.weight[0]).sum())*(x[j]-x[i]) for j in range(n))/n for i in range(n)])
        terms.append((v-u.detach()).square().mean());operands.extend((x,h,*p.parameters()))
    ref=sum(terms)/2
    torch.testing.assert_close(actual.loss,ref)
    for a,b in zip(torch.autograd.grad(actual.loss,operands,retain_graph=True),torch.autograd.grad(ref,operands)):
        torch.testing.assert_close(a,b,rtol=3e-4,atol=3e-4 if dtype==torch.float32 else 1e-8)
    assert torch.autograd.grad(actual.loss,t,allow_unused=True)==(None,None)
    assert actual.active_cases==(0,1)
    assert not torch.isclose(ref,(terms[0]*3+terms[1]*2)/5)
    for name in ('require_source_use','require_model_use','require_training_use'):
        with pytest.raises(ValueError):getattr(actual,name)()


@pytest.mark.parametrize('bad',['context','axis','xyz','target','extra','owner','ready','length'])
def test_late_refusal_before_learning(bad):
    c,t=cases();ready=(True,True);calls=[]
    hooks=[m.register_forward_pre_hook(lambda *a:calls.append(1)) for a in c for m in a['coefficients'].modules()]
    if bad=='context':c[1]['context']=c[1]['context']*float('nan')
    if bad=='axis':c[1]['context_axis']=c[1]['context_axis'][::-1]
    if bad=='xyz':c[1]['current_xyz']=c[1]['current_xyz']+1
    if bad=='target':t[1]=t[1]*float('nan')
    if bad=='extra':c[1]['target_xyz']=t[1]
    if bad=='owner':c[1]['coefficients']=object()
    if bad=='ready':ready=(True,1)
    if bad=='length':t.pop()
    with pytest.raises(ValueError):run(c,t,ready)
    for h in hooks:h.remove()
    assert calls==[]


def test_inactive_payload_opaque():
    c,t=cases();c[1]=object();t[1]=object()
    assert run(c,t,(True,False)).active_cases==(0,)
