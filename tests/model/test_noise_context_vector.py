"""Replay-qualified noise with caller synthetic invariant context only."""
import pytest
import torch
from test_independent_noise_vector import setup
import mora3.model.current_frame_velocity as mod


def inputs(device='cpu',dtype=torch.float64,width=8):
    a=setup(device,dtype);a.pop('readout')
    axis=tuple(f'h{i}' for i in range(width))
    c=mod.SplitContextualCoefficients(axis,max_atoms=8).to(device=device,dtype=dtype)
    with torch.no_grad():
        for p in c.parameters():p.copy_(torch.arange(p.numel(),device=device,dtype=dtype).reshape(p.shape)/31+.1)
    return dict(a,coefficients=c,context=torch.arange(3*width,device=device,dtype=dtype).reshape(3,width).div(29).requires_grad_(),context_axis=axis,origin='synthetic_literal',width_identity='diagnostic-width')


def run(a):
    assert hasattr(mod,'independent_noise_context_vector_diagnostic'),'missing contextual noise bridge'
    return mod.independent_noise_context_vector_diagnostic(**a)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('width',[8,12,16])
def test_equations_and_all_gradients(device,dtype,width):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('no CUDA')
    a=inputs(device,dtype,width);x=a['current_xyz'];h=a['context'];c=a['coefficients'];z=x-x.mean(0)
    before=a['binding'];v=run(a)
    ref=torch.stack([sum(((x[j]-x[i]).norm()*c.geometry.weight[0,0]+(z[i]*z[j]).sum()*c.geometry.weight[0,1]+(h[i]*c.receiver.weight[0]).sum()+(h[j]*c.neighbor.weight[0]).sum())*(x[j]-x[i]) for j in range(3))/3 for i in range(3)])
    torch.testing.assert_close(v,ref)
    operands=(x,h,*c.parameters());probe=torch.arange(9,device=device,dtype=dtype).reshape(3,3)
    for l,r in zip(torch.autograd.grad((v*probe).sum(),operands,retain_graph=True),torch.autograd.grad((ref*probe).sum(),operands)):
        torch.testing.assert_close(l,r,rtol=2e-4,atol=2e-4 if dtype==torch.float32 else 1e-9)
    assert a['binding']==before and not before.synthetic_diagnostic


@pytest.mark.parametrize('bad',['xyz','axis','context_hash','mask','context','context_axis','origin','owner','weight','bound','extra'])
def test_invalid_inputs_refuse_before_learned_calls(bad):
    a=inputs();c=a['coefficients'];calls=[]
    hooks=[m.register_forward_pre_hook(lambda *args:calls.append(1)) for m in (c.geometry,c.receiver,c.neighbor)]
    if bad=='xyz':a['current_xyz']=a['current_xyz']+1
    if bad=='axis':a['atom_axis']=a['atom_axis'][::-1]
    if bad=='context_hash':a['context_sha256']='a'*64
    if bad=='mask':a['free_atoms']=(False,True,True)
    if bad=='context':a['context']=a['context']*float('nan')
    if bad=='context_axis':a['context_axis']=a['context_axis'][::-1]
    if bad=='origin':a['origin']='observed'
    if bad=='owner':c.receiver=torch.nn.Linear(8,1)
    if bad=='weight':
        with torch.no_grad():c.neighbor.weight.fill_(float('nan'))
    if bad=='bound':c._max_atoms=2
    if bad=='extra':a['target_xyz']=torch.zeros(3,3)
    with pytest.raises((ValueError,TypeError)):run(a)
    for hook in hooks:hook.remove()
    assert calls==[]
