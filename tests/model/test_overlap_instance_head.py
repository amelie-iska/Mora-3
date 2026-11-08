import inspect
import pytest
import torch
from mora3.model import instance_affinity

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_head_affine_reference_gradients_and_overlapping_outputs(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(22101)
    head=instance_affinity.OverlappingInstanceHead(4,slots=3,max_output_bytes=4096).to(device=device,dtype=torch.float64)
    x=torch.randn(1,4,3,5,device=device,dtype=torch.float64,requires_grad=True)
    z=head(x)
    reference=torch.einsum('qc,chw->qhw',head.projection.weight[:,:,0,0],x[0])+head.projection.bias[:,None,None]
    torch.testing.assert_close(z,reference,rtol=1e-13,atol=1e-13)
    params=(x,*head.parameters())
    for a,b in zip(torch.autograd.grad(z.square().sum(),params,retain_graph=True),torch.autograd.grad(reference.square().sum(),params)):
        torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
    with torch.no_grad():
        head.projection.weight.zero_();head.projection.bias.fill_(2)
    assert (head(x)>0).all() # Multiple memberships allowed; no slot softmax.
    assert list(inspect.signature(head.forward).parameters)==['features']
    y=torch.zeros(2,3,5,device=device,dtype=torch.bool);y[0,0,:2]=True;y[1,0,1:3]=True
    loss=instance_affinity.overlapping_set_objective(head(x),y)
    loss.backward()
    assert torch.isfinite(x.grad).all()

@pytest.mark.parametrize('case',['slots0','slots65','boolslots','channels','budget','shape','channels_input','nan','dtype','extent','budget_input'])
def test_head_refuses_invalid_or_overbudget_inputs(case):
    kw=dict(channels=4,slots=3,max_output_bytes=4096)
    if case=='slots0':kw['slots']=0
    if case=='slots65':kw['slots']=65
    if case=='boolslots':kw['slots']=True
    if case=='channels':kw['channels']=0
    if case=='budget':kw['max_output_bytes']=0
    if case in ['slots0','slots65','boolslots','channels','budget']:
        with pytest.raises(ValueError):instance_affinity.OverlappingInstanceHead(**kw)
        return
    head=instance_affinity.OverlappingInstanceHead(**kw)
    x=torch.zeros(1,4,3,5)
    if case=='shape':x=x.expand(2,-1,-1,-1)
    if case=='channels_input':x=x[:,:3]
    if case=='nan':x[0,0,0,0]=float('nan')
    if case=='dtype':x=x.double()
    if case=='extent':x=torch.zeros(1,4,1001,1)
    if case=='budget_input':x=torch.zeros(1,4,20,20)
    with pytest.raises(ValueError):head(x)

def test_head_strict_configuration_and_parameter_count():
    head=instance_affinity.OverlappingInstanceHead(4,slots=3,max_output_bytes=4096)
    assert sum(p.numel() for p in head.parameters())==15
    same=instance_affinity.OverlappingInstanceHead(4,slots=3,max_output_bytes=4096)
    same.load_state_dict(head.state_dict())
    different=instance_affinity.OverlappingInstanceHead(4,slots=3,max_output_bytes=8192)
    with pytest.raises(ValueError):different.load_state_dict(head.state_dict())
