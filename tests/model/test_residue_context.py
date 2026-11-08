import pytest
import torch
from torch.nn import functional as F
from mora3.model import residue_objective as module

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_context_position_equation_and_gradients(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    assert hasattr(module,'ContextualResidueHead'), 'position/context-aware native head missing'
    torch.manual_seed(6801)
    head=module.ContextualResidueHead(6,('Region','Disulfide bond')).to(device).double()
    x=torch.ones(9,6,device=device,dtype=torch.float64,requires_grad=True)
    context=torch.arange(6,device=device,dtype=torch.float64,requires_grad=True)
    z=head(x,context)
    p=torch.arange(9,device=device,dtype=x.dtype)[:,None]
    frequency=torch.exp(-torch.arange(0,6,2,device=device,dtype=x.dtype)*torch.log(torch.tensor(10000.,device=device,dtype=x.dtype))/6)
    pos=torch.stack((torch.sin(p*frequency),torch.cos(p*frequency)),dim=-1).flatten(1)
    h=x+pos+head.context(context)
    expected=head.output(h+F.gelu(head.local(h.T[None])[0].T))
    torch.testing.assert_close(z,expected,rtol=1e-12,atol=1e-12)
    assert not torch.equal(z[0],z[1])
    assert not torch.equal(z,head(x,context+1))
    z.square().mean().backward()
    assert x.grad.abs().sum()>0 and context.grad.abs().sum()>0
    assert all(p.grad is not None and p.grad.abs().sum()>0 for p in head.parameters())
    # A local head must preserve sequence length even at a single residue.
    assert head(x[:1],context).shape==(1,2)
    clone=module.ContextualResidueHead(6,('Region','Disulfide bond')).to(device).double()
    clone.load_state_dict(head.state_dict())
    assert torch.equal(clone(x,context),z)
    wrong=module.ContextualResidueHead(6,('Disulfide bond','Region')).to(device).double()
    with pytest.raises(ValueError):wrong.load_state_dict(head.state_dict())


def test_contract_rejects_ambiguous_axes_and_shapes():
    assert hasattr(module,'ContextualResidueHead'), 'position/context-aware native head missing'
    for axis in [(),('Region','Region'),('',)]:
        with pytest.raises(ValueError):module.ContextualResidueHead(6,axis)
    with pytest.raises(ValueError):module.ContextualResidueHead(3,('Region',))
    head=module.ContextualResidueHead(6,('Region',))
    for x,c in [(torch.ones(0,6),torch.ones(6)),(torch.ones(4,5),torch.ones(6)),(torch.ones(4,6),torch.ones(1,6))]:
        with pytest.raises(ValueError):head(x,c)
