"""Independent exhaustive assignment/derivative oracle, not matching-code mirrors."""
import itertools
import pytest
import torch
from mora3.model import instance_affinity

@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('n',[0,1,2,3])
def test_assignment_matches_exhaustive_value_and_gradient(device,n):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(22001)
    x=torch.randn(3,2,3,device=device,dtype=torch.float64,requires_grad=True)
    y=torch.tensor([[[1,1,0],[0,0,0]],[[0,1,1],[0,0,0]],[[0,0,0],[1,1,0]]],device=device,dtype=torch.bool)[:n]
    loss=instance_affinity.overlapping_set_objective(x,y)
    padded=torch.cat([y,torch.zeros(3-n,2,3,device=device,dtype=torch.bool)])
    candidates=[]
    for order in itertools.permutations(range(3)):
        target=padded[list(order)].to(x.dtype)
        candidates.append((torch.nn.functional.softplus(x)-x*target).mean())
    reference=torch.stack(candidates).min()
    torch.testing.assert_close(loss,reference,rtol=1e-13,atol=1e-13)
    torch.testing.assert_close(torch.autograd.grad(loss,x,retain_graph=True)[0],torch.autograd.grad(reference,x)[0],rtol=1e-13,atol=1e-13)
    torch.testing.assert_close(loss,instance_affinity.overlapping_set_objective(x,y.flip(0)))
    torch.testing.assert_close(loss,instance_affinity.overlapping_set_objective(x.flip(0),y))

def test_overlap_is_not_semantic_union():
    x=torch.tensor([[[5.,5.,-5.]], [[-5.,5.,5.]]],requires_grad=True)
    y=torch.tensor([[[True,True,False]],[[False,True,True]]])
    good=instance_affinity.overlapping_set_objective(x,y)
    merged=instance_affinity.overlapping_set_objective(x,y.any(0,keepdim=True))
    assert good<merged

@pytest.mark.parametrize('case',['too_many','empty_plane','shape','dtype','nan','no_slots','target_float'])
def test_invalid_inputs_refuse(case):
    x=torch.zeros(2,2,2);y=torch.ones(1,2,2,dtype=torch.bool)
    if case=='too_many':y=y.expand(3,-1,-1)
    elif case=='empty_plane':y.zero_()
    elif case=='shape':y=y[:,:,:1]
    elif case=='dtype':x=x.long()
    elif case=='nan':x[0,0,0]=float('nan')
    elif case=='no_slots':x=x[:0]
    elif case=='target_float':y=y.float()
    with pytest.raises(ValueError):instance_affinity.overlapping_set_objective(x,y)
