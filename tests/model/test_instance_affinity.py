import math
import pytest
import torch
from mora3.model.instance_affinity import affinity_targets, affinity_objective


def test_touching_instances_and_background_equation():
    labels=torch.tensor([[1,1,2],[0,0,2]])
    masks=labels[None]==torch.tensor([1,2])[:,None,None]
    y,valid=affinity_targets(masks)
    assert y.shape==(2,2,3)
    assert y[0].tolist()==[[True,False,False],[False,False,False]]
    assert y[1].tolist()==[[False,False,True],[False,False,False]]
    assert valid.sum()==7
    z=torch.zeros_like(y,dtype=torch.float64,requires_grad=True)
    loss=affinity_objective(z,masks)
    assert loss.item()==pytest.approx(math.log(2))
    loss.backward()
    torch.testing.assert_close(z.grad,(.5-y.double())*valid/7)
    other=affinity_targets(masks.flip(0))
    assert all(torch.equal(a,b) for a,b in zip((y,valid),other))


def test_overlap_ignored_without_flattening_and_empty():
    m=torch.tensor([[[True,True,False]],[[False,True,True]]])
    y,v=affinity_targets(m)
    assert not v.any()  # both edges touch ambiguous membership
    z=torch.ones((2,1,3),requires_grad=True)
    loss=affinity_objective(z,m)
    assert loss.item()==0
    loss.backward(); assert torch.equal(z.grad,torch.zeros_like(z))
    y,v=affinity_targets(torch.zeros((0,2,2),dtype=torch.bool))
    assert not y.any() and v.sum()==4


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_independent_membership_oracle_and_gradients(device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('no CUDA')
    torch.manual_seed(41)
    m=torch.rand((3,4,5),device=device)>.7
    y,v=affinity_targets(m)
    for axis,(dy,dx) in enumerate(((0,1),(1,0))):
        for row in range(4):
            for col in range(5):
                inside=row+dy<4 and col+dx<5
                a=set(torch.where(m[:,row,col])[0].tolist())
                b=set(torch.where(m[:,row+dy,col+dx])[0].tolist()) if inside else set()
                assert bool(v[axis,row,col])==(inside and len(a)<=1 and len(b)<=1)
                if v[axis,row,col]: assert bool(y[axis,row,col])==bool(a & b)
    z=torch.randn((2,4,5),dtype=torch.float64,device=device,requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: affinity_objective(x,m),(z,))
    for bad in (m.float(),m[0]):
        with pytest.raises(ValueError): affinity_targets(bad)
    with pytest.raises(ValueError): affinity_objective(z[:1],m)
    with pytest.raises(ValueError): affinity_objective(z.detach()*float('nan'),m)
