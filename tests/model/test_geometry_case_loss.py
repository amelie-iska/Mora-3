"""Equal case weighting must differ from coordinate weighting on ragged atoms."""
import pytest
import torch
from mora3.model.geometry_case_loss import active_geometry_cases, geometry_case_velocity_loss

DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])

@pytest.mark.parametrize('device', DEVICES)
def test_ragged_reference_and_gradients(device):
    p = [torch.ones((1,3),dtype=torch.float64,device=device,requires_grad=True),
         torch.full((3,3),3.,dtype=torch.float64,device=device,requires_grad=True)]
    u = [torch.zeros_like(x,requires_grad=True) for x in p]
    masks = [torch.ones(x.shape[0],dtype=torch.bool,device=device) for x in p]
    loss = geometry_case_velocity_loss(p,u,masks,(True,True))
    assert loss.item() == 5
    assert torch.cat(p).square().mean().item() == 7
    loss.backward()
    for x in p:
        torch.testing.assert_close(x.grad, x.detach()/(x.shape[0]*3))
    assert all(x.grad is None for x in u)

@pytest.mark.parametrize('device', DEVICES)
def test_only_free_atoms_and_available_cases_contribute(device):
    p = [torch.tensor([[2.,2.,2.],[float('nan')]*3],device=device,requires_grad=True),
         torch.full((1,3),float('nan'),device=device,requires_grad=True)]
    masks=[torch.tensor([True,False],device=device),torch.tensor([True],device=device)]
    assert active_geometry_cases(masks,(True,False)) == (0,)
    loss=geometry_case_velocity_loss(p,[torch.zeros_like(x) for x in p],masks,(True,False))
    assert loss.item()==4
    loss.backward()
    assert p[1].grad is None
    assert p[0].grad[1].eq(0).all()

@pytest.mark.parametrize('required,masks', [((False,),[torch.tensor([True])]),((True,),[torch.tensor([False])])])
def test_inactive_preflight_preserves_rng(required,masks):
    before=torch.random.get_rng_state().clone()
    with pytest.raises(ValueError,match='active'):
        active_geometry_cases(masks,required)
    assert torch.equal(before,torch.random.get_rng_state())

@pytest.mark.parametrize('masks,required', [([],()),([torch.ones(2)],(True,)),([torch.tensor([],dtype=torch.bool)],(True,)),([torch.tensor([[True]])],(True,)),([torch.tensor([True])],(1,)),([torch.tensor([True])],())])
def test_invalid_preflight(masks,required):
    with pytest.raises(ValueError): active_geometry_cases(masks,required)

@pytest.mark.parametrize('failure', ['rank','atoms','dtype','nonfinite','length','overflow'])
def test_invalid_active_arrays(failure):
    p=torch.ones((2,3),dtype=torch.float64);u=torch.zeros_like(p)
    if failure=='rank':p=p.flatten()
    if failure=='atoms':p=p[:1]
    if failure=='dtype':u=u.float()
    if failure=='nonfinite':p[0,0]=float('inf')
    if failure=='overflow':p.fill_(1e308)
    with pytest.raises(ValueError):
        geometry_case_velocity_loss([p],[] if failure=='length' else [u],[torch.tensor([True,True])],(True,))
