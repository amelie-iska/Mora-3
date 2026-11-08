"""Independent vector-basis witnesses; fixed Cartesian linear maps fail covariance."""
import importlib
import pytest
import torch


def cls():
    return importlib.import_module('mora3.model.current_frame_velocity').CurrentFrameVelocityReadout


def fixture(device='cpu', dtype=torch.float64):
    torch.manual_seed(32401)
    m = cls()(('pair_scalar_a', 'pair_scalar_b'), max_atoms=8).to(device=device, dtype=dtype)
    with torch.no_grad():
        m.coefficients.weight.copy_(torch.tensor([[.3, -.2]], device=device, dtype=dtype))
    x = torch.tensor([[0.,1.,2.],[2.,0.,1.],[-1.,3.,0.]], device=device, dtype=dtype, requires_grad=True)
    f = torch.arange(18, device=device, dtype=dtype).reshape(3,3,2).div(10).requires_grad_()
    free = torch.tensor([True,False,True], device=device)
    return m,x,f,free


def run(m,x,f,free,**kw):
    return m(x,f,free, origin='synthetic_literal', frame_kind='current_internal', **kw)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_independent_values_all_gradients_covariance_permutation(device,dtype):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    m,x,f,free=fixture(device,dtype)
    actual=run(m,x,f,free)
    # Independent scalar loops retain all N neighbors, including self, denominator N.
    w=m.coefficients.weight
    reference=torch.stack([torch.stack([sum(sum(w[0,k]*f[i,j,k] for k in range(2))*(x[j,c]-x[i,c]) for j in range(3))/3 if free[i] else x[i,c]*0 for c in range(3)]) for i in range(3)])
    tol=2e-5 if dtype==torch.float32 else 1e-11
    torch.testing.assert_close(actual,reference,rtol=tol,atol=tol)
    probe=torch.arange(9,device=device,dtype=dtype).reshape(3,3)/7
    ag=torch.autograd.grad((actual*probe).sum(),(x,f,w),retain_graph=True)
    rg=torch.autograd.grad((reference*probe).sum(),(x,f,w),retain_graph=True)
    for a,r in zip(ag,rg): torch.testing.assert_close(a,r,rtol=tol,atol=tol)
    q=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],device=device,dtype=dtype)
    torch.testing.assert_close(run(m,x@q.T+7,f,free),actual@q.T,rtol=tol,atol=tol)
    perm=torch.tensor([2,0,1],device=device)
    torch.testing.assert_close(run(m,x[perm],f[perm][:,perm],free[perm]),actual[perm],rtol=tol,atol=tol)
    assert torch.equal(actual[~free],torch.zeros_like(actual[~free]))
    assert ag[0][1].abs().sum()>0  # Fixed input geometry still conditions free atoms.


@pytest.mark.parametrize('bad',['reference_unregistered','registered_reference','',None])
def test_frame_refusal_before_learned_operation(bad):
    m,x,f,free=fixture()
    called=[]
    hook=m.coefficients.register_forward_pre_hook(lambda *a: called.append(1))
    with pytest.raises(ValueError,match='current_internal'):
        m(x,f,free,origin='synthetic_literal',frame_kind=bad)
    hook.remove()
    assert called==[]


@pytest.mark.parametrize('bad',['source_observed','generated',None])
def test_native_authority_refused(bad):
    m,x,f,free=fixture()
    with pytest.raises(ValueError,match='synthetic'):
        m(x,f,free,origin=bad,frame_kind='current_internal')
    for name in ('require_source_use','require_model_use','require_training_use'):
        with pytest.raises(ValueError): getattr(m,name)()


@pytest.mark.parametrize('kind',['nan_xyz','nan_features','wrong_pair_axis','wrong_dtype','bad_mask','empty','oversize'])
def test_complete_axes_finite_and_bounds_before_neural_work(kind):
    m,x,f,free=fixture()
    if kind=='nan_xyz': x=x.detach().clone();x[0,0]=float('nan')
    if kind=='nan_features': f=f.detach().clone();f[0,0,0]=float('nan')
    if kind=='wrong_pair_axis': f=f[:,:2]
    if kind=='wrong_dtype': f=f.float()
    if kind=='bad_mask': free=free.float()
    if kind=='empty': x=x[:0];f=f[:0,:0];free=free[:0]
    if kind=='oversize': m=cls()(('pair_scalar_a','pair_scalar_b'),max_atoms=2).double()
    called=[]
    hook=m.coefficients.register_forward_pre_hook(lambda *a: called.append(1))
    with pytest.raises(ValueError): run(m,x,f,free)
    hook.remove()
    assert called==[]


def test_singleton_all_fixed_targets_and_strict_checkpoint():
    m,x,f,free=fixture()
    assert torch.equal(run(m,x[:1],f[:1,:1],free[:1]),torch.zeros_like(x[:1]))
    assert torch.equal(run(m,x,f,torch.zeros_like(free)),torch.zeros_like(x))
    with pytest.raises(TypeError): run(m,x,f,free,target_xyz=x)
    state=m.state_dict()
    other=cls()(('pair_scalar_b','pair_scalar_a'),max_atoms=8).double()
    with pytest.raises(ValueError,match='configuration'): other.load_state_dict(state)
    same=cls()(('pair_scalar_a','pair_scalar_b'),max_atoms=8).double()
    same.load_state_dict(state)
    torch.testing.assert_close(run(same,x,f,free),run(m,x,f,free))


@pytest.mark.parametrize('axis,limit',[((),8),(('x','x'),8),(['x'],8),(('x',),True),(('x',),0)])
def test_invalid_configuration(axis,limit):
    with pytest.raises(ValueError): cls()(axis,max_atoms=limit)


def test_arithmetic_overflow_refused():
    m,x,f,free=fixture(dtype=torch.float32)
    with torch.no_grad(): m.coefficients.weight.fill_(3e38)
    with pytest.raises(ValueError,match='nonfinite'): run(m,x,f,free)
