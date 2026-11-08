"""Catch NCE sign/reduction/negative errors, normalized edit gradients and rank claims."""
import copy
import importlib
import math
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import pytest
import torch


def api():
    return importlib.import_module('mora3.reasoning.graphcg')


def editor(device='cpu'):
    torch.manual_seed(19)
    return api().LinearConceptEditor(torch.eye(3,dtype=torch.float64)).to(device)


def test_linear_equation_inverse_and_parameter_gradients():
    m=editor()
    with torch.no_grad():
        m.projection.weight.copy_(torch.diag(torch.tensor([3.,4.,5.])))
        m.projection.bias.zero_()
    z=torch.tensor([[1.,2.,3.],[-1.,0.,2.]],dtype=torch.float64,requires_grad=True)
    alpha=torch.tensor([2.,-3.],dtype=torch.float64,requires_grad=True); ids=torch.tensor([0,1])
    result=m(z,ids,alpha)
    torch.testing.assert_close(result,torch.tensor([[3.,2.,3.],[-1.,-3.,2.]],dtype=torch.float64))
    torch.testing.assert_close(m(result,ids,-alpha),z)
    torch.testing.assert_close(m(z,ids,alpha*0),z,rtol=0,atol=0)
    result.sum().backward()
    torch.testing.assert_close(z.grad,torch.ones_like(z))
    torch.testing.assert_close(alpha.grad,torch.ones_like(alpha))
    torch.testing.assert_close(m.projection.bias.grad,torch.tensor([-.75,2/3,2/3-.75],dtype=torch.float64))
    assert 'basis' in m.state_dict() and not m.basis.requires_grad
    saved=copy.deepcopy(m.state_dict());n=editor();n.load_state_dict(saved)
    torch.testing.assert_close(n(z,ids,alpha),result)


def test_nce_scalar_equation_and_all_view_gradients():
    # Independent scalar log(1+exp) oracle catches missing symmetric term and scaling.
    vals=[torch.tensor(x,dtype=torch.float64,requires_grad=True) for x in
          [[[.2,-.4],[.5,.7]],[[.3,.1],[-.2,.6]],[[-.4,.5],[.1,-.3]],[[.8,-.1],[.2,.9]]]]
    actual=api().symmetric_nce(*vals)
    def dot(x,y): return sum(x[k]*y[k] for k in range(2))
    oracle=sum(2*torch.log1p(torch.exp(-dot(vals[0][b],vals[1][b])))+
               torch.log1p(torch.exp(dot(vals[2][b],vals[1][b])))+
               torch.log1p(torch.exp(dot(vals[3][b],vals[0][b]))) for b in range(2))/2
    torch.testing.assert_close(actual,oracle)
    for a,b in zip(torch.autograd.grad(actual,vals,retain_graph=True),torch.autograd.grad(oracle,vals)):
        torch.testing.assert_close(a,b)
    torch.testing.assert_close(actual,api().symmetric_nce(vals[1],vals[0],vals[3],vals[2]))
    zero=torch.zeros(2,3,dtype=torch.float64)
    assert api().symmetric_nce(zero,zero,zero,zero).item()==pytest.approx(4*math.log(2))
    huge=torch.full((1,2),100.,dtype=torch.float64,requires_grad=True)
    loss=api().symmetric_nce(huge,huge,huge,huge);loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(huge.grad).all()


def test_objective_equation_and_regularizers():
    m=editor()
    with torch.no_grad(): m.projection.weight.copy_(torch.eye(3));m.projection.bias.zero_()
    z=torch.zeros(2,3,dtype=torch.float64)
    i=torch.tensor([0,1]);j=torch.tensor([1,2]);a=torch.ones(2,dtype=torch.float64);b=-a
    terms=api().graphcg_objective(m,z,z,i,a,j,b,coefficients=(2.,3.,4.))
    expected=2*math.log1p(math.exp(-1))+2*math.log(2)
    assert terms['nce'].item()==pytest.approx(expected)
    assert terms['similarity'].item()==0
    assert terms['sparsity'].item()==1
    assert terms['total'].item()==pytest.approx(2*expected+4)
    terms['total'].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_full_objective_input_and_parameter_gradcheck():
    m=editor();torch.manual_seed(2)
    u=torch.randn(2,3,dtype=torch.float64,requires_grad=True)
    v=torch.randn_like(u,requires_grad=True);a=torch.tensor([.8,-.5],dtype=torch.float64,requires_grad=True)
    b=torch.tensor([-.9,.4],dtype=torch.float64,requires_grad=True)
    i=torch.tensor([0,1]);j=torch.tensor([1,2])
    def fn(u,v,a,b,w,bias):
        # Functional call tests learned normalization gradients as well as input gradients.
        d=torch.nn.functional.normalize(torch.nn.functional.linear(m.basis,w,bias),dim=-1)
        return api().symmetric_nce(u+a[:,None]*d[i],v+a[:,None]*d[i],
                                  u+b[:,None]*d[j],v+b[:,None]*d[j])
    assert torch.autograd.gradcheck(fn,(u,v,a,b,m.projection.weight,m.projection.bias))
    # Verify public objective uses those exact edit paths (no detached parameters).
    actual=api().graphcg_objective(m,u,v,i,a,j,b)['total']
    expected=fn(u,v,a,b,m.projection.weight,m.projection.bias)
    args=(u,v,a,b,m.projection.weight,m.projection.bias)
    for x,y in zip(torch.autograd.grad(actual,args),torch.autograd.grad(expected,args)):
        torch.testing.assert_close(x,y)


def test_rank_diagnostics_distinguish_antiparallel_from_independent_axes():
    d=torch.eye(3,dtype=torch.float64)
    info=api().direction_diagnostics(d)
    assert info['rank']==3 and info['max_abs_cosine']==0 and info['off_diagonal_rms']==0
    d[1]=-d[0]
    info=api().direction_diagnostics(d)
    assert info['rank']==2 and info['max_abs_cosine']==1
    assert info['off_diagonal_rms']==pytest.approx(math.sqrt(1/3))


def test_invalid_contracts_fail_closed():
    for basis in [torch.zeros(0,3),torch.ones(4,3),torch.ones(2,3,dtype=torch.long),torch.full((2,3),float('nan'))]:
        with pytest.raises(ValueError):api().LinearConceptEditor(basis)
    m=editor();z=torch.ones(2,3,dtype=torch.float64);i=torch.tensor([0,1]);a=torch.ones(2,dtype=torch.float64)
    for zz,ii,aa in [(z,torch.tensor([-1,0]),a),(z,i.float(),a),(z,i,a[:,None]),(z.float(),i,a),
                      (z[:0],i[:0],a[:0]),(z,i,a*float('inf'))]:
        with pytest.raises(ValueError): m(zz,ii,aa)
    for j,b in [(i,-a),(i+1,a)]:
        with pytest.raises(ValueError):api().graphcg_objective(m,z,z,i,a,j,b)
    for c in [(-1,0,0),(1,float('nan'),0),(1,2)]:
        with pytest.raises(ValueError):api().graphcg_objective(m,z,z,i,a,i+1,-a,coefficients=c)
    with pytest.raises(ValueError):api().symmetric_nce(z,z[:1],z,z)
    with pytest.raises(ValueError):api().direction_diagnostics(z*0)
    with torch.no_grad():m.projection.weight.zero_();m.projection.bias.zero_()
    with pytest.raises(ValueError):m(z,i,a)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_gpu_objective_and_all_gradient_parity():
    cpu=editor();gpu=copy.deepcopy(cpu).cuda();torch.manual_seed(19)
    values=[torch.randn(4,3,dtype=torch.float64),torch.randn(4,3,dtype=torch.float64),
            torch.tensor([0,1,2,0]),torch.tensor([1.,2.,-1.,-2.],dtype=torch.float64),
            torch.tensor([1,2,0,2]),torch.tensor([-2.,-1.,2.,1.],dtype=torch.float64)]
    a=api().graphcg_objective(cpu,*values,coefficients=(1,.1,.1))
    b=api().graphcg_objective(gpu,*(x.cuda() for x in values),coefficients=(1,.1,.1))
    a['total'].backward();b['total'].backward()
    for k in a:torch.testing.assert_close(a[k],b[k].cpu())
    for p,q in zip(cpu.parameters(),gpu.parameters()):torch.testing.assert_close(p.grad,q.grad.cpu())
