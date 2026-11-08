import pytest
import torch
from mora3.model.maximin import maximin_objective


def oracle(x, edges, labels):
    # Independent max-min Floyd closure; fixtures avoid ties for gradients.
    n=len(labels)
    a=[[x.sum()*0 if i!=j else x.sum()*0+1 for j in range(n)] for i in range(n)]
    for k,(i,j) in enumerate(edges.tolist()):
        a[i][j]=a[j][i]=x[k].sigmoid()
    for k in range(n):
        a=[[torch.maximum(a[i][j],torch.minimum(a[i][k],a[k][j])) for j in range(n)] for i in range(n)]
    terms=[.5*(a[i][j]-float(labels[i]==labels[j]))**2 for i in range(n) for j in range(i+1,n) if labels[i]>=0 and labels[j]>=0]
    return torch.stack(terms).mean() if terms else x.sum()*0


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_equation_gradient_and_gradcheck(device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA absent')
    e=torch.tensor([[0,1],[1,2],[2,3],[0,3],[0,2]],device=device)
    y=torch.tensor([9,9,4,-1],device=device)
    x=torch.tensor([.9,-.7,.3,-1.2,.1],dtype=torch.double,device=device,requires_grad=True)
    actual=maximin_objective(x,e,y); expected=oracle(x,e,y)
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(torch.autograd.grad(actual,x)[0],torch.autograd.grad(expected,x)[0])
    assert torch.autograd.gradcheck(lambda v:maximin_objective(v,e,y),(x,))


def test_global_paths_not_local_edges_and_permutation():
    e=torch.tensor([[0,1],[1,2],[0,2]])
    x=torch.tensor([2.,1.,-3.],requires_grad=True)
    y=torch.tensor([5,5,5])
    loss=maximin_objective(x,e,y); loss.backward()
    assert x.grad[2]==0 and x.grad[1]<0
    torch.testing.assert_close(loss,maximin_objective(x.detach(),e,torch.tensor([99,99,99])))
    p=torch.tensor([2,0,1]); torch.testing.assert_close(loss,maximin_objective(x.detach()[p],e[p],y))


def test_ties_choose_input_order_and_ignore_endpoints_not_paths():
    e=torch.tensor([[0,1],[1,2],[0,2]])
    x=torch.zeros(3,requires_grad=True)
    loss=maximin_objective(x,e,torch.tensor([7,-1,7])); loss.backward()
    assert x.grad.tolist()==[0.,-.125,0.]
    empty=maximin_objective(x,e,torch.tensor([-1,-1,-1])); assert empty==0
    assert torch.autograd.grad(empty,x)[0].eq(0).all()


@pytest.mark.parametrize('edges,labels', [([[0,1]],[0,1,2]), ([[0,3]],[0,1]), ([[0,0]],[0]), ([[0,1]],[-2,1])])
def test_invalid_graph_refused(edges,labels):
    with pytest.raises(ValueError): maximin_objective(torch.zeros(len(edges)),torch.tensor(edges),torch.tensor(labels))


def test_nonfinite_refused():
    with pytest.raises(ValueError): maximin_objective(torch.tensor([float('nan')]),torch.tensor([[0,1]]),torch.tensor([0,1]))


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
def test_low_precision_counts_refused(dtype):
    # Full-image pair weights can exceed low-precision finite/range accuracy.
    with pytest.raises(ValueError):
        maximin_objective(torch.zeros(1,dtype=dtype),torch.tensor([[0,1]]),torch.tensor([0,1]))
