"""Detect softmax substitution, wrong max-plus axes, mask leakage and bad gradients."""
import copy
import importlib
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import pytest
import torch


def api():
    return importlib.import_module('mora3.model.tropical')


def scalar_core(q,k,v,valid):
    batches=[]
    for b in range(q.shape[0]):
        heads=[]
        for h in range(q.shape[1]):
            rows=[]
            for i in range(q.shape[2]):
                scores=[]; js=[]
                for j in range(k.shape[2]):
                    if valid[b,j]:
                        ds=[q[b,h,i,r]-k[b,h,j,r] for r in range(q.shape[3])]
                        scores.append(-(max(ds)-min(ds))); js.append(j)
                rows.append(torch.stack([max(s+v[b,h,j,r] for s,j in zip(scores,js))
                                         for r in range(v.shape[3])]))
            heads.append(torch.stack(rows))
        batches.append(torch.stack(heads))
    return torch.stack(batches)


def fixture(device='cpu'):
    torch.manual_seed(19)
    return [torch.randn(2,2,n,d,dtype=torch.double,device=device,requires_grad=True)
            for n,d in [(3,4),(5,4),(5,3)]]


def test_equations_gradients_and_gradcheck():
    q,k,v=fixture(); mask=torch.tensor([[True,False,True,True,True],[True]*5])
    result=api().tropical_context(q,k,v,mask); oracle=scalar_core(q,k,v,mask)
    torch.testing.assert_close(result,oracle)
    w=torch.randn_like(result)
    actual=torch.autograd.grad((result*w).sum(),(q,k,v),retain_graph=True)
    expected=torch.autograd.grad((oracle*w).sum(),(q,k,v))
    for a,b in zip(actual,expected): torch.testing.assert_close(a,b)
    assert torch.autograd.gradcheck(lambda *xs:api().tropical_context(*xs,mask),(q,k,v))


def test_maxplus_projection_equation_and_ties():
    x=torch.tensor([[1.,1.]],requires_grad=True); w=torch.tensor([[2.],[2.]],requires_grad=True)
    y=api().maxplus(x,w); assert y.item()==3
    y.sum().backward()
    assert x.grad.tolist()==[[1.,0.]] and w.grad.tolist()==[[1.],[0.]]
    q=torch.zeros(1,1,1,2,requires_grad=True)
    k=torch.zeros(1,1,2,2,requires_grad=True)
    v=torch.ones(1,1,2,1,requires_grad=True)
    api().tropical_context(q,k,v,torch.ones(1,2,dtype=torch.bool)).sum().backward()
    assert v.grad.flatten().tolist()==[1.,0.]
    assert q.grad.abs().sum()==0 and k.grad.abs().sum()==0


def test_projective_shift_permutation_and_masked_key_gradients():
    q,k,v=fixture(); mask=torch.tensor([[True,False,True,True,True],[True]*5])
    y=api().tropical_context(q,k,v,mask)
    shifted=api().tropical_context(q+torch.randn(2,2,3,1),k+torch.randn(2,2,5,1),v,mask)
    torch.testing.assert_close(y,shifted)
    order=torch.tensor([3,1,4,0,2])
    torch.testing.assert_close(y,api().tropical_context(q,k[:,:,order],v[:,:,order],mask[:,order]))
    y.sum().backward()
    assert k.grad[0,:,1].abs().sum()==0 and v.grad[0,:,1].abs().sum()==0
    altered=v.detach().clone(); altered[0,:,1]=1e6
    torch.testing.assert_close(y,api().tropical_context(q,k,altered,mask))


def test_invalid_contracts():
    q,k,v=fixture(); mask=torch.ones(2,5,dtype=torch.bool)
    for args in [(q,k,v,mask.float()),(q,k,v,~mask),(q,k[:,:,:,:2],v,mask),
                 (q,k,v.float(),mask),(q,k,v,mask[:,:3])]:
        with pytest.raises(ValueError): api().tropical_context(*args)
    bad=q.detach().clone(); bad[0,0,0,0]=float('nan')
    with pytest.raises(ValueError): api().tropical_context(bad,k,v,mask)
    with pytest.raises(ValueError): api().TropicalCodeAttention(7,2)


def scalar_layer(m,x,mask):
    z=x.relu().log1p().reshape(x.shape[0],x.shape[1],m.heads,m.head_width).transpose(1,2)
    projected=[]
    for w in (m.query,m.key,m.value):
        projected.append(torch.stack([torch.stack([z[...,i]+w[o,i] for i in range(m.head_width)],-1).max(-1).values
                                      for o in range(m.head_width)],-1))
    c=scalar_core(*projected,mask).transpose(1,2).flatten(2).expm1()
    return c @ m.output.weight.T


def test_code_wrapper_equations_and_parameter_gradients():
    torch.manual_seed(19); m=api().TropicalCodeAttention(8,2).double()
    x=(torch.rand(2,3,8,dtype=torch.double)+.1).requires_grad_()
    mask=torch.tensor([[True,True,False],[True]*3])
    a=m(x,mask); b=scalar_layer(m,x,mask)
    torch.testing.assert_close(a,b)
    params=(x,*m.parameters()); weights=torch.randn_like(a)
    ga=torch.autograd.grad((a*weights).sum(),params,retain_graph=True)
    gb=torch.autograd.grad((b*weights).sum(),params)
    for u,v in zip(ga,gb): torch.testing.assert_close(u,v)


def test_backbone_selection_padding_and_target_exclusion():
    from mora3.model.training import RNAClassifier
    from mora3.model.tokengt import TokenGT, DenseAttention
    from mora3.tokenization.shortest_path import ByteTokenizer
    from test_training import record
    m=RNAClassifier(ByteTokenizer(controls=('pad','node','edge')),('a','b'),attention='tropical-code')
    assert isinstance(m.backbone.layers[0].attention,api().TropicalCodeAttention)
    torch.testing.assert_close(m([record('a')]),m([record('b')]),rtol=0,atol=0)
    assert isinstance(TokenGT(4,4).layers[0].attention,DenseAttention)
    with pytest.raises(ValueError): TokenGT(4,4,attention='unknown')
    from mora3.model.tokengt import orthogonal_identifiers
    def graph(n):
        return dict(features=torch.randn(n,4),endpoints=torch.arange(n)[:,None].repeat(1,2),
                    roles=torch.zeros(n,dtype=torch.long),identifiers=orthogonal_identifiers(n,4))
    net=TokenGT(4,4,width=8,heads=2,attention='tropical-code'); a,b=graph(1),graph(3)
    batched,valid=net([a,b]); alone,_=net([a])
    torch.testing.assert_close(batched[0,:2],alone[0]); assert batched[~valid].abs().sum()==0


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_gpu_forward_backward_parity():
    torch.manual_seed(19); cpu=api().TropicalCodeAttention(8,2); gpu=copy.deepcopy(cpu).cuda()
    x=torch.randn(2,4,8,requires_grad=True); z=x.detach().cuda().requires_grad_()
    mask=torch.tensor([[True,True,False,False],[True]*4])
    a,b=cpu(x,mask),gpu(z,mask.cuda())
    torch.testing.assert_close(a,b.cpu(),atol=2e-6,rtol=2e-5)
    a.square().sum().backward(); b.square().sum().backward()
    for u,v in [(x.grad,z.grad),*[(p.grad,q.grad) for p,q in zip(cpu.parameters(),gpu.parameters())]]:
        torch.testing.assert_close(u,v.cpu(),atol=2e-4,rtol=2e-4)
