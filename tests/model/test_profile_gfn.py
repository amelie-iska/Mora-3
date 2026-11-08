"""Conditional policy must match explicit Gaussian/MLP and shared gradients."""
import copy
from dataclasses import replace
import importlib
import pytest
import torch
from torch.nn import functional as F
from test_molecule_profile_model import model, view
from mora3.reasoning.continuous_gfn import diagonal_gaussian_log_density, trajectory_balance


def api():
    m=importlib.import_module('mora3.model')
    assert importlib.util.find_spec('mora3.model.profile_gfn') is not None, 'conditional policy missing'
    return importlib.import_module('mora3.model.profile_gfn')


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_shared_conditional_kernels_equations_and_every_gradient(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    a=api();torch.manual_seed(11601)
    m=a.ProfileGFNPolicy(model(device).double(),dimension=2,horizon=3,hidden=5,
                         source_binding={'pool_sha256':'a'*64})
    vs=(view('profile_to_molecule'),view('profile_to_molecule','CellPainting'))
    x=torch.tensor([[.2,-.1],[.5,.8]],device=device,dtype=torch.float64)
    c=m.context(vs);ref=copy.deepcopy(m);cr=ref.profiles.condition(vs)
    actual=[];expected=[]
    for backward in (False,True):
        mu,ls,stop=m.parameters_at(x,1,c,backward=backward)
        h=torch.tanh(F.linear(torch.cat((x,x.new_full((2,1),1/3),cr),-1),
            (ref.backward_net if backward else ref.forward_net)[0].weight,
            (ref.backward_net if backward else ref.forward_net)[0].bias))
        head=ref.backward_head if backward else ref.forward_head
        mr,lr=F.linear(h,head.weight,head.bias).chunk(2,-1)
        sr=F.linear(h,ref.stop_head.weight,ref.stop_head.bias).squeeze(-1)
        for z,r in zip((mu,ls,stop),(mr,lr,sr)):torch.testing.assert_close(z,r)
        actual.extend((diagonal_gaussian_log_density(x,mu,ls),F.logsigmoid(stop)))
        expected.extend((torch.distributions.Normal(mr,lr.exp()).log_prob(x).sum(-1),-F.softplus(-sr)))
    z=m.log_partition(c);zr=F.linear(cr,ref.normalizer.weight,ref.normalizer.bias).squeeze(-1)
    la=sum(t.sum() for t in actual)+z.sum();lr=sum(t.sum() for t in expected)+zr.sum()
    ga=torch.autograd.grad(la,tuple(m.parameters()),allow_unused=True)
    gr=torch.autograd.grad(lr,tuple(ref.parameters()),allow_unused=True)
    for (name,_),g,r in zip(m.named_parameters(),ga,gr):
        assert (g is None)==(r is None),name
        if g is not None:torch.testing.assert_close(g,r,atol=1e-10,rtol=1e-9)
    assert len(list(m.parameters()))==len({id(p) for p in m.parameters()})
    assert ga[0] is not None
    scrub=tuple(replace(v,target_bytes=b'not-readable',provenance_bytes=b'not-readable') for v in vs)
    assert torch.equal(c,m.context(scrub))
    assert torch.equal(c.flip(0),m.context(vs[::-1]))
    assert not torch.equal(z,z.flip(0))


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_conditional_tb_per_row_reference_and_gradients(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    a=api();dtype=torch.float64
    z=torch.tensor([.2,.7],dtype=dtype,device=device,requires_grad=True)
    pf=torch.tensor([[.2,-.3],[.7,99]],dtype=dtype,device=device,requires_grad=True)
    pb=torch.zeros_like(pf,requires_grad=True);s=z*-1;r=z.detach()+1
    mask=torch.tensor([[True,True],[True,False]],device=device)
    out=a.conditional_trajectory_balance(z,pf,pb,s,r,mask)
    rows=[trajectory_balance(z[i],pf[i:i+1],pb[i:i+1],s[i:i+1],r[i:i+1],mask[i:i+1]) for i in range(2)]
    expected=torch.stack([q['loss'] for q in rows]).mean()
    torch.testing.assert_close(out['loss'],expected)
    for g,h in zip(torch.autograd.grad(out['loss'],(z,pf,pb),retain_graph=True),torch.autograd.grad(expected,(z,pf,pb))):torch.testing.assert_close(g,h)
    with pytest.raises(ValueError):a.conditional_trajectory_balance(z[:1],pf,pb,s,r,mask)


def test_policy_binding_direction_shapes_and_nonfinite_fail_closed():
    a=api();m=a.ProfileGFNPolicy(model(),dimension=2,horizon=3,hidden=5,source_binding={'pool_sha256':'a'*64})
    c=m.context((view('profile_to_molecule'),));x=torch.zeros(1,2)
    for xx,ll,cc in [(x,4,c),(x,1,c[:,:1]),(x.double(),1,c),(x*float('nan'),1,c)]:
        with pytest.raises(ValueError):m.parameters_at(xx,ll,cc)
    with pytest.raises(ValueError):m.context((view(),))
    state=copy.deepcopy(m.get_extra_state());state['source_binding']['pool_sha256']='b'*64
    with pytest.raises(ValueError):m.set_extra_state(state)
    state=copy.deepcopy(m.get_extra_state());state['horizon']=4
    with pytest.raises(ValueError):m.set_extra_state(state)


def test_conditional_tb_rejects_extra_rows_in_every_input():
    a=api();z=torch.zeros(2);pf=torch.zeros(2,2);s=torch.zeros(2);mask=torch.ones(2,2,dtype=torch.bool)
    for pb,stop,reward,m in [(torch.zeros(3,2),s,s,mask),(pf,torch.zeros(3),s,mask),
                              (pf,s,torch.zeros(3),mask),(pf,s,s,torch.ones(3,2,dtype=torch.bool))]:
        with pytest.raises(ValueError):a.conditional_trajectory_balance(z,pf,pb,stop,reward,m)
