"""Ragged conditional rollout must retain row identity and exact path measures."""
import copy
import pytest
import torch
from torch.nn import functional as F
from test_molecule_profile_model import model, view
from mora3.model.profile_gfn import ProfileGFNPolicy, conditional_trajectory_balance
from mora3.reasoning.gfn_sampling import sample_paths, score_paths

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_conditional_ragged_paths_reference_and_shared_gradients(device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    torch.manual_seed(11701)
    p=ProfileGFNPolicy(model(device).double(),dimension=2,horizon=4,hidden=5,source_binding={'pool':'test'})
    views=tuple(view('profile_to_molecule', 'CellPainting' if i%2 else 'L1000') for i in range(24))
    verify_rollout(p,views,device)


def verify_rollout(p,views,device):
    c=p.context(views)
    rng=torch.Generator(device=device).manual_seed(11701)
    before=torch.get_rng_state().clone()
    cuda_before=torch.cuda.get_rng_state().clone() if device=="cuda" else None
    paths=sample_paths(p,len(c),generator=rng,context=c)
    assert torch.equal(before,torch.get_rng_state())
    if cuda_before is not None:assert torch.equal(cuda_before,torch.cuda.get_rng_state())
    again=sample_paths(p,len(c),generator=torch.Generator(device=device).manual_seed(11701),context=c)
    assert torch.equal(paths['states'],again['states'])
    assert not paths['states'].requires_grad
    assert len(paths['lengths'].unique())>1
    assert (paths['states'][~paths['mask']]==0).all()
    scores=score_paths(p,paths,context=c)
    ref=copy.deepcopy(p); cr=ref.context(views)
    pf=[];pb=[];st=[]
    for row,n in enumerate(paths['lengths'].tolist()):
        forward=[];back=[]
        for k in range(n):
            current=paths['states'][row:row+1,k]
            prev=paths['states'][row:row+1,k-1] if k else torch.zeros_like(current)
            mu,ls,logit=ref.parameters_at(prev,k,cr[row:row+1])
            f=torch.distributions.Normal(mu,ls.exp()).log_prob(current).sum()
            if k: f=f-F.softplus(logit).sum()
            forward.append(f)
            if k:
                mu,ls,_=ref.parameters_at(current,k+1,cr[row:row+1],backward=True)
                back.append(torch.distributions.Normal(mu,ls.exp()).log_prob(prev).sum())
            else: back.append(f.new_zeros(()))
        pf.append(F.pad(torch.stack(forward),(0,p.horizon-n)))
        pb.append(F.pad(torch.stack(back),(0,p.horizon-n)))
        st.append(-F.softplus(-ref.parameters_at(current,n,cr[row:row+1])[2]).sum() if n<p.horizon else f.new_zeros(()))
    expected=dict(log_forward=torch.stack(pf),log_backward=torch.stack(pb),log_stop=torch.stack(st))
    for key in scores:torch.testing.assert_close(scores[key],expected[key],atol=1e-12,rtol=1e-12)
    reward=paths['states'].new_zeros(len(c)) # algebraic fixture only, no therapeutic reward
    actual=conditional_trajectory_balance(p.log_partition(c),**scores,log_reward=reward,mask=paths['mask'])['loss']
    residual=ref.log_partition(cr)+(expected['log_forward']-expected['log_backward']).sum(1)+expected['log_stop']-reward
    wanted=residual.square().mean()
    ga=torch.autograd.grad(actual,tuple(p.parameters()),allow_unused=True)
    gr=torch.autograd.grad(wanted,tuple(ref.parameters()),allow_unused=True)
    errors=[]
    for a,b in zip(ga,gr):
        assert (a is None)==(b is None)
        if a is not None:
            torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-9)
            errors.append(float((a-b).abs().max()))
    # Rescoring a jointly permuted ragged batch preserves each conditional density.
    perm=torch.arange(len(c)-1,-1,-1,device=device)
    rotated={k:v[perm] if isinstance(v,torch.Tensor) else v for k,v in paths.items()}
    for key,value in score_paths(p,rotated,context=c[perm]).items():torch.testing.assert_close(value,scores[key][perm])
    forced=sample_paths(p,len(c),context=c,optional_stop=False,generator=rng)
    fs=score_paths(p,forced,context=c)
    assert (forced['lengths']==p.horizon).all() and (fs['log_stop']==0).all()
    assert (fs['log_backward'][:,0]==0).all()
    with pytest.raises(ValueError):sample_paths(p,len(c),context=c[:-1],generator=rng)
    with pytest.raises(ValueError):score_paths(p,paths,context=c[:-1])
    return dict(records=len(c),gradient_tensors=len(errors),max_gradient_error=max(errors),lengths=paths["lengths"].tolist(),loss=float(actual.detach()))


def test_conditional_sampling_requires_isolated_generator_before_rng_use():
    p=ProfileGFNPolicy(model(),dimension=2,horizon=2,hidden=5,source_binding={'pool':'test'})
    c=p.context((view('profile_to_molecule'),))
    before=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match='Generator'):sample_paths(p,1,context=c)
    assert torch.equal(before,torch.get_rng_state())
    with pytest.raises(ValueError):score_paths(p,dict(states=torch.zeros(0,2,2),lengths=torch.zeros(0,dtype=torch.long),mask=torch.zeros(0,2,dtype=torch.bool),optional_stop=True),context=c[:0])


def test_conditional_single_layer_forced_masses():
    p=ProfileGFNPolicy(model(),dimension=2,horizon=1,hidden=5,source_binding={'pool':'test'})
    c=p.context((view('profile_to_molecule'),))
    paths=sample_paths(p,1,context=c,generator=torch.Generator().manual_seed(11701))
    scores=score_paths(p,paths,context=c)
    assert paths['lengths'].tolist()==[1]
    assert scores['log_stop'].item()==0 and scores['log_backward'].item()==0
