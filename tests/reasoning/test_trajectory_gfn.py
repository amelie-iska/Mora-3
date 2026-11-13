"""Native continuous GFN path-space equations and shared gradients, zero updates."""
from dataclasses import replace
import math
import pytest
import torch
from test_molecule_profile_model import model,view
from mora3.model.profile_gfn import ProfileGFNPolicy,conditional_trajectory_balance
from mora3.reasoning.gfn_sampling import score_paths
from mora3.reasoning.trajectory_gfn import (PathSpaceReward,sample_native_path_batch,
    pathspace_objective,native_pathspace_objective,pathspace_balance)

H='a'*64


def setup():
    torch.manual_seed(20261110)
    policy=ProfileGFNPolicy(model().double(),dimension=2,horizon=3,hidden=5,source_binding={'pool_sha256':H})
    views=(view('profile_to_molecule'),view('profile_to_molecule','CellPainting'))
    batch=sample_native_path_batch(policy,views,generator=torch.Generator().manual_seed(17),source_context_sha256=H)
    guide=torch.tensor([[[0.,0.],[1.,0.],[1.,1.],[2.,1.]],[[0.,0.],[0.,1.],[1.,1.],[1.,2.]]],dtype=torch.float64)
    reward=PathSpaceReward(guide,source_sha256='b'*64,encoder_sha256='c'*64,
        alignment_sha256='d'*64,group_policy_sha256='e'*64,
        guide_groups=(('problem:a',),('problem:b',)),query_groups=(('problem:c',),('problem:d',)),
        scale=1.,metric_weight=.2,topology_weight=.1,evidence_kind='synthetic_fixture')
    return policy,views,batch,reward


def test_native_pathspace_tb_uses_all_steps_and_deterministic_prefix_deletion():
    policy,views,batch,reward=setup();context=policy.context(views)
    output=pathspace_objective(policy,views,batch,reward)
    scores=score_paths(policy,batch.paths,context=context)
    independent=conditional_trajectory_balance(policy.log_partition(context),scores['log_forward'],
        torch.zeros_like(scores['log_backward']),torch.zeros_like(scores['log_stop']),reward(batch.paths),batch.paths['mask'])
    torch.testing.assert_close(output['loss'],independent['loss'],rtol=1e-12,atol=1e-12)
    output['loss'].backward()
    assert policy.forward_head.weight.grad.abs().sum()>0
    assert policy.normalizer.weight.grad.abs().sum()>0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in policy.profiles.parameters())
    assert policy.backward_head.weight.grad is None
    assert output['backward_semantics']=='deterministic-prefix-deletion-Dirac'
    assert output['terminal_object']=='entire-ordered-reasoning-trajectory'
    assert output['optimizer_updates']==0 and output['biological_causality_verified']is False


def test_full_path_reward_distinguishes_same_endpoint_different_middle():
    _,_,batch,reward=setup();a={k:v.clone()if isinstance(v,torch.Tensor)else v for k,v in batch.paths.items()}
    b={k:v.clone()if isinstance(v,torch.Tensor)else v for k,v in a.items()}
    b['states'][:,0,:]+=5
    assert torch.equal(a['states'][:,-1],b['states'][:,-1])
    assert not torch.equal(reward(a),reward(b))
    assert not reward(a).requires_grad


def test_base_reward_has_independently_checkable_normalized_path_density():
    _,_,batch,reward=setup();zero=PathSpaceReward(reward.reference,source_sha256='b'*64,encoder_sha256='c'*64,
        alignment_sha256='d'*64,group_policy_sha256='e'*64,
        guide_groups=reward.guide_groups,query_groups=reward.query_groups,
        scale=.7,metric_weight=0.,topology_weight=0.,evidence_kind='synthetic_fixture')
    x=batch.paths['states'];delta=torch.cat((x[:,:1],x[:,1:]-x[:,:-1]),1)
    expected=torch.distributions.Normal(torch.zeros_like(delta),torch.full_like(delta,.7)).log_prob(delta).sum((1,2))
    torch.testing.assert_close(zero(batch.paths),expected,rtol=1e-12,atol=1e-12)
    assert zero.binding()['reward_partition_upper_bound']==1.
    # For the normalized base path sampler Z=1 and the exact TB residual is0.
    logpf=torch.distributions.Normal(torch.zeros_like(delta),torch.full_like(delta,.7)).log_prob(delta).sum(-1)
    assert pathspace_balance(torch.zeros(len(x),dtype=x.dtype),logpf,expected)['loss'].item()==pytest.approx(0.,abs=1e-24)


@pytest.mark.parametrize('damage',('state','policy','beam','context','reference'))
def test_stale_or_pruned_rollouts_cannot_masquerade_as_native_on_policy_samples(damage):
    p,vs,b,r=setup()
    if damage=='state':b.paths['states'][0,0,0]+=1
    elif damage=='policy':
        with torch.no_grad():p.forward_head.weight.add_(.01)
    elif damage=='beam':b=replace(b,sampler='fot-beam-selected')
    elif damage=='context':vs=vs[::-1]
    else:r.reference[0,1,0]+=1
    with pytest.raises(ValueError):pathspace_objective(p,vs,b,r)


def test_source_role_leakage_and_incomplete_paths_refuse():
    _,_,b,r=setup()
    kwargs=dict(source_sha256='b'*64,encoder_sha256='c'*64,alignment_sha256='d'*64,
        group_policy_sha256='e'*64,guide_groups=r.guide_groups,query_groups=r.query_groups,
        scale=1.,metric_weight=1.,topology_weight=0.,evidence_kind='model_generated')
    with pytest.raises(ValueError):PathSpaceReward(r.reference,guide_split='test',**kwargs)
    with pytest.raises(ValueError):PathSpaceReward(r.reference,**(kwargs|{'query_groups':r.guide_groups}))
    altered={k:v.clone()if isinstance(v,torch.Tensor)else v for k,v in b.paths.items()}
    altered['lengths'][0]=2;altered['mask'][0,-1]=False
    with pytest.raises(ValueError):r(altered)
    with pytest.raises(ValueError):native_pathspace_objective(object(),'x',(),b,r)


def test_hidden_profile_targets_do_not_change_sampled_paths_or_loss():
    p,vs,b,r=setup()
    scrub=tuple(replace(v,target_bytes=b'private outcome changed',provenance_bytes=b'audit changed')for v in vs)
    other=sample_native_path_batch(p,scrub,generator=torch.Generator().manual_seed(17),source_context_sha256=H)
    assert torch.equal(b.paths['states'],other.paths['states'])
    torch.testing.assert_close(pathspace_objective(p,vs,b,r)['loss'],pathspace_objective(p,scrub,other,r)['loss'],rtol=0,atol=0)
