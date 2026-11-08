"""New differentiable source-likelihood TB path; synthetic groups, no updates."""
import pytest,torch
from test_frozen_profile_reward import native
from mora3.model.profile_gfn import ProfileGFNPolicy
from mora3.model.frozen_profile_reward import FrozenProfileLikelihoodReward,ProfileObservedMolecules


def test_normalized_source_guided_tb_exact_scores_active_policy_gradients_and_frozen_reward():
    from mora3.model.profile_source_training import profile_source_guided_tb
    m,v=native();policy=ProfileGFNPolicy(m.decoder.profiles,dimension=32,horizon=8,hidden=64,
        source_binding=m._policy['source_binding'],parameterization_policy='mup-profile-gfn-v1')
    ref=FrozenProfileLikelihoodReward(m,stage_optimizer_update=0,source_group_manifest_sha256='a'*64)
    vs=(v,)*16;groups=(ProfileObservedMolecules(v.input_bytes,('CCO','CCO','CCCCO'),'a'*64),)*16
    generator=torch.Generator().manual_seed(2901);rng=torch.get_rng_state().clone()
    result=profile_source_guided_tb(m,policy,vs,groups,ref,optimizer_update=0,generator=generator,optional_stop=True)
    assert torch.equal(rng,torch.get_rng_state()) and (result['paths']['lengths']>1).any()
    scores=result['scores'];mask=result['paths']['mask'];explicit=result['log_z']+(scores['log_forward']*mask).sum(-1)+scores['log_stop']-(scores['log_backward']*mask).sum(-1)-result['log_reward']
    torch.testing.assert_close(result['residual'],explicit,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(result['loss'],explicit.square().mean(),rtol=1e-12,atol=1e-12)
    assert result['loss'].dtype==torch.float64 and result['log_reward'].requires_grad is False
    result['loss'].backward()
    for p in(policy.forward_head.weight,policy.backward_head.weight,policy.stop_head.weight,policy.normalizer.weight,
             policy.forward_net[0].context_weight,policy.forward_net[0].fixed_weight,policy.profiles.backbone.feature.weight):
        assert p.grad is not None and p.grad.abs().sum()>0
    assert all(p.grad is None for p in ref.parameters())
    assert m.decoder.output.weight.grad is None # Decoder supervised objective is separate.


def test_stale_stage_refusal_preserves_isolated_sampling_rng():
    from mora3.model.profile_source_training import profile_source_guided_tb
    m,v=native();p=ProfileGFNPolicy(m.decoder.profiles,dimension=32,horizon=8,hidden=64,
        source_binding=m._policy['source_binding'],parameterization_policy='mup-profile-gfn-v1')
    r=FrozenProfileLikelihoodReward(m,stage_optimizer_update=0,source_group_manifest_sha256='a'*64)
    generator=torch.Generator().manual_seed(2);before=generator.get_state().clone()
    g=ProfileObservedMolecules(v.input_bytes,('CCO',),'a'*64)
    with pytest.raises(ValueError):profile_source_guided_tb(m,p,(v,),(g,),r,optimizer_update=250,generator=generator)
    assert torch.equal(before,generator.get_state())
