"""Source-likelihood-guided latent GFN objective; caller owns training admission.

Uses one live target-free context for detached rollout and differentiable scoring.
The reference decoder stays independently frozen for the explicit250-update
stage. Complete source groups must come from the reviewed native group reader;
this function does not issue source eligibility or a campaign start permission.
"""
import torch
from .profile_gfn import ProfileGFNPolicy,conditional_trajectory_balance
from .profile_latent_decoder import ProfileLatentSelfiesDecoder
from .frozen_profile_reward import FrozenProfileLikelihoodReward,ProfileObservedMolecules
from .mixed_payload_projection import exact_wire
from mora3.reasoning.gfn_sampling import sample_paths,score_paths


def profile_source_guided_tb(reconstruction,policy,views,groups,reference,*,optimizer_update,
        generator,optional_stop=True):
    if (type(reconstruction)is not ProfileLatentSelfiesDecoder or type(policy)is not ProfileGFNPolicy
            or type(reference)is not FrozenProfileLikelihoodReward
            or policy.profiles is not reconstruction.decoder.profiles
            or not exact_wire(reconstruction._policy,policy.get_extra_state())
            or not exact_wire(reference._base_configuration['source_owner']['profile_policy'],policy.get_extra_state())
            or type(optimizer_update)is not int
            or not reference.stage_optimizer_update<=optimizer_update<reference.stage_optimizer_update+250
            or type(optional_stop)is not bool or not isinstance(generator,torch.Generator)):
        raise ValueError('exact live profile policy/reconstruction and current frozen source stage required')
    reference._validate();views=tuple(views);groups=tuple(groups)
    if not views or len(views)!=len(groups):raise ValueError('complete matching source groups required')
    for v,g in zip(views,groups):
        if (type(g)is not ProfileObservedMolecules or g.condition_bytes!=v.input_bytes
                or g.source_group_manifest_sha256!=reference.source_group_manifest_sha256
                or type(g.smiles)is not tuple or not g.smiles):
            raise ValueError('complete same-condition source observations required before rollout')
        for smiles in g.smiles:reference.codec.encode(smiles,max_tokens=reference.max_tokens)
    context=policy.context(views)
    paths=sample_paths(policy,len(views),context=context,generator=generator,optional_stop=optional_stop)
    terminal=reconstruction.terminal_latents(paths)
    reward=reference.log_reward(views,terminal,groups,optimizer_update=optimizer_update,optional_stop=optional_stop)
    # Accumulate all normalized log densities in FP64 while retaining gradients
    # to the live policy and its shared encoder; reward remains detached.
    scores={k:v.double()for k,v in score_paths(policy,paths,context=context).items()}
    log_z=policy.log_partition(context).double()
    result=conditional_trajectory_balance(log_z,**scores,log_reward=reward,mask=paths['mask'])
    return dict(result,paths=paths,scores=scores,log_z=log_z,log_reward=reward,
        frozen_stage_optimizer_update=reference.stage_optimizer_update,
        next_explicit_refresh_optimizer_update=reference.stage_optimizer_update+250,
        source_group_manifest_sha256=reference.source_group_manifest_sha256)
