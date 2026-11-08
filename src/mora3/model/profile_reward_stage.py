"""Explicit frozen likelihood stages and a condition-constant GFN reward gauge."""
import copy
import math
import re

import torch

from .frozen_profile_reward import FrozenProfileLikelihoodReward
from .mixed_payload_projection import exact_wire
from .profile_gfn import conditional_trajectory_balance
from .profile_latent_decoder import ProfileLatentSelfiesDecoder
from .profile_source_training import profile_source_guided_tb


class ProfileRewardStageController:
    """Non-model owner of the independent frozen decoder and a private RNG.

    At optimizer counts divisible by 250, an explicit barrier must refresh
    the frozen decoder before another update. A checkpoint exactly at a
    barrier can represent either side of that refresh, with the phase explicit.
    """
    interval=250
    gauge_policy='frozen-full-source-logmeanlikelihood-at-zero-latent-v1'

    def __init__(self,template,*,source_group_manifest_sha256,private_seed):
        if (type(template) is not ProfileLatentSelfiesDecoder
                or type(source_group_manifest_sha256) is not str or not re.fullmatch('[0-9a-f]{64}',source_group_manifest_sha256)
                or type(private_seed) is not int or not 0<=private_seed<2**63):
            raise ValueError('exact reconstruction, source group digest, and explicit private seed required')
        self.template=template;self._template_owner=(template,)
        self.source_group_manifest_sha256=source_group_manifest_sha256
        self.private_seed=private_seed
        self.reference=FrozenProfileLikelihoodReward(template,stage_optimizer_update=0,
            source_group_manifest_sha256=source_group_manifest_sha256)
        self.generator=torch.Generator(device=template.latent_projection.weight.device).manual_seed(private_seed)
        self._generator_owner=(self.generator,)
        self._configuration=self._local_configuration()

    def _local_configuration(self):
        return dict(schema='Mora3.ProfileRewardStageController/v1',source_group_manifest_sha256=self.source_group_manifest_sha256,
            refresh_interval=self.interval,gauge=self.gauge_policy,private_seed=self.private_seed,
            generator_device=str(self.template.latent_projection.weight.device),
            model_policy=copy.deepcopy(self.template._policy),training_admission=False)

    def _validate(self):
        if (self.template is not self._template_owner[0] or self.generator is not self._generator_owner[0]
                or type(self.reference) is not FrozenProfileLikelihoodReward
                or self.reference.source_group_manifest_sha256!=self.source_group_manifest_sha256
                or str(self.generator.device)!=self._configuration['generator_device']
                or not exact_wire(self._local_configuration(),self._configuration)):
            raise ValueError('frozen profile stage, source binding, or private RNG owner drift')
        self.reference._validate()

    def _phase(self,steps,stage):
        if type(steps) is not int or steps<0 or type(stage) is not int or stage<0 or stage%self.interval:
            raise ValueError('exact nonnegative optimizer/stage coordinates required')
        expected=steps-steps%self.interval
        if stage==expected:return 'current'
        if steps==expected and stage+self.interval==expected:return 'refresh_pending'
        raise ValueError('invalid skipped or stale frozen profile stage')

    def require_current(self,steps):
        self._validate()
        if self._phase(steps,self.reference.stage_optimizer_update)!='current':
            raise ValueError('explicit frozen profile refresh required before next objective/update')
        return self.reference

    def prepare(self,steps):
        self._validate()
        if self._phase(steps,self.reference.stage_optimizer_update)=='refresh_pending':
            # Candidate construction and validation precede replacement; a
            # failed refresh leaves the old stage checkpointable at the barrier.
            candidate=FrozenProfileLikelihoodReward(self.template,stage_optimizer_update=steps,
                source_group_manifest_sha256=self.source_group_manifest_sha256)
            candidate._validate();self.reference=candidate
        return self.require_current(steps)

    def checkpoint(self,steps):
        self._validate();phase=self._phase(steps,self.reference.stage_optimizer_update)
        return dict(schema='Mora3.ProfileRewardStageState/v1',configuration=copy.deepcopy(self._configuration),
            optimizer_update=steps,phase=phase,reference=self.reference.checkpoint(),
            generator_state=self.generator.get_state().clone())

    def stage_restore(self,state,*,steps,staged_template):
        """Validate a checkpoint using a staged student; mutate no live owner."""
        self._validate()
        if (type(state) is not dict or set(state)!={'schema','configuration','optimizer_update','phase','reference','generator_state'}
                or state['schema']!='Mora3.ProfileRewardStageState/v1'
                or not exact_wire(state['configuration'],self._configuration)
                or type(state['optimizer_update']) is not int or state['optimizer_update']!=steps
                or type(staged_template) is not ProfileLatentSelfiesDecoder
                or not exact_wire(staged_template._policy,self.template._policy)):
            raise ValueError('exact frozen stage checkpoint/campaign/template identity required')
        checkpoint=state['reference']
        if type(checkpoint) is not dict or type(checkpoint.get('configuration')) is not dict:
            raise ValueError('complete frozen reference sidecar required')
        stage=checkpoint['configuration'].get('stage_optimizer_update')
        if state['phase']!=self._phase(steps,stage):raise ValueError('frozen stage refresh phase mismatch')
        private=state['generator_state']
        if type(private) is not torch.Tensor or private.ndim!=1 or private.dtype!=torch.uint8 or private.device.type!='cpu':
            raise ValueError('exact checkpointed private generator byte state required')
        gen=torch.Generator(device=staged_template.latent_projection.weight.device)
        gen.set_state(private)
        candidate=FrozenProfileLikelihoodReward.from_checkpoint(staged_template,checkpoint,
            source_group_manifest_sha256=self.source_group_manifest_sha256,stage_optimizer_update=stage)
        return candidate,gen

    def apply_staged_restore(self,staged):
        self.reference,self.generator=staged
        self._generator_owner=(self.generator,)
        self._validate()

    def gauged_objective(self,policy,views,groups,*,optimizer_update,optional_stop=True):
        """Subtract a fixed conditional log-likelihood offset, preserving p(z|x)."""
        reference=self.require_current(optimizer_update);views=tuple(views);groups=tuple(groups)
        p=reference.latent_projection.weight
        # This uses complete source observations only in the reward gauge.
        # The offset is constant in z and never an encoder/log-Z input feature.
        zeros=p.new_zeros((len(views),32))
        offset=reference.log_reward(views,zeros,groups,optimizer_update=optimizer_update,optional_stop=False)
        offset=offset+0.5*32*math.log(2*math.pi)
        raw=profile_source_guided_tb(self.template,policy,views,groups,reference,optimizer_update=optimizer_update,
            generator=self.generator,optional_stop=optional_stop)
        gauged=raw['log_reward']-offset
        objective=conditional_trajectory_balance(raw['log_z'],**raw['scores'],log_reward=gauged,mask=raw['paths']['mask'])
        return dict(raw,**objective,log_reward=gauged,raw_log_reward=raw['log_reward'],reward_gauge_offset=offset,
            estimated_raw_log_partition=raw['log_z']+offset,reward_gauge_policy=self.gauge_policy)
