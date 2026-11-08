"""Opt-in profile-conditioned layered continuous GFN kernels.

Independent concatenated-context adaptation of 2301.12594v1 sections 3.5/4.2.
States are Euclidean reasoning latents, not molecules or therapeutic rewards.
No rollout, decoder bridge, reward, or optimizer is implied by this interface.
"""
import copy
import math
from .mixed_payload_projection import exact_wire
from .fixed_context_projection import FixedContextProjection
import json
import torch
from torch import nn
from .molecule_profile import SharedMoleculeProfileModel
from mora3.reasoning.continuous_gfn import _finite, trajectory_balance


class ProfileGFNPolicy(nn.Module):
    def __init__(self, profiles, *, dimension, horizon, hidden=64, source_binding,parameterization_policy='legacy'):
        super().__init__()
        if not isinstance(profiles, SharedMoleculeProfileModel):
            raise ValueError('native shared profile encoder required')
        if any(type(v) is not int or v<=0 for v in (dimension,horizon,hidden)):
            raise ValueError('positive dimensions and horizon required')
        if not isinstance(source_binding,dict) or not source_binding:
            raise ValueError('explicit source binding required')
        if parameterization_policy not in ('legacy','mup-profile-gfn-v1'):
            raise ValueError('known explicit profile GFN parameterization required')
        new=parameterization_policy!='legacy'
        if new:profiles.validate_parameterization_configuration()
        elif getattr(profiles,'parameterization_policy','legacy')!='legacy':
            raise ValueError('new profile owner requires explicit GFN parameterization')
        self.parameterization_policy=parameterization_policy
        self._profiles_reference=(profiles,)
        self._mup_configuration=None
        # Immutable private copy; source identity never enters neural features.
        self._source_binding=json.dumps(source_binding,sort_keys=True,allow_nan=False)
        self.profiles=profiles
        self.dimension=dimension;self.horizon=horizon;self.hidden=hidden
        self.context_width=profiles.shared.payload.embedding.embedding_dim
        if new:
            spec=profiles._mup_configuration
            fixed=tuple('latent-coordinate:'+str(i) for i in range(dimension))+('layer-over-horizon',)
            outputs=tuple('fixed-GFN-hidden:'+str(i) for i in range(hidden))
            options=dict(source_namespace='profile-GFN-fixed-latent-time-context-v1',
                base_context_width=spec['base_width'],context_mult=spec['output_mult'])
            self.forward_net=nn.Sequential(FixedContextProjection(self.context_width,fixed,outputs,**options),nn.Tanh())
            self.backward_net=nn.Sequential(FixedContextProjection(self.context_width,fixed,outputs,**options),nn.Tanh())
        else:
            self.forward_net=nn.Sequential(nn.Linear(dimension+1+self.context_width,hidden),nn.Tanh())
            self.backward_net=nn.Sequential(nn.Linear(dimension+1+self.context_width,hidden),nn.Tanh())
        self.forward_head=nn.Linear(hidden,2*dimension)
        self.backward_head=nn.Linear(hidden,2*dimension)
        self.stop_head=nn.Linear(hidden,1)
        self.normalizer=nn.Linear(self.context_width,1)
        if new:
            with torch.no_grad():
                ratio=math.sqrt(self.context_width/spec['base_width'])
                self.normalizer.weight.mul_(ratio);self.normalizer.bias.mul_(ratio)
            self._mup_configuration=self._current_mup_configuration()
        p=next(profiles.parameters());self.to(device=p.device,dtype=p.dtype)

    def _current_mup_configuration(self):
        spec=self.profiles._mup_configuration
        return dict(schema='profile-GFN-parameterization-v1',policy=self.parameterization_policy,
            context_width=self.context_width,base_context_width=spec['base_width'],context_mult=spec['output_mult'],
            dimension=self.dimension,horizon=self.horizon,hidden=self.hidden,
            hidden_role='fixed-finite-coordinate-set-not-main-width-axis',
            forward=self.forward_net[0].get_extra_state(),backward=self.backward_net[0].get_extra_state(),
            finite_heads='retain-native-fixed-hidden-fan-in-and-adamLR1',
            normalizer='native-fresh-times-sqrt-context-width-over-base-input-divided-by-context-width-over-base')

    def validate_parameterization_configuration(self):
        if self._mup_configuration is None:raise ValueError('explicit applied source GFN adaptation required')
        if (self.profiles is not self._profiles_reference[0] or self.parameterization_policy!='mup-profile-gfn-v1'
                or set(self._modules)!={'profiles','forward_net','backward_net','forward_head','backward_head','stop_head','normalizer'}):
            raise ValueError('source GFN fixed/context/base configuration drift')
        for name in ('forward_net','backward_net'):
            net=getattr(self,name)
            if type(net) is not nn.Sequential or set(net._modules)!={'0','1'} or type(net[0]) is not FixedContextProjection or type(net[1]) is not nn.Tanh:
                raise ValueError('source GFN split projection/activation configuration drift')
        for name,shape in (('forward_head',(2*self.dimension,self.hidden)),('backward_head',(2*self.dimension,self.hidden)),
                           ('stop_head',(1,self.hidden)),('normalizer',(1,self.context_width))):
            head=getattr(self,name)
            if (type(head) is not nn.Linear or head._modules or head._buffers
                    or (head.out_features,head.in_features)!=shape or tuple(head.weight.shape)!=shape
                    or tuple(head.bias.shape)!=(shape[0],)):
                raise ValueError('source GFN finite head/normalizer configuration drift')

        if not exact_wire(self._current_mup_configuration(),self._mup_configuration):
            raise ValueError('source GFN fixed/context/base configuration drift')

    def get_extra_state(self):
        if self._mup_configuration is not None:self.validate_parameterization_configuration()
        result=dict(version=1,dimension=self.dimension,horizon=self.horizon,
            hidden=self.hidden,context_width=self.context_width,
            source_binding=json.loads(self._source_binding),profiles=self.profiles.get_extra_state(),
            kernels='concatenated-context-time-tanh-diagonal-Gaussian-v1',
            stopping='optional-Bernoulli-forced-origin-and-horizon',
            normalizer='conditional-linear-logZ',state_space='time-layered-Euclidean-latent')
        if self._mup_configuration is not None:
            result.update(version=2,kernels='split-fixed-latent-time-and-finite-hidden-context-readout-v1',
                mup=copy.deepcopy(self._mup_configuration))
        return copy.deepcopy(result)

    def set_extra_state(self,state):
        if state!=self.get_extra_state():raise ValueError('profile GFN configuration mismatch')

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        if self._mup_configuration is not None:
            for key,value in self.state_dict().items():
                if not isinstance(value,torch.Tensor) and not exact_wire(state_dict.get(prefix+key),value):
                    raise ValueError('source GFN checkpoint configuration/policy mismatch')
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def context(self,views):
        views=self.profiles._validate(views)
        if any(v.direction!='profile_to_molecule' for v in views):
            raise ValueError('profile_to_molecule direction required')
        return self.profiles.condition(views)

    def _context(self,context):
        if self._mup_configuration is not None:self.validate_parameterization_configuration()
        _finite(context)
        p=self.normalizer.weight
        if (context.ndim!=2 or context.shape[1]!=self.context_width
                or context.dtype!=p.dtype or context.device!=p.device):
            raise ValueError('matching complete shared context required')

    def parameters_at(self,states,layer,context,*,backward=False):
        _finite(states);self._context(context)
        if (states.shape!=(len(context),self.dimension) or states.dtype!=context.dtype
                or states.device!=context.device or type(layer) is not int
                or not 0<=layer<=self.horizon or type(backward) is not bool):
            raise ValueError('matching states, context and valid layer required')
        features=torch.cat((states,states.new_full((len(states),1),layer/self.horizon),context),-1)
        h=(self.backward_net if backward else self.forward_net)(features)
        mean,log_std=(self.backward_head if backward else self.forward_head)(h).chunk(2,-1)
        stop=self.stop_head(h).squeeze(-1)
        for value in (mean,log_std,stop):_finite(value)
        return mean,log_std,stop

    def log_partition(self,context):
        self._context(context)
        if self._mup_configuration is not None:
            spec=self._mup_configuration
            context=context*(spec['context_mult']/(self.context_width/spec['base_context_width']))
        z=self.normalizer(context).squeeze(-1);_finite(z)
        return z


def conditional_trajectory_balance(log_z,log_forward,log_backward,log_stop,log_reward,mask):
    """Per-condition log Z with unchanged mixed-measure TB semantics.

    Deliberate row-wise correctness reference, not an optimized training kernel.
    All paths for each condition must use that same context in both policies.
    Caller owns fixed rewards, detached sampling and forced boundary masses.
    """
    _finite(log_z);_finite(log_forward)
    if (log_forward.ndim!=2 or log_z.shape!=(log_forward.shape[0],)
            or log_z.dtype!=log_forward.dtype or log_z.device!=log_forward.device):
        raise ValueError('one matching log partition per path required')
    for value,shape in ((log_backward,log_forward.shape),(log_stop,log_z.shape),
                        (log_reward,log_z.shape)):
        _finite(value)
        if value.shape!=shape or value.device!=log_z.device or value.dtype!=log_z.dtype:
            raise ValueError('complete matching density and stop/reward arrays required')
    if (not isinstance(mask,torch.Tensor) or mask.shape!=log_forward.shape
            or mask.device!=log_z.device or mask.dtype!=torch.bool):
        raise ValueError('matching boolean prefix mask required')
    rows=[trajectory_balance(log_z[i],log_forward[i:i+1],log_backward[i:i+1],
        log_stop[i:i+1],log_reward[i:i+1],mask[i:i+1]) for i in range(len(log_z))]
    return dict(residual=torch.cat([r['residual'] for r in rows]),
                loss=torch.stack([r['loss'] for r in rows]).mean())
