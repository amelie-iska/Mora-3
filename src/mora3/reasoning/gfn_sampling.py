"""Independent layered Euclidean GFN policies and forward rollout.

Adaptation of arXiv:2301.12594v1 §4.2: small concatenated-time MLPs,
learned diagonal scales and optional intermediate exits. Not the paper's PIS
architecture or fixed Brownian backward policy. Sampling detaches states;
score_paths recomputes differentiable densities for score-function TB training.
"""
import torch
from torch import nn
from .continuous_gfn import _finite, diagonal_gaussian_log_density, stop_log_probabilities


class LayeredGaussianPolicy(nn.Module):
    def __init__(self, dimension, horizon, hidden=64):
        super().__init__()
        if any(type(v) is not int or v<=0 for v in (dimension,horizon,hidden)):
            raise ValueError('Require positive integer dimensions/horizon')
        self.dimension=dimension
        self.horizon=horizon
        self.forward_net=nn.Sequential(nn.Linear(dimension+1,hidden),nn.Tanh())
        self.backward_net=nn.Sequential(nn.Linear(dimension+1,hidden),nn.Tanh())
        self.forward_head=nn.Linear(hidden,2*dimension)
        self.backward_head=nn.Linear(hidden,2*dimension)
        self.stop_head=nn.Linear(hidden,1)
        self.log_z=nn.Parameter(torch.tensor(0.))

    def parameters_at(self, states, layer, backward=False):
        _finite(states)
        if states.ndim!=2 or states.shape[1]!=self.dimension:
            raise ValueError('Require [batch,dimension] states')
        if type(layer) is not int or not 0<=layer<=self.horizon:
            raise ValueError('Invalid time layer')
        if states.dtype!=self.log_z.dtype or states.device!=self.log_z.device:
            raise ValueError('Policy/state dtype and device must match')
        clock=states.new_full((len(states),1),layer/self.horizon)
        features=torch.cat([states,clock],-1)
        h=(self.backward_net if backward else self.forward_net)(features)
        mean,log_std=(self.backward_head if backward else self.forward_head)(h).chunk(2,-1)
        return mean,log_std,self.stop_head(h).squeeze(-1)


@torch.no_grad()
def sample_paths(policy, batch_size, *, generator=None, optional_stop=True, context=None):
    """Sample complete paths, forced origin continuation and horizon stopping.

    No exploration temperature, clipping, reparameterization gradient or hidden
    RNG is used. Supply a same-device Generator for isolated reproducibility.
    Stops occur after arrival at layers 1..T; sink is implicit. Padding is zero.
    Optional context has one row per path, selected with the active indices;
    conditional sampling requires an explicit same-device Generator.
    """
    if type(batch_size) is not int or batch_size<=0 or type(optional_stop) is not bool:
        raise ValueError('Require positive batch size and bool optional_stop')
    b,t,d=batch_size,policy.horizon,policy.dimension
    if context is not None:
        policy._context(context)
        if len(context)!=b:raise ValueError("One context per sampled path required")
        if not isinstance(generator,torch.Generator):
            raise ValueError("Conditional sampling requires an isolated same-device Generator")
        generator_device=generator.device
        if generator_device.type=="cuda" and generator_device.index is None:
            generator_device=torch.device("cuda",torch.cuda.current_device())
        if generator_device!=context.device:
            raise ValueError("Conditional sampling requires an isolated same-device Generator")
    template=policy.log_z if context is None else context
    states=template.new_zeros(b,t,d)
    lengths=torch.full((b,),t,device=states.device,dtype=torch.long)
    active=torch.ones(b,dtype=torch.bool,device=states.device)
    previous=states.new_zeros(b,d)
    for layer in range(1,t+1):
        idx=active.nonzero().flatten()
        if not len(idx):break
        mean,ls,_=_parameters_at(policy,previous[idx],layer-1,context,idx)
        scale=ls.exp()
        if not torch.isfinite(scale).all() or (scale<=0).any():
            raise ValueError('Unrepresentable Gaussian scale')
        value=mean+scale*torch.randn(mean.shape,dtype=mean.dtype,device=mean.device,generator=generator)
        _finite(value)
        states[idx,layer-1]=value
        previous[idx]=value
        if optional_stop and layer<t:
            _,_,logits=_parameters_at(policy,value,layer,context,idx)
            _finite(logits)
            stopped=torch.rand(logits.shape,dtype=logits.dtype,device=logits.device,generator=generator)<logits.sigmoid()
            done=idx[stopped];lengths[done]=layer;active[done]=False
    mask=torch.arange(t,device=states.device)[None,:]<lengths[:,None]
    return dict(states=states,lengths=lengths,mask=mask,optional_stop=optional_stop)


def score_paths(policy, paths, *, context=None):
    """Recompute normalized forward/backward densities on detached sampled paths.

    Returns keyword arguments for trajectory_balance; caller owns fixed reward.
    Forward continuation mass is included once before each noninitial move.
    Reverse origin is a Dirac mass; no reverse sink factor. Padded scores zero.
    Caller supplies the same condition rows used for sampling, with gradients
    enabled for the shared encoder. No context/target is stored in paths.
    """
    x=paths['states'];lengths=paths['lengths'];mask=paths['mask'];optional=paths['optional_stop']
    _finite(x)
    if x.ndim!=3 or x.shape[1:]!=(policy.horizon,policy.dimension) or type(optional) is not bool:
        raise ValueError('Invalid path shape or stop mode')
    b,t,d=x.shape
    if b==0:raise ValueError("Nonempty paths required")
    if context is not None:
        policy._context(context)
        if len(context)!=b:raise ValueError("One context per scored path required")
    if (lengths.shape!=(b,) or lengths.dtype!=torch.long or lengths.device!=x.device
        or (lengths<1).any() or (lengths>t).any()
        or (not optional and (lengths!=t).any())):
        raise ValueError('Invalid lengths')
    expected=torch.arange(t,device=x.device)[None,:]<lengths[:,None]
    if mask.dtype!=torch.bool or mask.device!=x.device or mask.shape!=(b,t) or not torch.equal(mask,expected):
        raise ValueError('Invalid prefix mask')
    if x.requires_grad:raise ValueError('Sampled states must be detached for score-function training')
    pf=x.new_zeros(b,t);pb=x.new_zeros(b,t);stop=x.new_zeros(b)
    for layer in range(1,t+1):
        idx=mask[:,layer-1].nonzero().flatten()
        if not len(idx):continue
        current=x[idx,layer-1]
        previous=x[idx,layer-2] if layer>1 else torch.zeros_like(current)
        mean,ls,logits=_parameters_at(policy,previous,layer-1,context,idx)
        forward=diagonal_gaussian_log_density(current,mean,ls)
        if optional and layer>1:forward=forward+stop_log_probabilities(logits)[1]
        pf[idx,layer-1]=forward
        if layer>1:
            mean,ls,_=_parameters_at(policy,current,layer,context,idx,backward=True)
            pb[idx,layer-1]=diagonal_gaussian_log_density(previous,mean,ls)
        if optional and layer<t:
            terminal=lengths[idx]==layer
            if terminal.any():
                _,_,logits=_parameters_at(policy,current[terminal],layer,context,idx[terminal])
                stop[idx[terminal]]=stop_log_probabilities(logits)[0]
    return dict(log_forward=pf,log_backward=pb,log_stop=stop)


def _parameters_at(policy, states, layer, context, indices, *, backward=False):
    """Select identical condition rows for both kernels and terminal mass."""
    if context is None:
        return policy.parameters_at(states,layer,backward=backward)
    return policy.parameters_at(states,layer,context[indices],backward=backward)
