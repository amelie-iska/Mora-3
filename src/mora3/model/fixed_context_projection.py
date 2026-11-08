"""Explicit fixed-prefix plus learned-context readout to a finite coordinate set.

The original operation is an affine map on [fixed_latent_D,time,context_W].
Only context_W scales. Splitting the columns preserves the mathematical map and
parameter count while avoiding a false homogeneous W+D+1 width declaration.
The output coordinates remain finite (e.g. fixed GFN hidden64), never inferred
as a second scalable hidden dimension from matrix shape.
"""
import copy
import hashlib
import json
import math

import torch
from torch import nn
from torch.nn import functional as F
from .mixed_payload_projection import exact_wire
from .parameterization import Axis,ParameterRole,parameter_ledger,width_family
from .mu_optimizer import _storage_signature


class FixedContextProjection(nn.Module):
    def __init__(self,context_width,fixed_axis,output_axis,*,source_namespace,base_context_width,context_mult=1.0):
        super().__init__()
        axes=(fixed_axis,output_axis)
        if (type(context_width) is not int or type(base_context_width) is not int or base_context_width<1
                or context_width<base_context_width or any(type(a) is not tuple or not a or len(set(a))!=len(a)
                    or any(type(x) is not str or not x for x in a) for a in axes)
                or type(source_namespace) is not str or not source_namespace.strip()
                or type(context_mult) not in (int,float) or not math.isfinite(context_mult) or context_mult<=0):
            raise ValueError('exact finite axes/namespace and compatible context width/multiplier required')
        self.context_width,self.base_context_width=context_width,base_context_width
        self.fixed_axis,self.output_axis,self.source_namespace=fixed_axis,output_axis,source_namespace
        self.context_mult=float(context_mult)
        # Draw once with the native source affine ordering, then split into
        # independently owned parameters; views would violate optimizer storage.
        affine=nn.Linear(len(fixed_axis)+context_width,len(output_axis))
        scale=math.sqrt((len(fixed_axis)+context_width)/(len(fixed_axis)+base_context_width))
        self.fixed_weight=nn.Parameter((affine.weight[:,:len(fixed_axis)].detach()*scale).clone())
        self.context_weight=nn.Parameter((affine.weight[:,len(fixed_axis):].detach()*scale).clone())
        self.bias=nn.Parameter((affine.bias.detach()*scale).clone())
        self._configuration=self._current_configuration()

    def _current_configuration(self):
        return dict(schema='Mora3.FixedContextProjection/v1',context_width=self.context_width,
            base_context_width=self.base_context_width,context_mult=self.context_mult,
            fixed_axis=self.fixed_axis,output_axis=self.output_axis,source_namespace=self.source_namespace,
            layout='fixed-prefix-then-complete-learned-context',
            initialization='native-affine-draw-restored-to-base-total-fanin-before-independent-column-split',
            fixed_scaling='invariant-finite-columns-adamLR1',context_scaling='finite-output-readout-input-divided-by-context-width-ratio',
            context_adam_lr='one',output_coordinates='finite-not-width-scaled')

    def get_extra_state(self):
        if not exact_wire(self._current_configuration(),self._configuration):
            raise ValueError('fixed/context projection configuration drift')
        shapes={'fixed_weight':(len(self.output_axis),len(self.fixed_axis)),
            'context_weight':(len(self.output_axis),self.context_width),'bias':(len(self.output_axis),)}
        params=dict(self.named_parameters(remove_duplicate=False))
        if self._modules or self._buffers or set(params)!=set(shapes) or any(tuple(params[n].shape)!=shape for n,shape in shapes.items()):
            raise ValueError('fixed/context projection parameter ownership/shape drift')
        return copy.deepcopy(self._configuration)

    def set_extra_state(self,state):
        if not exact_wire(state,self.get_extra_state()):raise ValueError('fixed/context projection configuration mismatch')

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        if not exact_wire(state_dict.get(prefix+'_extra_state'),self.get_extra_state()):
            raise ValueError('fixed/context checkpoint source/axis/policy configuration mismatch')
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def forward(self,features):
        self.get_extra_state()
        split=len(self.fixed_axis)
        if (not isinstance(features,torch.Tensor) or features.ndim<1 or features.shape[-1]!=split+self.context_width
                or features.dtype!=self.context_weight.dtype or features.device!=self.context_weight.device):
            raise ValueError('complete fixed-prefix and learned-context features on model dtype/device required')
        context=features[...,split:]*(self.context_mult/(self.context_width/self.base_context_width))
        return F.linear(features[...,:split],self.fixed_weight,self.bias)+F.linear(context,self.context_weight)


def fixed_context_parameter_roles(model):
    if type(model) is not FixedContextProjection:raise ValueError('exact source-reviewed fixed/context projection required')
    model.get_extra_state()
    def finite(name,axis):
        return Axis('finite',hashlib.sha256(json.dumps([model.source_namespace,name,axis],
            ensure_ascii=True,separators=(',',':')).encode()).hexdigest())
    output=finite('output',model.output_axis)
    fixed=finite('fixed-prefix',model.fixed_axis)
    roles={'fixed_weight':ParameterRole('input',(output,fixed)),
        'context_weight':ParameterRole('output',(output,Axis('width','hidden'))),
        'bias':ParameterRole('vector',(output,))}
    _storage_signature(model);parameter_ledger(model,roles)
    return roles


def fixed_context_width_family(base,delta,target):
    models=(base,delta,target)
    ledgers=[parameter_ledger(m,fixed_context_parameter_roles(m)) for m in models]
    expected=None
    for m in models:
        cfg=m.get_extra_state();cfg.pop('context_width')
        if expected is None:expected=cfg
        if not exact_wire(cfg,expected):raise ValueError('fixed/context finite axes/base/multiplier policy drift')
    if base.context_width!=base.base_context_width:raise ValueError('actual context base must equal declared width')
    return width_family(*ledgers)
