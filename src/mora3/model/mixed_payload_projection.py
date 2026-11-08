"""Explicit μP adaptation of pooled hidden features plus fixed token count.

The source operation is one affine map on [pooled_W, log1p(token_count)].
Splitting its hidden and scalar columns preserves that mathematical operation
and parameter count. It lets the optimizer apply distinct width rules without
pretending W+1 is one homogeneous scalable axis. Legacy modules are untouched.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def exact_wire(left,right):
    if type(left) is not type(right):return False
    if type(left) is dict:
        return left.keys()==right.keys() and all(exact_wire(left[k],right[k]) for k in left)
    if type(left) is tuple:
        return len(left)==len(right) and all(exact_wire(a,b) for a,b in zip(left,right))
    return left==right


def mixed_payload_configuration(width,base_width):
    if (type(width) is not int or type(base_width) is not int
            or base_width<=0 or width<base_width):
        raise ValueError('positive base width not exceeding payload width required')
    return dict(schema='mixed-payload-projection-v1',width=width,base_width=base_width,
        fixed_axis='log1p-complete-token-count',
        hidden_initialization='base-linear-uniform-times-sqrt-base-over-width',
        fixed_initialization='base-linear-uniform',bias_initialization='base-linear-uniform',
        hidden_adam_lr='base-over-width',fixed_adam_lr='one',
        operation='hidden-affine-plus-fixed-token-count-column')


class MixedPayloadProjection(nn.Module):
    def __init__(self,width,*,base_width):
        super().__init__()
        mixed_payload_configuration(width,base_width)
        self.width,self.base_width=width,base_width
        self._declaration=(width,base_width)
        self.hidden_weight=nn.Parameter(torch.empty(width,width))
        self.length_weight=nn.Parameter(torch.empty(width,1))
        self.bias=nn.Parameter(torch.empty(width))
        bound=1/math.sqrt(base_width+1)
        with torch.no_grad():
            self.hidden_weight.uniform_(-bound*math.sqrt(base_width/width),bound*math.sqrt(base_width/width))
            self.length_weight.uniform_(-bound,bound)
            self.bias.uniform_(-bound,bound)

    def get_extra_state(self):
        if (self._declaration!=(self.width,self.base_width)
                or tuple(self.hidden_weight.shape)!=(self.width,self.width)
                or tuple(self.length_weight.shape)!=(self.width,1)
                or tuple(self.bias.shape)!=(self.width,)):
            raise ValueError('mixed payload projection declaration/shape drift')
        return mixed_payload_configuration(self.width,self.base_width)

    def set_extra_state(self,state):
        if not exact_wire(state,self.get_extra_state()):
            raise ValueError('mixed payload projection checkpoint configuration mismatch')

    def validate_checkpoint(self,state_dict,prefix):
        if not exact_wire(state_dict.get(prefix+'_extra_state'),self.get_extra_state()):
            raise ValueError('mixed payload projection checkpoint policy mismatch')

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,
                              missing_keys,unexpected_keys,error_msgs):
        self.validate_checkpoint(state_dict,prefix)
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,
                                     missing_keys,unexpected_keys,error_msgs)

    def forward(self,features):
        self.get_extra_state()
        if features.ndim<1 or features.shape[-1]!=self.width+1:
            raise ValueError('complete hidden vector and one fixed token-count input required')
        return F.linear(features[...,:self.width],self.hidden_weight)+F.linear(
            features[...,self.width:],self.length_weight,self.bias)
