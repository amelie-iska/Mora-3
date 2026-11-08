"""Exact selected RGB semantic/slot/context width rules, not source admission."""
from torch import nn
from .color_semantic import ColorNuclearSemantic,ColorInstancePredictor
from .color_conditioning import ConditionedColorSemantic,ConditionedColorInstance
from .parameterization import (ParameterRole,parameter_ledger,width_family,
    semantic_parameter_ledger,instance_parameter_ledger)
from .morphology_mup import pixel_mup_parameter_roles,_native_roles,_geometry_roles
from .mixed_payload_projection import exact_wire
from .mu_optimizer import _storage_signature


def color_mup_parameter_roles(model):
    if type(model) not in (ColorNuclearSemantic,ColorInstancePredictor) or model.parameterization_policy!='mup-color-v1':
        raise ValueError('exact applied source RGB semantic/instance owner required')
    model.get_extra_state()
    if set(model._modules)!=({'pixels','backbone','head','count'}if type(model)is ColorInstancePredictor and model.head.slot_policy=='source-layer-list-v4'else{'pixels','backbone','head'}):raise ValueError('RGB native ownership drift')
    pixels=pixel_mup_parameter_roles(model.pixels,channel_axis=('R','G','B'))
    native=_native_roles(model.backbone,('image',))
    builder=semantic_parameter_ledger if type(model) is ColorNuclearSemantic else instance_parameter_ledger
    ledger=builder(model,channel_axis=('R','G','B'),identifier_axis=('image',))
    roles={}
    for entry in ledger.parameters:
        for n in entry.aliases:
            if n.startswith('pixels.'):r=pixels[n[len('pixels.'):]]
            elif n.startswith('backbone.'):r=native[n[len('backbone.'):]]
            elif n in ('head.weight','head.projection.weight','count.weight'):r=ParameterRole('output',entry.axes)
            elif n in ('head.bias','head.projection.bias','count.bias'):r=ParameterRole('vector',entry.axes)
            else:raise ValueError('unreviewed RGB operator')
            roles[n]=r
    _storage_signature(model);parameter_ledger(model,roles)
    return roles


def conditioned_color_mup_parameter_roles(model,*,identifier_axis):
    if type(model) not in (ConditionedColorSemantic,ConditionedColorInstance) or model.parameterization_policy!='mup-color-context-v1':
        raise ValueError('exact applied conditioned RGB owner required')
    model.get_extra_state()
    if set(model._modules)!={'color','context','offset'}:raise ValueError('conditioned RGB ownership drift')
    color=color_mup_parameter_roles(model.color)
    context=_geometry_roles(model.context,identifier_axis=identifier_axis)
    classes=1 if type(model) is ConditionedColorSemantic else model.color.head.slots
    if (type(model.offset) is not nn.Linear or model.offset._modules
            or model.offset.in_features!=model.color.width or model.offset.out_features!=classes
            or tuple(model.offset.weight.shape)!=(classes,model.color.width)
            or tuple(model.offset.bias.shape)!=(classes,)):
        raise ValueError('source RGB offset readout operator drift')
    bias='head.bias' if type(model) is ConditionedColorSemantic else 'head.projection.bias'
    output=color[bias].axes[0]
    from .parameterization import Axis
    roles={'color.'+n:r for n,r in color.items()}
    roles.update({'context.'+n:r for n,r in context.items()})
    roles.update({'offset.weight':ParameterRole('output',(output,Axis('width','hidden'))),
                  'offset.bias':ParameterRole('vector',(output,))})
    _storage_signature(model);parameter_ledger(model,roles)
    return roles


def _without_dimensions(value):
    if type(value) is dict:
        return {k:_without_dimensions(v) for k,v in value.items() if k not in ('width','feature_width','ffn_width','channels')}
    if type(value) is tuple:return tuple(_without_dimensions(v) for v in value)
    if type(value) is list:return [_without_dimensions(v) for v in value]
    return value


def _family(models,builder,**kwargs):
    if any(type(m) is not type(models[0]) for m in models):raise ValueError('one exact source RGB class across widths required')
    ledgers=[parameter_ledger(m,builder(m,**kwargs)) for m in models]
    expected=None
    for m in models:
        cfg=_without_dimensions(m.get_extra_state())
        if expected is None:expected=cfg
        if not exact_wire(cfg,expected):raise ValueError('RGB finite source axes/base/readout/operator policy drift')
    base=models[0].color if type(models[0]) in (ConditionedColorSemantic,ConditionedColorInstance) else models[0]
    if base.width!=base.base_width:raise ValueError('actual RGB base must equal declared base width')
    return width_family(*ledgers)


def color_mup_width_family(base,delta,target):
    return _family((base,delta,target),color_mup_parameter_roles)


def conditioned_color_mup_width_family(base,delta,target,*,identifier_axis):
    return _family((base,delta,target),conditioned_color_mup_parameter_roles,identifier_axis=identifier_axis)
