"""Source-bound fixed-kernel image and native geometry μP owner adaptations.

Spatial reuse, pooling and interpolation are explicit finite operations; they do
not acquire a width axis. Concatenation coefficients stay fixed across members.
This qualifies named active operators, not corpus admission or image quality.
"""
import copy
from torch import nn
from .morphology import MultiscalePixels,MorphologyInstance
from .cooperative_training import CooperativeModel,GeometryCooperativeModel
from .geometry_conditioning import GeometryConditioner
from .parameterization import (ParameterRole,Axis,parameter_ledger,width_family,
    image_encoder_parameter_roles,_morphology_parameter_ledger,tokengt_parameter_ledger)
from .sciplex_mup import sciplex_mup_parameter_roles
from .payload_mup import payload_mup_parameter_roles
from .mixed_payload_projection import exact_wire
from .mu_optimizer import _storage_signature


def pixel_mup_parameter_roles(model,*,channel_axis):
    if type(model) is not MultiscalePixels or model.parameterization_policy!='mup-image-v1':
        raise ValueError('exact applied multiscale fixed-kernel image encoder required')
    cfg=model.parameterization_configuration()
    if cfg['input_channels']!=len(channel_axis):raise ValueError('complete ordered image input channels required')
    roles=image_encoder_parameter_roles(model,channel_axis=channel_axis)
    return {n:ParameterRole('input' if n=='enc0.0.weight' else 'hidden',r.axes)
            if r.role=='custom' else r for n,r in roles.items()}


def _native_roles(graph,identifier_axis):
    cfg=graph.configuration()
    if cfg['schema']!='tokengt-native-configuration-v3':
        raise ValueError('explicit applied native image graph required')
    w=cfg['width'];h=Axis('width','hidden')
    ledger=tokengt_parameter_ledger(graph,feature_axis=tuple('hidden-coordinate:'+str(i) for i in range(w)),
                                  identifier_axis=identifier_axis)
    roles={}
    packed={f'layers.{i}.attention.qkv.weight' for i in range(len(graph.layers))}
    for entry in ledger.parameters:
        for n in entry.aliases:
            role=entry.role;axes=entry.axes
            if n=='feature.weight':role='hidden';axes=(h,h)
            elif n=='incidence.weight':role='input'
            elif n in packed:role='hidden'
            elif role=='custom':raise ValueError('unreviewed native image graph operator')
            roles[n]=ParameterRole(role,axes)
    return roles


def morphology_mup_parameter_roles(model,*,identifier_axis):
    if type(model) is not MorphologyInstance or model.parameterization_policy!='mup-image-v1':
        raise ValueError('exact applied native morphology instance owner required')
    model.get_extra_state()
    local=pixel_mup_parameter_roles(model.pixels,channel_axis=('intensity',))
    ledger=_morphology_parameter_ledger(model,identifier_axis=identifier_axis)
    roles={}
    native=_native_roles(model.backbone,identifier_axis)
    for entry in ledger.parameters:
        for n in entry.aliases:
            if n.startswith('pixels.'):r=local[n[len('pixels.'):]]
            elif n.startswith('backbone.'):r=native[n[len('backbone.'):]]
            elif n=='head.weight':r=ParameterRole('output',entry.axes)
            elif n=='head.bias':r=ParameterRole(entry.role,entry.axes)
            else:raise ValueError('unreviewed native morphology operator')
            roles[n]=r
    _storage_signature(model);parameter_ledger(model,roles)
    return roles


def _geometry_roles(model,*,identifier_axis):
    if type(model) is not GeometryConditioner or model.parameterization_policy!='mup-geometry-v1':
        raise ValueError('exact applied geometry conditioner required')
    model.get_extra_state()
    if set(model._modules)!={'payload','graph'}:raise ValueError('geometry ownership drift')
    roles={'payload.'+n:r for n,r in payload_mup_parameter_roles(model.payload).items()}
    roles.update({'graph.'+n:r for n,r in _native_roles(model.graph,identifier_axis).items()})
    parameter_ledger(model,roles)
    return roles


def cooperative_mup_parameter_roles(model,*,geometry_identifier_axis=None):
    if type(model) not in (CooperativeModel,GeometryCooperativeModel):
        raise ValueError('exact native grayscale cooperative owner required')
    model.validate_parameterization_configuration()
    geometry=type(model) is GeometryCooperativeModel
    if set(model._modules)!=({'response','image','fusion'} if geometry else {'response','image'}):
        raise ValueError('native cooperative owner module drift')
    response=sciplex_mup_parameter_roles(model.response)
    image=morphology_mup_parameter_roles(model.image,identifier_axis=('context','observation'))
    roles={'response.'+n:r for n,r in response.items()}
    roles.update({'image.'+n:r for n,r in image.items()})
    if geometry:
        if (model.fusion.image is not model.image or type(geometry_identifier_axis) is not tuple
                or not geometry_identifier_axis):raise ValueError('complete ordered geometry identifier axis required')
        model.fusion.get_extra_state()
        roles.update({'fusion.image.'+n:r for n,r in image.items()})
        roles.update({'fusion.geometry.'+n:r for n,r in _geometry_roles(model.fusion.geometry,identifier_axis=geometry_identifier_axis).items()})
    elif geometry_identifier_axis is not None:
        raise ValueError('geometry axis supplied without active geometry owner')
    _storage_signature(model);parameter_ledger(model,roles)
    return roles


def _without_member_dimensions(value):
    if type(value) is dict:
        return {k:_without_member_dimensions(v) for k,v in value.items() if k not in ('width','feature_width','ffn_width')}
    if type(value) is tuple:return tuple(_without_member_dimensions(v) for v in value)
    if type(value) is list:return [_without_member_dimensions(v) for v in value]
    return value


def _family(models,role_builder,**kwargs):
    if any(type(m) is not type(models[0]) for m in models):raise ValueError('one exact image owner class across widths required')
    ledgers=[parameter_ledger(m,role_builder(m,**kwargs)) for m in models]
    expected=None
    for m in models:
        cfg=_without_member_dimensions(m.get_extra_state())
        if expected is None:expected=cfg
        if not exact_wire(cfg,expected):raise ValueError('image finite source axes/base/operator/depth configuration drift')
    base=models[0]
    if type(base) is MorphologyInstance:
        actual,declared=base.width,base.base_width
    else:actual,declared=base.response.config['width'],base.response.config['mup']['base_width']
    if actual!=declared:raise ValueError('actual image base must equal declared base width')
    return width_family(*ledgers)


def morphology_mup_width_family(base,delta,target,*,identifier_axis):
    return _family((base,delta,target),morphology_mup_parameter_roles,identifier_axis=identifier_axis)


def cooperative_mup_width_family(base,delta,target,*,geometry_identifier_axis=None):
    return _family((base,delta,target),cooperative_mup_parameter_roles,geometry_identifier_axis=geometry_identifier_axis)
