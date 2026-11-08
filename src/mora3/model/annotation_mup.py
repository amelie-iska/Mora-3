"""Complete ordinary annotation/physical μP owner families, not corpus admission."""
import copy

from .annotation_multihead import SharedAnnotationModel
from .physical_trainer import SharedPhysicalModel
from .parameterization import (ParameterRole,parameter_ledger,annotation_parameter_ledger,
                              physical_parameter_ledger,width_family)
from .mixed_payload_projection import exact_wire
from .mu_optimizer import _storage_signature


def annotation_mup_parameter_roles(model,*,identifier_axis):
    if type(model) not in (SharedAnnotationModel,SharedPhysicalModel):
        raise ValueError('exact source-reviewed annotation/physical owner required')
    model.validate_parameterization_configuration()
    if model.get_extra_state()['version']!=2:
        raise ValueError('explicit applied annotation parameterization required')
    build=physical_parameter_ledger if type(model) is SharedPhysicalModel else annotation_parameter_ledger
    ledger=build(model,identifier_axis=identifier_axis)
    roles={}
    for entry in ledger.parameters:
        if len(entry.aliases)!=1:raise ValueError('unreviewed annotation internal parameter ties')
        name=entry.aliases[0];role=entry.role
        if role=='custom':
            if name=='backbone.incidence.weight':
                if entry.shape[1]!=2*len(identifier_axis):raise ValueError('complete ordered incidence coordinates required')
                role='input'
            elif name in {f'backbone.layers.{i}.attention.qkv.weight' for i in range(len(model.backbone.layers))}:
                w=model._configuration['width']
                if entry.shape!=(3*w,w):raise ValueError('complete source packedQKV matrix required')
                role='hidden'
            else:raise ValueError('unreviewed active annotation custom operator')
        roles[name]=ParameterRole(role,entry.axes)
    _storage_signature(model)
    parameter_ledger(model,roles)
    return roles


def annotation_mup_width_family(base,delta,target,*,identifier_axis):
    models=(base,delta,target)
    if any(type(m) is not type(base) for m in models):
        raise ValueError('one exact annotation/physical owner class across widths required')
    ledgers=[parameter_ledger(m,annotation_mup_parameter_roles(m,identifier_axis=identifier_axis)) for m in models]
    expected=None
    for model in models:
        cfg=copy.deepcopy(model.get_extra_state());cfg.pop('width')
        for name in ('width','feature_width','ffn_width'):cfg['native_backbone'].pop(name)
        cfg['payload_parameterization']['projection'].pop('width')
        if 'physical' in cfg:cfg['physical'].pop('width')
        if expected is None:expected=cfg
        if not exact_wire(cfg,expected):raise ValueError('annotation finite axis/base/readout/depth/source policy drift')
    if base._configuration['width']!=base._configuration['mup']['base_width']:
        raise ValueError('actual annotation base must equal declared base width')
    return width_family(*ledgers)
