"""Exact finite GOA readout width rules; source pool admission stays external."""
import hashlib
import json
from torch import nn
from .uniprot_go import NativeGOAHead
from .parameterization import Axis,ParameterRole,parameter_ledger,width_family
from .mixed_payload_projection import exact_wire
from .mu_optimizer import _storage_signature


def goa_mup_parameter_roles(model):
    if type(model) is not NativeGOAHead or getattr(model,'_goa_mup',None) is None:
        raise ValueError('exact explicitly applied native GOA head required')
    model._validate_goa_parameterization()
    modules=dict(model.named_modules(remove_duplicate=False))
    if set(modules)!={'','output'} or type(modules['output']) is not nn.Linear or any(m._buffers for m in modules.values()):
        raise ValueError('native GOA output ownership drift')
    axis=Axis('finite',hashlib.sha256(json.dumps(['native-GOA-ordered-output-axis.v1',model.axis_ids,model.policy],
        ensure_ascii=True,separators=(',',':')).encode()).hexdigest())
    roles={'output.weight':ParameterRole('output',(axis,Axis('width','hidden'))),
           'output.bias':ParameterRole('vector',(axis,))}
    _storage_signature(model);parameter_ledger(model,roles)
    return roles


def goa_mup_width_family(base,delta,target):
    models=(base,delta,target)
    ledgers=[parameter_ledger(m,goa_mup_parameter_roles(m)) for m in models]
    expected=None
    for model in models:
        cfg=model.get_extra_state();cfg['mup'].pop('width')
        if expected is None:expected=cfg
        if not exact_wire(cfg,expected):raise ValueError('GOA source pool/axis/base/readout policy drift')
    if base._goa_mup['width']!=base._goa_mup['base_width']:
        raise ValueError('actual GOA base must equal declared base width')
    return width_family(*ledgers)
