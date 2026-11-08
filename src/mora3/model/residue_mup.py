"""Versioned native residue fixed-kernel convolution adaptation, not data admission."""
import copy
from .residue_objective import SharedResidueModel
from .annotation_mup import annotation_mup_parameter_roles
from .parameterization import ParameterRole,parameter_ledger,residue_head_parameter_roles,width_family
from .mixed_payload_projection import exact_wire
from .mu_optimizer import _storage_signature


def residue_mup_parameter_roles(model,*,identifier_axis):
    if type(model) is not SharedResidueModel or model.parameterization_policy!='mup-residue-v1':
        raise ValueError('exact explicitly parameterized shared residue owner required')
    model.get_extra_state()
    if set(model._modules)!={'shared','head'}:raise ValueError('residue module ownership drift')
    roles={'shared.'+n:r for n,r in annotation_mup_parameter_roles(model.shared,identifier_axis=identifier_axis).items()}
    local=residue_head_parameter_roles(model.head)
    for n,r in local.items():
        if r.role=='custom':
            if n!='local.weight':raise ValueError('unreviewed residue custom operator')
            r=ParameterRole('hidden',r.axes)
        roles['head.'+n]=r
    _storage_signature(model);parameter_ledger(model,roles)
    return roles


def residue_mup_width_family(base,delta,target,*,identifier_axis):
    models=(base,delta,target)
    ledgers=[parameter_ledger(m,residue_mup_parameter_roles(m,identifier_axis=identifier_axis)) for m in models]
    from .annotation_mup import annotation_mup_width_family
    annotation_mup_width_family(*(m.shared for m in models),identifier_axis=identifier_axis)
    expected=None
    for m in models:
        cfg=copy.deepcopy(m.head.get_extra_state());cfg.pop('width')
        if expected is None:expected=cfg
        if not exact_wire(cfg,expected):raise ValueError('residue finite axis/base/operator policy drift')
    return width_family(*ledgers)
