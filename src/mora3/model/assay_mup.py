"""Exact native Tox21/LINCS source-axis readout width families."""
from .tox21 import SharedTox21Model
from .lincs_model import SharedLincsModel
from .parameterization import (ParameterRole,parameter_ledger,width_family,
    tox21_parameter_ledger,lincs_parameter_ledger)
from .annotation_mup import annotation_mup_parameter_roles,annotation_mup_width_family
from .morphology_mup import _without_member_dimensions
from .mixed_payload_projection import exact_wire
from .mu_optimizer import _storage_signature


def assay_mup_parameter_roles(model,*,identifier_axis):
    if type(model) not in (SharedTox21Model,SharedLincsModel):raise ValueError('exact native assay owner required')
    model.validate_parameterization_configuration()
    native=annotation_mup_parameter_roles(model.shared,identifier_axis=identifier_axis)
    builder=tox21_parameter_ledger if type(model) is SharedTox21Model else lincs_parameter_ledger
    ledger=builder(model,identifier_axis=identifier_axis)
    roles={}
    for entry in ledger.parameters:
        for name in entry.aliases:
            roles[name]=(native[name[len('shared.'):]] if name.startswith('shared.') else ParameterRole(entry.role,entry.axes))
    _storage_signature(model);parameter_ledger(model,roles)
    return roles


def assay_mup_width_family(base,delta,target,*,identifier_axis):
    models=(base,delta,target)
    if any(type(m) is not type(base) for m in models):raise ValueError('same exact native assay owner class required')
    ledgers=[parameter_ledger(m,assay_mup_parameter_roles(m,identifier_axis=identifier_axis)) for m in models]
    annotation_mup_width_family(*(m.shared for m in models),identifier_axis=identifier_axis)
    expected=None
    for model in models:
        config=_without_member_dimensions(model.get_extra_state())
        if expected is None:expected=config
        if not exact_wire(config,expected):raise ValueError('native assay source/axis/base/readout policy drift')
    return width_family(*ledgers)
