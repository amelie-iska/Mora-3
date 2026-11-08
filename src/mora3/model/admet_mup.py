"""Exact source-partition ADMET readout adaptation, not source admission."""
from .admet_model import SharedAdmetModel
from .annotation_mup import annotation_mup_parameter_roles
from .parameterization import ParameterRole,parameter_ledger,admet_parameter_ledger,width_family
from .morphology_mup import _without_member_dimensions
from .mixed_payload_projection import exact_wire
from .mu_optimizer import _storage_signature


def admet_mup_parameter_roles(model,*,identifier_axis):
    if type(model)is not SharedAdmetModel or model.parameterization_policy!='mup-admet-v1':
        raise ValueError('exact applied endpoint/assay/class/native-unit ADMET owner required')
    model.validate_parameterization_configuration()
    roles={'shared.'+n:r for n,r in annotation_mup_parameter_roles(model.shared,identifier_axis=identifier_axis).items()}
    ledger=admet_parameter_ledger(model,identifier_axis=identifier_axis)
    for entry in ledger.parameters:
        for name in entry.aliases:
            if not name.startswith('shared.'):roles[name]=ParameterRole(entry.role,entry.axes)
    _storage_signature(model);parameter_ledger(model,roles)
    return roles


def admet_mup_width_family(base,delta,target,*,identifier_axis):
    ms=(base,delta,target);ledgers=[parameter_ledger(m,admet_mup_parameter_roles(m,identifier_axis=identifier_axis)) for m in ms]
    expected=_without_member_dimensions(base.get_extra_state())
    for m in ms:
        if not exact_wire(_without_member_dimensions(m.get_extra_state()),expected):raise ValueError('ADMET finite partition/source/base/policy drift')
    if base.shared.payload.embedding.embedding_dim!=base._mup_configuration['base_width']:
        raise ValueError('actual ADMET base width must equal declared base')
    return width_family(*ledgers)
