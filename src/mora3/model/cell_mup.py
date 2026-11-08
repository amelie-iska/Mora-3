"""Exact full source gene-axis sparse input/readout μP operator roles."""
import hashlib
import json
from .cell_native import SharedNativeCellModel,GENE_COUNT
from .annotation_mup import annotation_mup_parameter_roles
from .parameterization import Axis,ParameterRole,parameter_ledger,width_family
from .full_owner_mup import _without_member_dimensions
from .mixed_payload_projection import exact_wire
from .mu_optimizer import _storage_signature


def cell_mup_parameter_roles(model,*,identifier_axis):
    if type(model)is not SharedNativeCellModel:raise ValueError('exact applied source-native cell owner required')
    model.get_extra_state()
    roles={'shared.'+n:r for n,r in annotation_mup_parameter_roles(model.shared,identifier_axis=identifier_axis).items()}
    axis=Axis('finite',hashlib.sha256(json.dumps(['scCello-native-gene-token-axis-v1',model.vocabulary.source_sha256,
        model.vocabulary.gene_keys[2:]],separators=(',',':')).encode()).hexdigest())
    h=Axis('width','hidden')
    cls=Axis('finite',hashlib.sha256(b'scCello-one-source-CLS-25426-role-not-a-gene-v1').hexdigest())
    roles.update({'value_input.weight':ParameterRole('input',(h,axis)),
        'value_input.bias':ParameterRole('vector',(h,)),
        'availability_input.weight':ParameterRole('input',(h,axis)),
        'source_cls.weight':ParameterRole('embedding',(cls,h)),
        'value_prediction.weight':ParameterRole('output',(axis,h)),
        'value_prediction.bias':ParameterRole('vector',(axis,))})
    _storage_signature(model);parameter_ledger(model,roles)
    return roles


def cell_mup_width_family(base,delta,target,*,identifier_axis):
    ms=(base,delta,target)
    ledgers=[parameter_ledger(m,cell_mup_parameter_roles(m,identifier_axis=identifier_axis)) for m in ms]
    expected=_without_member_dimensions(base.get_extra_state())
    for m in ms:
        if not exact_wire(_without_member_dimensions(m.get_extra_state()),expected):
            raise ValueError('cell full gene/source/recipe/base/availability/CLS policy drift')
    if base.shared.payload.embedding.embedding_dim!=base._configuration['base_width']:
        raise ValueError('actual cell base width must equal declaration')
    return width_family(*ledgers)
