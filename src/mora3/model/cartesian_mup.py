"""Exact shared all-atom Cartesian source adaptation; no physical admission."""
import hashlib
import json
from .cartesian_gfn_policy import SharedCartesianGFNPolicy,FEATURE_NAMES
from .parameterization import Axis,ParameterRole,parameter_ledger,width_family
from .morphology_mup import _native_roles,_without_member_dimensions
from .mixed_payload_projection import exact_wire
from .mu_optimizer import _storage_signature


def _finite(namespace,values):
    return Axis('finite',hashlib.sha256(json.dumps([namespace,values],separators=(',',':')).encode()).hexdigest())


def cartesian_mup_parameter_roles(model):
    if type(model)is not SharedCartesianGFNPolicy:
        raise ValueError('exact applied shared Cartesian policy required')
    model.get_extra_state()
    native=_native_roles(model.backbone,model.backbone_identifier_axis)
    graph=_native_roles(model.graph,tuple('Cartesian-orthogonal-ID-coordinate:'+str(i) for i in range(model.id_width)))
    roles={'backbone.'+n:r for n,r in native.items()}
    roles.update({'graph.'+n:r for n,r in graph.items()})
    h=Axis('width','hidden');features=_finite('Cartesian-complete-source-state-features-v1',FEATURE_NAMES)
    roles['geometry_features.weight']=ParameterRole('input',(h,features))
    gaussian=_finite('Cartesian-peratom-mean-then-logstd-v1',('mean-x','mean-y','mean-z','logstd-x','logstd-y','logstd-z'))
    stop=_finite('Cartesian-global-stop-v1',('stop',));logz=_finite('Cartesian-condition-only-logZ-v1',('logZ',))
    for name,axis in (('forward_head',gaussian),('backward_head',gaussian),('stop_head',stop),('normalizer',logz)):
        roles[name+'.weight']=ParameterRole('output',(axis,h));roles[name+'.bias']=ParameterRole('vector',(axis,))
    _storage_signature(model);parameter_ledger(model,roles)
    return roles


def cartesian_mup_width_family(base,delta,target):
    models=(base,delta,target)
    ledgers=[parameter_ledger(m,cartesian_mup_parameter_roles(m)) for m in models]
    expected=None
    for m in models:
        current=_without_member_dimensions(m.get_extra_state())
        if expected is None:expected=current
        if not exact_wire(current,expected):raise ValueError('Cartesian finite axis/teacher/source/depth/base/resource configuration drift')
    if base.width!=base.base_width:raise ValueError('actual Cartesian base width must match declared base')
    return width_family(*ledgers)
