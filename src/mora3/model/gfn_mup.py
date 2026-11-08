"""Exact profile GFN fixed-hidden source adaptation, not reward/data admission."""
import hashlib
import json
from .profile_gfn import ProfileGFNPolicy
from .fixed_context_projection import fixed_context_parameter_roles
from .sequence_mup import profile_mup_parameter_roles,profile_mup_width_family
from .parameterization import Axis,ParameterRole,parameter_ledger,width_family
from .mixed_payload_projection import exact_wire
from .mu_optimizer import _storage_signature


def gfn_mup_parameter_roles(model,*,identifier_axis):
    if type(model) is not ProfileGFNPolicy or model.parameterization_policy!='mup-profile-gfn-v1':
        raise ValueError('exact applied native profile GFN source owner required')
    model.validate_parameterization_configuration()
    roles={'profiles.'+n:r for n,r in profile_mup_parameter_roles(model.profiles,identifier_axis=identifier_axis).items()}
    for direction in ('forward','backward'):
        roles.update({direction+'_net.0.'+n:r for n,r in fixed_context_parameter_roles(getattr(model,direction+'_net')[0]).items()})
    def finite(namespace,axis):
        return Axis('finite',hashlib.sha256(json.dumps([namespace,axis],separators=(',',':')).encode()).hexdigest())
    h=fixed_context_parameter_roles(model.forward_net[0])['bias'].axes[0]
    if h!=fixed_context_parameter_roles(model.backward_net[0])['bias'].axes[0]:
        raise ValueError('source GFN forward/backward fixed hidden coordinate mismatch')
    gaussian=finite('GFN-mean-then-logstd-coordinate-v1',tuple(('mean',i) for i in range(model.dimension))+tuple(('logstd',i) for i in range(model.dimension)))
    stop=finite('GFN-stop-logit-v1',('stop',));logz=finite('GFN-conditional-logZ-v1',('logZ',))
    for name,output,inputs in (('forward_head',gaussian,h),('backward_head',gaussian,h),('stop_head',stop,h),
                               ('normalizer',logz,Axis('width','hidden'))):
        roles[name+'.weight']=ParameterRole('output',(output,inputs))
        roles[name+'.bias']=ParameterRole('vector',(output,))
    _storage_signature(model);parameter_ledger(model,roles)
    return roles


def gfn_mup_width_family(base,delta,target,*,identifier_axis):
    models=(base,delta,target)
    ledgers=[parameter_ledger(m,gfn_mup_parameter_roles(m,identifier_axis=identifier_axis)) for m in models]
    profile_mup_width_family(*(m.profiles for m in models),identifier_axis=identifier_axis)
    def normalized(value):
        if type(value) is dict:return {k:normalized(v) for k,v in value.items() if k not in ('width','feature_width','ffn_width','context_width')}
        if type(value) is tuple:return tuple(normalized(v) for v in value)
        if type(value) is list:return [normalized(v) for v in value]
        return value
    expected=None
    for model in models:
        config=normalized(model.get_extra_state())
        if expected is None:expected=config
        if not exact_wire(config,expected):raise ValueError('GFN finite hidden/latent/source/base/policy drift')
    return width_family(*ledgers)
