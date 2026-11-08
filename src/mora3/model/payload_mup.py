"""Exact source-bound split payload width rules, not source admission."""
import hashlib

from torch import nn
from .payload import PayloadEncoder
from .mixed_payload_projection import MixedPayloadProjection,exact_wire
from .parameterization import Axis,ParameterRole,parameter_ledger,width_family
from .mu_optimizer import _storage_signature


def payload_mup_parameter_roles(model):
    from .annotation_vocabulary import AnnotationPayloadEncoder,AnnotationVocabulary
    if type(model) not in (PayloadEncoder,AnnotationPayloadEncoder):
        raise ValueError('exact reviewed payload owner required')
    spec=model._payload_mup_configuration()
    modules=dict(model.named_modules(remove_duplicate=False))
    if (set(modules)!={'','embedding','projection'} or type(model.embedding) is not nn.Embedding
            or type(model.projection) is not MixedPayloadProjection
            or model._parameters or any(m._buffers for m in modules.values())):
        raise ValueError('complete reviewed payload module ownership required')
    if type(model) is AnnotationPayloadEncoder:
        source=AnnotationVocabulary(model._vocabulary_json).tokenizer();live=model.codec.tokenizer
        if source.vocabulary!=live.vocabulary or source._controls!=live._controls:
            raise ValueError('bound annotation vocabulary order mismatch')
    hidden=Axis('width','hidden')
    length=Axis('finite',hashlib.sha256(b'payload.log1p-complete-token-count.v1').hexdigest())
    roles={'embedding.weight':ParameterRole('embedding',(Axis('finite',spec['token_axis_sha256']),hidden)),
        'projection.hidden_weight':ParameterRole('hidden',(hidden,hidden)),
        'projection.length_weight':ParameterRole('input',(hidden,length)),
        'projection.bias':ParameterRole('vector',(hidden,))}
    ledger=parameter_ledger(model,roles)
    if any(len(entry.aliases)!=1 for entry in ledger.parameters):
        raise ValueError('unreviewed payload parameter ties')
    _storage_signature(model)
    return roles


def payload_mup_width_family(base,delta,target):
    models=(base,delta,target)
    if any(type(m) is not type(base) for m in models):
        raise ValueError('same payload owner class across widths required')
    ledgers=[parameter_ledger(m,payload_mup_parameter_roles(m)) for m in models]
    expected=None
    for model in models:
        spec=model._payload_mup_configuration();spec['projection'].pop('width')
        if expected is None:expected=spec
        if not exact_wire(spec,expected):
            raise ValueError('payload finite-axis/base/pooling/token-bound policy drift')
    if base.embedding.embedding_dim!=base.projection.base_width:
        raise ValueError('actual payload base owner must match declared base width')
    return width_family(*ledgers)
