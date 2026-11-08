"""Read-only explicit parameter/alias inventory for future width parameterization.

Axis declarations are caller assertions, not inferred biological identity or a
qualified muP scaling plan. No initialization, optimizer or checkpoint mutation.
"""
from dataclasses import dataclass
import re
from typing import Mapping
from torch import nn


@dataclass(frozen=True)
class Axis:
    kind: str
    identity: str

    def __post_init__(self):
        if self.kind not in ('finite', 'width') or not isinstance(self.identity, str) or not self.identity:
            raise ValueError('explicit finite or width identity required')
        if self.kind == 'finite' and re.fullmatch('[0-9a-f]{64}', self.identity) is None:
            raise ValueError('finite axis requires ordered identity SHA256')


@dataclass(frozen=True)
class ParameterRole:
    role: str
    axes: tuple[Axis, ...]

    def __post_init__(self):
        if self.role not in ('input', 'hidden', 'output', 'embedding',
                             'embedding_readout', 'vector', 'normalization', 'scalar', 'custom'):
            raise ValueError('unsupported parameter role')
        if type(self.axes) is not tuple or not all(isinstance(a, Axis) for a in self.axes):
            raise ValueError('axes must be an immutable tuple of Axis declarations')


@dataclass(frozen=True)
class ParameterEntry:
    aliases: tuple[str, ...]
    shape: tuple[int, ...]
    axes: tuple[Axis, ...]
    role: str
    numel: int
    trainable: bool
    dtype: str


@dataclass(frozen=True)
class ParameterLedger:
    parameters: tuple[ParameterEntry, ...]
    total_numel: int
    trainable_numel: int
    schema: str = 'mora3.parameter-alias-inventory.v1'


def parameter_ledger(model: nn.Module, declarations: Mapping[str, ParameterRole]) -> ParameterLedger:
    """Count identity-tied parameters once and require every alias declaration.

    Distinct Parameter objects are distinct owners, including equal-valued tensors.
    This function does not infer scaling from rank or validate future width families.
    """
    if not isinstance(model, nn.Module) or not isinstance(declarations, Mapping):
        raise ValueError('module and complete declarations required')
    named = tuple(model.named_parameters(remove_duplicate=False))
    if set(declarations) != {name for name, _ in named}:
        raise ValueError('missing or extra parameter declarations')
    groups = {}
    for name, parameter in named:
        declaration = declarations[name]
        if not isinstance(declaration, ParameterRole) or len(declaration.axes) != parameter.ndim:
            raise ValueError('parameter rank or declaration mismatch')
        groups.setdefault(id(parameter), []).append((name, parameter, declaration))
    entries = []
    for aliases in groups.values():
        _, parameter, declaration = aliases[0]
        if any(other != declaration for _, _, other in aliases):
            raise ValueError('conflicting shared parameter alias declarations')
        entries.append(ParameterEntry(
            tuple(sorted(name for name, _, _ in aliases)), tuple(parameter.shape),
            declaration.axes, declaration.role, parameter.numel(), parameter.requires_grad,
            str(parameter.dtype)))
    entries.sort(key=lambda entry: entry.aliases)
    return ParameterLedger(tuple(entries), sum(e.numel for e in entries),
                           sum(e.numel for e in entries if e.trainable))


@dataclass(frozen=True)
class WidthFamily:
    """Validated declarations only; does not qualify operators for muP scaling."""
    base: ParameterLedger
    delta: ParameterLedger
    target: ParameterLedger
    widths: tuple[tuple[str, int, int, int], ...]
    total_numel: tuple[int, int, int]
    schema: str = 'mora3.declared-width-family.v1'


def width_family(base: ParameterLedger, delta: ParameterLedger,
                 target: ParameterLedger) -> WidthFamily:
    """Require fixed ordered finite identities and increasing declared widths.

    No tensor access, initialization, optimizer rule or custom-operator inference.
    Ledgers are caller-supplied inventories, not source or live-model certificates.
    """
    ledgers = (base, delta, target)
    maps = []
    widths = []
    for ledger in ledgers:
        if not isinstance(ledger, ParameterLedger) or ledger.schema != 'mora3.parameter-alias-inventory.v1':
            raise ValueError('supported parameter ledger required')
        mapping = {}
        axis_sizes = {}
        aliases = set()
        for entry in ledger.parameters:
            if entry.role == 'custom':
                raise ValueError('custom operators require a separate reviewed contract')
            if (not entry.aliases or aliases.intersection(entry.aliases)
                    or len(entry.axes) != len(entry.shape)):
                raise ValueError('ambiguous parameter ownership or rank')
            aliases.update(entry.aliases)
            mapping[entry.aliases] = entry
            for axis, size in zip(entry.axes, entry.shape):
                if type(size) is not int or size < 1:
                    raise ValueError('positive declared dimensions required')
                if axis.kind == 'width':
                    if axis.identity in axis_sizes and axis_sizes[axis.identity] != size:
                        raise ValueError('inconsistent named width dimension')
                    axis_sizes[axis.identity] = size
        maps.append(mapping)
        widths.append(axis_sizes)
    if not maps[0] or any(set(m) != set(maps[0]) for m in maps[1:]):
        raise ValueError('exact alias ownership required across widths')
    for aliases, original in maps[0].items():
        for mapping in maps[1:]:
            other = mapping[aliases]
            if (original.axes, original.role, original.trainable, original.dtype) != (
                    other.axes, other.role, other.trainable, other.dtype):
                raise ValueError('axis orientation, role, dtype or trainability drift')
            if any(axis.kind == 'finite' and x != y for axis, x, y in
                   zip(original.axes, original.shape, other.shape)):
                raise ValueError('finite dimension changed')
    if not widths[0] or any(set(w) != set(widths[0]) for w in widths[1:]):
        raise ValueError('explicit shared scalable axes required')
    result = tuple((name, *(w[name] for w in widths)) for name in sorted(widths[0]))
    if any(not (a < b <= c) for _, a, b, c in result):
        raise ValueError('delta must widen base and target must not shrink delta')
    return WidthFamily(base, delta, target, result,
                       tuple(ledger.total_numel for ledger in ledgers))


def tokengt_parameter_ledger(model, *, feature_axis: tuple[str, ...],
                             identifier_axis: tuple[str, ...]) -> ParameterLedger:
    """Explicit role map for the reviewed native TokenGT backbone only.

    Input names are caller declarations, not authenticated source certificates.
    Identifier coordinates are fixed; ordered left/right incidence concatenation,
    packed vanilla QKV and tropical max-plus parameters remain custom. Consequently
    this inventory deliberately cannot qualify a width family for scaling. Shared
    multimodal wrappers/heads need separate maps; no forward or mutation occurs.
    """
    import hashlib
    import json
    from .tokengt import TokenGT, EncoderLayer, DenseAttention
    from .tropical import TropicalCodeAttention

    def finite(namespace, names):
        if (type(names) is not tuple or not names or
                any(type(n) is not str or not n for n in names) or len(set(names)) != len(names)):
            raise ValueError('immutable unique nonempty ordered axis names required')
        payload = json.dumps([namespace, names], ensure_ascii=True, separators=(',', ':'))
        return Axis('finite', hashlib.sha256(payload.encode('utf-8')).hexdigest())

    if type(model) is not TokenGT:
        raise ValueError('exact reviewed native TokenGT required')
    model.validate_attention_configuration()  # Per-layer policies; wrapper owns shared root agreement.
    feature = finite('tokengt.features.v1', feature_axis)
    finite('tokengt.identifier-coordinates.v1', identifier_axis)
    incidence = finite('tokengt.ordered-incidence.v1', tuple(
        side + ':' + name for side in ('left', 'right') for name in identifier_axis))
    role = finite('tokengt.node-edge-role.v1', ('node', 'edge'))
    h, f = Axis('width', 'hidden'), Axis('width', 'ffn')
    declarations = {}
    expected_modules = {'': TokenGT, 'feature': nn.Linear, 'incidence': nn.Linear,
                        'role': nn.Embedding, 'layers': nn.ModuleList, 'final_norm': nn.LayerNorm}
    w = model.feature.out_features
    expected_shapes = {}

    def add(name, shape, kind, axes):
        expected_shapes[name] = shape
        declarations[name] = ParameterRole(kind, axes)

    def linear(prefix, output, input_, axes, kind='hidden', bias=True):
        expected_modules[prefix] = nn.Linear
        add(prefix + '.weight', (output, input_), kind, axes)
        if bias:
            add(prefix + '.bias', (output,), 'vector', (axes[0],))

    def norm(prefix):
        expected_modules[prefix] = nn.LayerNorm
        for suffix in ('weight', 'bias'):
            add(prefix + '.' + suffix, (w,), 'normalization', (h,))

    linear('feature', w, len(feature_axis), (h, feature), 'input', False)
    linear('incidence', w, 2 * len(identifier_axis), (h, incidence), 'custom', False)
    add('role.weight', (2, w), 'embedding', (role, h))
    add('graph_token', (w,), 'vector', (h,))
    if not len(model.layers):
        raise ValueError('native layers required')
    ff = model.layers[0].ffn[0].out_features
    for i, layer in enumerate(model.layers):
        prefix = f'layers.{i}'
        expected_modules[prefix] = EncoderLayer
        norm(prefix + '.norm1'); norm(prefix + '.norm2')
        att = layer.attention
        ap = prefix + '.attention'
        if type(att) is DenseAttention:
            expected_modules[ap] = DenseAttention
            packed = Axis('width', 'packed-qkv')
            linear(ap + '.qkv', 3*w, w, (packed, h), 'custom')
            linear(ap + '.output', w, w, (h, h))
        elif type(att) is TropicalCodeAttention:
            expected_modules[ap] = TropicalCodeAttention
            d = att.head_width
            local = Axis('width', 'head-local')
            for name in ('query', 'key', 'value'):
                add(ap + '.' + name, (d, d), 'custom', (local, local))
            linear(ap + '.output', w, w, (h, h), bias=False)
        else:
            raise ValueError('unreviewed native attention')
        if att.heads * att.head_width != w:
            raise ValueError('inconsistent attention head partition')
        expected_modules[prefix + '.ffn'] = nn.Sequential
        expected_modules[prefix + '.ffn.1'] = nn.GELU
        linear(prefix + '.ffn.0', ff, w, (f, h))
        linear(prefix + '.ffn.2', w, ff, (h, f))
    norm('final_norm')
    modules = dict(model.named_modules(remove_duplicate=False))
    if set(modules) != set(expected_modules) or any(
            type(modules[name]) is not kind for name, kind in expected_modules.items()):
        raise ValueError('unreviewed native module structure')
    named = dict(model.named_parameters(remove_duplicate=False))
    if set(named) != set(expected_shapes) or any(
            tuple(named[name].shape) != shape for name, shape in expected_shapes.items()):
        raise ValueError('native parameter shape or ownership structure changed')
    return parameter_ledger(model, declarations)


def _response_gene_axis(native, *, source_column_positions):
    import hashlib
    import json
    from .sciplex import SciPlexSharedModel

    if type(native) is not SciPlexSharedModel or type(source_column_positions) is not bool:
        raise ValueError('Exact native response owner and ordered axis policy required')
    names = native.axis_ids
    if type(names) is not tuple or not names or any(type(name) is not str or not name.strip() for name in names):
        raise ValueError('Immutable ordered gene axis required')
    if source_column_positions:
        native.validate_configuration()
        if native.config['version'] != 4 or native.config['mup']['policy'] != 'mup-sciplex-v1':
            raise ValueError('Source column positions require the exact applied native response owner')
    if len(set(names)) == len(names):
        # Preserve the exact finite identity used by existing unique-axis owners.
        payload = ['response.genes.v1', names]
    elif source_column_positions:
        payload = ['response.source-column-positions.v1', tuple(enumerate(names))]
    else:
        raise ValueError('Immutable unique ordered gene axis required')
    raw = json.dumps(payload, ensure_ascii=True, separators=(',', ':')).encode('utf-8')
    return names, Axis('finite', hashlib.sha256(raw).hexdigest())


def response_head_parameter_roles(model, *, source_column_positions=False) -> dict[str, ParameterRole]:
    """Local native flow-head declarations for later alias-ledger composition.

    Deliberately excludes the registered native owner (and its backbone). This
    is not a complete model inventory or a scaling certificate. Ordered gene
    names are caller/live-model declarations, not source authenticity evidence.
    Channel-major desired concatenation stays custom pending scaling review.
    No module registration, tensor allocation, forward or state mutation occurs.
    """
    import hashlib
    import json
    from .response_flow import ResponseFlowHead, ResponseFlowDenoiserV2
    from .response_desired_flow import DesiredResponseFlow

    if type(model) not in (ResponseFlowHead, ResponseFlowDenoiserV2, DesiredResponseFlow):
        raise ValueError('exact reviewed response head required')
    desired = type(model) is DesiredResponseFlow
    flow = model.flow if desired else model
    if type(flow) not in (ResponseFlowHead, ResponseFlowDenoiserV2):
        raise ValueError('reviewed nested flow required')
    native = flow.native
    if desired and source_column_positions:
        raise ValueError('Source-column policy is qualified only for the native response-flow heads')
    names, gene = _response_gene_axis(native, source_column_positions=source_column_positions)

    def finite(namespace, labels):
        payload = json.dumps([namespace, labels], ensure_ascii=True, separators=(',', ':'))
        return Axis('finite', hashlib.sha256(payload.encode('utf-8')).hexdigest())

    time = finite('response.time.v1', ('scalar-[0,1]',))
    h = Axis('width', 'hidden')
    w, g = native.edge.numel(), len(names)
    prefix = 'flow.' if desired else ''
    declarations, shapes = {}, {}
    expected = {'': type(model)}
    if desired:
        expected['flow'] = ResponseFlowDenoiserV2

    def linear(name, out_size, in_size, axes, role, bias=True):
        expected[name] = nn.Linear
        declarations[name+'.weight'] = ParameterRole(role, axes)
        shapes[name+'.weight'] = (out_size, in_size)
        if bias:
            declarations[name+'.bias'] = ParameterRole('vector', (axes[0],))
            shapes[name+'.bias'] = (out_size,)

    linear(prefix+'state', w, g, (h, gene), 'input')
    linear(prefix+'time', w, 1, (h, time), 'input')
    linear(prefix+'output', g, w, (gene, h), 'output')
    if desired:
        request = finite('response.channel-major-request.v1', tuple(
            channel+':'+name for channel in ('requested','available','lower','upper') for name in names))
        linear('request', w, 4*g, (h, request), 'custom', False)
    excluded = prefix+'native'
    modules = {n:m for n,m in model.named_modules(remove_duplicate=False)
               if n != excluded and not n.startswith(excluded+'.')}
    if set(modules) != set(expected) or any(type(modules[n]) is not t for n,t in expected.items()):
        raise ValueError('unreviewed response head module ownership')
    parameters = {n:p for n,p in model.named_parameters(remove_duplicate=False)
                  if not n.startswith(excluded+'.')}
    if set(parameters) != set(shapes) or any(tuple(parameters[n].shape) != s for n,s in shapes.items()):
        raise ValueError('response head parameter shape or ownership changed')
    return declarations


def response_parameter_ledger(model, *, source_column_positions=False) -> ParameterLedger:
    """Compose reviewed native response owners and aliases without tensor mutation.

    Accept one native owner/head, or an exact ModuleDict of these components
    sharing ONE native owner. No arbitrary cooperative wrapper is inferred.
    Learned backbone input coordinates are hidden width axes, not finite genes.
    This is a caller/live-model inventory, not source authentication, runtime
    configuration certification or custom-operator scaling qualification.
    """
    import hashlib
    import json
    from .sciplex import SciPlexSharedModel
    from .response_flow import ResponseFlowHead, ResponseFlowDenoiserV2
    from .response_desired_flow import DesiredResponseFlow
    from .response_sampling import INTERVENTIONS

    kinds = (SciPlexSharedModel, ResponseFlowHead, ResponseFlowDenoiserV2, DesiredResponseFlow)
    roots = tuple(model.items()) if type(model) is nn.ModuleDict else (('', model),)
    if not roots or any(type(m) not in kinds for _, m in roots):
        raise ValueError('reviewed nonempty native response components required')
    owners = [m if type(m) is SciPlexSharedModel else m.native for _, m in roots]
    if any(n is not owners[0] for n in owners):
        raise ValueError('one shared native response owner required')
    native = owners[0]
    names, gene = _response_gene_axis(native, source_column_positions=source_column_positions)

    def finite(namespace, labels):
        payload = json.dumps([namespace, labels], ensure_ascii=True, separators=(',', ':'))
        return Axis('finite', hashlib.sha256(payload.encode('utf-8')).hexdigest())

    h = Axis('width', 'hidden')
    intervention = finite('response.intervention-dose.v1', tuple(INTERVENTIONS)+('dose-coordinate',))
    exposure = finite('response.exposure.v1', ('hours-divided-by-24',))
    direction = finite('response.direction.v1', ('response','intervention'))
    w,g = native.edge.numel(),len(names)
    expected_modules = {'':SciPlexSharedModel,'expression':nn.Linear,'intervention':nn.Linear,
        'context':nn.Linear,'direction':nn.Embedding,'response_head':nn.Linear,'intervention_head':nn.Linear}
    actual_modules = {n:m for n,m in native.named_modules(remove_duplicate=False)
                      if n!='backbone' and not n.startswith('backbone.')}
    if set(actual_modules)!=set(expected_modules) or any(type(actual_modules[n]) is not t for n,t in expected_modules.items()):
        raise ValueError('unreviewed native owner module structure')
    declarations,shapes = {},{}
    for name,out_size,in_size,axes,role in (
        ('expression',w,g,(h,gene),'input'),
        ('intervention',w,6,(h,intervention),'input'),
        ('context',w,1,(h,exposure),'input'),
        ('response_head',g,w,(gene,h),'output'),
        ('intervention_head',6,w,(intervention,h),'output')):
        declarations[name+'.weight']=ParameterRole(role,axes)
        declarations[name+'.bias']=ParameterRole('vector',(axes[0],))
        shapes[name+'.weight']=(out_size,in_size);shapes[name+'.bias']=(out_size,)
    declarations['edge']=ParameterRole('vector',(h,));shapes['edge']=(w,)
    declarations['direction.weight']=ParameterRole('embedding',(direction,h));shapes['direction.weight']=(2,w)
    backbone = tokengt_parameter_ledger(native.backbone,
        feature_axis=tuple('hidden-coordinate:'+str(i) for i in range(w)),
        identifier_axis=('context','observation'))
    for entry in backbone.parameters:
        for alias in entry.aliases:
            declarations['backbone.'+alias]=(ParameterRole('hidden',(h,h)) if alias=='feature.weight'
                else ParameterRole(entry.role,entry.axes))
            shapes['backbone.'+alias]=entry.shape
    params=dict(native.named_parameters(remove_duplicate=False))
    if set(params)!=set(shapes) or any(tuple(params[n].shape)!=s for n,s in shapes.items()):
        raise ValueError('native owner parameter shape or ownership changed')
    combined={}
    for prefix,component in roots:
        lead=prefix+'.' if prefix else ''
        owner_prefix='' if type(component) is SciPlexSharedModel else ('flow.native.' if type(component) is DesiredResponseFlow else 'native.')
        combined.update({lead+owner_prefix+n:r for n,r in declarations.items()})
        if type(component) is not SciPlexSharedModel:
            local_roles = response_head_parameter_roles(
                component, source_column_positions=source_column_positions)
            combined.update({lead+n:r for n,r in local_roles.items()})
    return parameter_ledger(model,combined)


def annotation_head_parameter_roles(model) -> dict[str, ParameterRole]:
    """Exact native annotation local map; no subclass or physical-owner admission."""
    return _annotation_head_parameter_roles(model)


def _annotation_head_parameter_roles(model, *, physical_owner=False) -> dict[str, ParameterRole]:
    """Local annotation payload/GO/EC map; native backbone deliberately excluded.

    Vocabulary/label identities are live declarations, not source authentication.
    Positional pooling and concatenated hidden+length projection require a custom
    scaling contract. No initialization, forward, state or tensor mutation.
    """
    import hashlib
    import json
    from .annotation_multihead import SharedAnnotationModel
    from .annotation_vocabulary import AnnotationPayloadEncoder, AnnotationVocabulary
    from .uniprot_go import GOAnnotationHead, _axis as go_axis
    from .uniprot_ec import ECAnnotationHead, _axis as ec_axis
    from mora3.tokenization.graph_payload import GraphPayloadCodec
    from mora3.tokenization.shortest_path import ByteTokenizer
    from .mixed_payload_projection import MixedPayloadProjection

    from .physical_trainer import SharedPhysicalModel
    owner_type = SharedPhysicalModel if physical_owner else SharedAnnotationModel
    if type(model) is not owner_type:
        raise ValueError('exact reviewed shared annotation model required')
    expected={'':owner_type,'payload':AnnotationPayloadEncoder,
        'payload.embedding':nn.Embedding,'payload.projection':nn.Linear,
        'go':GOAnnotationHead,'go.output':nn.Linear,'ec':ECAnnotationHead,'ec.output':nn.Linear}
    split=(type(getattr(model,'payload',None)) is AnnotationPayloadEncoder
           and type(getattr(model.payload,'projection',None)) is MixedPayloadProjection)
    if split:expected['payload.projection']=MixedPayloadProjection
    modules={n:m for n,m in model.named_modules(remove_duplicate=False)
             if n!='backbone' and not n.startswith('backbone.')
             and not (physical_owner and (n=='physical' or n.startswith('physical.')))}
    if set(modules)!=set(expected) or any(type(modules[n]) is not t for n,t in expected.items()):
        raise ValueError('unreviewed annotation module ownership')
    p=model.payload
    if type(p.codec) is not GraphPayloadCodec or type(p.codec.tokenizer) is not ByteTokenizer:
        raise ValueError('exact reviewed annotation tokenizer required')
    try:
        spec=json.loads(p._vocabulary_json)
        tokenizer=AnnotationVocabulary(p._vocabulary_json).tokenizer()
        live=p.codec.tokenizer
        controls=tuple(spec['controls'])
        if (live.vocabulary!=tokenizer.vocabulary or live.size!=tokenizer.size
                or live._controls!=tokenizer._controls or p.offset!=tokenizer.size):
            raise ValueError('annotation vocabulary order/offset drift')
    except (KeyError,TypeError,AttributeError,json.JSONDecodeError) as exc:
        raise ValueError('valid bound annotation vocabulary required') from exc
    for head,validate in ((model.go,go_axis),(model.ec,ec_axis)):
        if type(head.axis_ids) is not tuple or validate(head.axis_ids)!=head.axis_ids:
            raise ValueError('immutable valid ordered annotation axis required')

    def finite(namespace,names):
        raw=json.dumps([namespace,names],ensure_ascii=True,separators=(',',':'))
        return Axis('finite',hashlib.sha256(raw.encode('utf-8')).hexdigest())

    tokens=tuple('payload-byte-string:'+b.hex() for b in tokenizer.vocabulary)
    tokens+=tuple('control:'+c for c in controls)
    tokens+=tuple('metadata-byte:'+str(i) for i in range(256))+('field-start','field-value')
    token_axis=finite('annotation.payload-embedding-rows.v1',tokens)
    h=Axis('width','hidden')
    pooled=Axis('width','annotation-hidden-plus-length')
    declarations={
        'payload.embedding.weight':ParameterRole('embedding',(token_axis,h)),
        'payload.projection.weight':ParameterRole('custom',(h,pooled)),
        'payload.projection.bias':ParameterRole('vector',(h,))}
    w=p.embedding.embedding_dim
    shapes={'payload.embedding.weight':(len(tokens),w),
            'payload.projection.weight':(w,w+1),'payload.projection.bias':(w,)}
    if split:
        from .payload_mup import payload_mup_parameter_roles
        declarations={'payload.'+n:r for n,r in payload_mup_parameter_roles(p).items()}
        shapes={'payload.'+n:tuple(value.shape) for n,value in p.named_parameters(remove_duplicate=False)}
    for name,head in (('go',model.go),('ec',model.ec)):
        axis=finite('annotation.'+name+'-labels.v1',head.axis_ids)
        declarations[name+'.output.weight']=ParameterRole('output',(axis,h))
        declarations[name+'.output.bias']=ParameterRole('vector',(axis,))
        shapes[name+'.output.weight']=(len(head.axis_ids),w)
        shapes[name+'.output.bias']=(len(head.axis_ids),)
    params={n:p for n,p in model.named_parameters(remove_duplicate=False)
            if not n.startswith('backbone.')
            and not (physical_owner and n.startswith('physical.'))}
    if set(params)!=set(shapes) or any(tuple(params[n].shape)!=s for n,s in shapes.items()):
        raise ValueError('annotation local parameter shape or ownership drift')
    return declarations


def annotation_parameter_ledger(model, *, identifier_axis: tuple[str, ...]) -> ParameterLedger:
    """Complete native annotation inventory, including shared backbone aliases.

    Accept an exact annotation model or ModuleDict of such models sharing ONE
    backbone. Identifier names are explicit caller declarations, not biological
    provenance. Learned feature coordinates use the hidden width axis. Custom
    pooling, incidence and QKV remain unqualified for scaling. No tensor access
    beyond shape/count metadata, forward, initialization or state mutation.
    """
    from .annotation_multihead import SharedAnnotationModel

    roots = tuple(model.items()) if type(model) is nn.ModuleDict else (('', model),)
    if not roots or any(type(m) is not SharedAnnotationModel for _, m in roots):
        raise ValueError('reviewed nonempty annotation components required')
    backbone = roots[0][1].backbone
    if any(m.backbone is not backbone for _, m in roots):
        raise ValueError('one shared annotation backbone required')
    w = roots[0][1].payload.embedding.embedding_dim
    native = tokengt_parameter_ledger(backbone,
        feature_axis=tuple('hidden-coordinate:'+str(i) for i in range(w)),
        identifier_axis=identifier_axis)
    if backbone.feature.out_features != w:
        raise ValueError('annotation backbone hidden width mismatch')
    combined = {}
    h = Axis('width', 'hidden')
    for prefix, component in roots:
        lead = prefix+'.' if prefix else ''
        if component.payload.embedding.embedding_dim != w:
            raise ValueError('shared annotation hidden width mismatch')
        combined.update({lead+n:r for n,r in annotation_head_parameter_roles(component).items()})
        for entry in native.parameters:
            declaration = (ParameterRole('hidden', (h,h)) if 'feature.weight' in entry.aliases
                           else ParameterRole(entry.role, entry.axes))
            for alias in entry.aliases:
                combined[lead+'backbone.'+alias] = declaration
    return parameter_ledger(model, combined)


def physical_head_parameter_roles(model) -> dict[str, ParameterRole]:
    """Local observed-assertion metadata head axes, never numeric properties.

    Parameter type is a supplied query; unit/scope/method remain prediction
    outputs. Ordered labels are caller/live-model declarations, not authenticated
    source eligibility. This local map does not qualify latent-plus-query addition
    or the cooperative owner for muP scaling. No tensor/RNG/state mutation occurs.
    """
    import hashlib
    import json
    from .physical_head import StructuredPhysicalHead

    if type(model) is not StructuredPhysicalHead:
        raise ValueError('exact reviewed structured physical head required')

    def finite(namespace, names):
        if (type(names) is not tuple or not names or
                any(type(n) is not str or not n.strip() for n in names) or
                len(set(names)) != len(names)):
            raise ValueError('immutable unique nonempty ordered physical axis required')
        payload = json.dumps([namespace, names], ensure_ascii=True, separators=(',', ':'))
        return Axis('finite', hashlib.sha256(payload.encode('utf-8')).hexdigest())

    parameter = finite('physical.parameter-types.v1', model.parameter_axis)
    method = finite('physical.method-labels.v1', model.method_axis)
    modules = dict(model.named_modules(remove_duplicate=False))
    expected = {'': StructuredPhysicalHead, 'query': nn.Embedding,
                'unit': nn.Linear, 'scope': nn.Linear, 'method': nn.Linear}
    if set(modules) != set(expected) or any(type(modules[n]) is not t for n,t in expected.items()):
        raise ValueError('unreviewed physical head module structure')
    w = model.query.embedding_dim
    expected_config=dict(
            version=1, width=w, parameters=model.parameter_axis, methods=model.method_axis,
            objective='observed-slot-CE-sum')
    if model.configuration.get('version')==2:
        model.validate_parameterization_configuration()
        expected_config.update(version=2,mup=model.get_extra_state()['mup'])
    if type(w) is not int or w < 1 or model.configuration!=expected_config:
        raise ValueError('physical head configuration drift')
    h = Axis('width', 'hidden')
    declarations = {'query.weight': ParameterRole('embedding', (parameter, h))}
    shapes = {'query.weight': (len(model.parameter_axis), w)}
    for name, axis, size in (
            ('unit', finite('physical.unit-labels.v1', ('unknown', 'explicit_source_unit')), 2),
            ('scope', finite('physical.scope-labels.v1', ('unscoped', 'molecule-scoped')), 2),
            ('method', method, len(model.method_axis))):
        declarations[name+'.weight'] = ParameterRole('output', (axis, h))
        declarations[name+'.bias'] = ParameterRole('vector', (axis,))
        shapes[name+'.weight'] = (size, w)
        shapes[name+'.bias'] = (size,)
    parameters = dict(model.named_parameters(remove_duplicate=False))
    if set(parameters) != set(shapes) or any(tuple(parameters[n].shape) != s for n,s in shapes.items()):
        raise ValueError('physical head parameter shape or ownership drift')
    return declarations


def physical_parameter_ledger(model, *, identifier_axis: tuple[str, ...]) -> ParameterLedger:
    """Complete physical/GO/EC owner inventory with one shared native backbone.

    Ordered finite axes are declarations, not source authentication or admission.
    Latent/query addition and custom backbone operators remain unqualified for
    scaling. Only shape/ownership metadata is inspected; no forward or mutation.
    """
    from .physical_trainer import SharedPhysicalModel

    roots = tuple(model.items()) if type(model) is nn.ModuleDict else (('', model),)
    if not roots or any(type(m) is not SharedPhysicalModel for _, m in roots):
        raise ValueError('reviewed nonempty physical components required')
    backbone = roots[0][1].backbone
    if any(m.backbone is not backbone for _, m in roots):
        raise ValueError('one shared physical backbone required')
    w = roots[0][1].payload.embedding.embedding_dim
    native = tokengt_parameter_ledger(backbone,
        feature_axis=tuple('hidden-coordinate:'+str(i) for i in range(w)),
        identifier_axis=identifier_axis)
    if backbone.feature.out_features != w:
        raise ValueError('physical backbone hidden width mismatch')
    combined = {}
    h = Axis('width', 'hidden')
    for prefix, component in roots:
        lead = prefix+'.' if prefix else ''
        if (component.payload.embedding.embedding_dim != w or
                component.physical.query.embedding_dim != w):
            raise ValueError('shared physical hidden width mismatch')
        combined.update({lead+n:r for n,r in
                         _annotation_head_parameter_roles(component, physical_owner=True).items()})
        combined.update({lead+'physical.'+n:r for n,r in
                         physical_head_parameter_roles(component.physical).items()})
        for entry in native.parameters:
            declaration = (ParameterRole('hidden', (h,h)) if 'feature.weight' in entry.aliases
                           else ParameterRole(entry.role, entry.axes))
            for alias in entry.aliases:
                combined[lead+'backbone.'+alias] = declaration
    return parameter_ledger(model, combined)


def residue_head_parameter_roles(model) -> dict[str, ParameterRole]:
    """Local residue affine and custom convolution axes, without scaling rules.

    Ordered feature labels remain finite. The width-three local convolution is
    explicitly custom: width_family refuses it until its operator scaling is
    reviewed. Sinusoidal positions/additions and shared ownership are not qualified
    by this parameter inventory. No forward, RNG or parameter mutation occurs.
    """
    import hashlib
    import json
    from .residue_objective import ContextualResidueHead

    if type(model) is not ContextualResidueHead:
        raise ValueError('exact reviewed contextual residue head required')
    w, labels = model.width, model.feature_axis
    if (type(w) is not int or w < 2 or w % 2 or type(labels) is not tuple
            or not labels or any(type(n) is not str or not n.strip() for n in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError('even native width and immutable ordered residue axis required')
    modules = dict(model.named_modules(remove_duplicate=False))
    expected = {'': ContextualResidueHead, 'context': nn.Linear,
                'local': nn.Conv1d, 'output': nn.Linear}
    if set(modules) != set(expected) or any(type(modules[n]) is not t for n,t in expected.items()):
        raise ValueError('unreviewed residue head module structure')
    if (model.context.in_features, model.context.out_features,
            model.output.in_features, model.output.out_features) != (w,w,w,len(labels)):
        raise ValueError('residue affine configuration drift')
    conv = model.local
    if (conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
            conv.padding, conv.dilation, conv.groups, conv.padding_mode,
            conv.transposed, conv.output_padding) != (w,w,(3,),(1,),(1,),(1,),1,'zeros',False,(0,)):
        raise ValueError('residue convolution configuration drift')

    def finite(namespace, names):
        payload = json.dumps([namespace,names],ensure_ascii=True,separators=(',',':'))
        return Axis('finite',hashlib.sha256(payload.encode('utf-8')).hexdigest())

    h = Axis('width','hidden')
    f = finite('residue.feature-types.v1', labels)
    k = finite('residue.local-offsets.v1', ('-1','0','1'))
    declarations = {
        'context.weight': ParameterRole('hidden',(h,h)),
        'context.bias': ParameterRole('vector',(h,)),
        'local.weight': ParameterRole('custom',(h,h,k)),
        'local.bias': ParameterRole('vector',(h,)),
        'output.weight': ParameterRole('output',(f,h)),
        'output.bias': ParameterRole('vector',(f,))}
    shapes = {'context.weight': (w,w), 'context.bias': (w,),
              'local.weight': (w,w,3), 'local.bias': (w,),
              'output.weight': (len(labels),w), 'output.bias': (len(labels),)}
    parameters = dict(model.named_parameters(remove_duplicate=False))
    if set(parameters) != set(shapes) or any(tuple(parameters[n].shape) != s for n,s in shapes.items()):
        raise ValueError('residue parameter shape or ownership drift')
    return declarations


def residue_parameter_ledger(model, *, identifier_axis: tuple[str, ...]) -> ParameterLedger:
    """Complete exact residue owners sharing one native annotation backbone.

    Finite labels are caller declarations, not source/admission certificates.
    Convolution, pooling, addition and custom backbone scaling stay unqualified.
    This inspects parameter metadata only; it does not forward or mutate state.
    """
    from .residue_objective import SharedResidueModel
    from .annotation_multihead import SharedAnnotationModel
    from .physical_trainer import SharedPhysicalModel

    roots = tuple(model.items()) if type(model) is nn.ModuleDict else (('', model),)
    if not roots or any(type(m) is not SharedResidueModel for _, m in roots):
        raise ValueError('exact nonempty native residue components required')
    combined = {}
    backbone = None
    for prefix, component in roots:
        if set(component._modules) != {'shared', 'head'}:
            raise ValueError('unreviewed residue owner module structure')
        owner = component.shared
        if type(owner) is SharedAnnotationModel:
            native = annotation_parameter_ledger(owner, identifier_axis=identifier_axis)
        elif type(owner) is SharedPhysicalModel:
            native = physical_parameter_ledger(owner, identifier_axis=identifier_axis)
        else:
            raise ValueError('exact native annotation or physical owner required')
        if backbone is None:
            backbone = owner.backbone
        elif owner.backbone is not backbone:
            raise ValueError('one shared residue backbone required')
        local = residue_head_parameter_roles(component.head)
        if component.head.width != owner.payload.embedding.embedding_dim:
            raise ValueError('residue owner hidden width mismatch')
        lead = prefix+'.' if prefix else ''
        for entry in native.parameters:
            for alias in entry.aliases:
                combined[lead+'shared.'+alias] = ParameterRole(entry.role,entry.axes)
        combined.update({lead+'head.'+name:role for name,role in local.items()})
    return parameter_ledger(model,combined)


def image_encoder_parameter_roles(model, *, channel_axis: tuple[str, ...]) -> dict[str, ParameterRole]:
    """Native multiscale pixel axes only; convolutions remain custom/unqualified.

    Channel labels are caller declarations, not source authentication. Distinct
    concatenation axes preserve skip-before-decoder ordering. Pooling, bilinear
    interpolation and spatial reuse receive no scaling certificate here.
    """
    import hashlib
    import json
    from .morphology import MultiscalePixels
    if type(model) is not MultiscalePixels:
        raise ValueError('exact native multiscale encoder required')
    if (type(channel_axis) is not tuple or not channel_axis
            or any(type(n) is not str or not n.strip() for n in channel_axis)
            or len(set(channel_axis)) != len(channel_axis)):
        raise ValueError('immutable unique ordered image channels required')
    modules=dict(model.named_modules(remove_duplicate=False))
    expected={'':MultiscalePixels}
    blocks=('enc0','enc1','bottom','dec1','dec0')
    for name in blocks:
        expected[name]=nn.Sequential
        for i,t in enumerate((nn.Conv2d,nn.GELU,nn.Conv2d,nn.GELU)):
            expected[name+'.'+str(i)]=t
    if set(modules)!=set(expected) or any(type(modules[n]) is not t for n,t in expected.items()):
        raise ValueError('unreviewed image encoder structure')
    w=model.enc0[0].out_channels
    if type(w) is not int or w<2 or w%2:
        raise ValueError('even native image width required')
    def finite(namespace,names):
        data=json.dumps([namespace,names],ensure_ascii=True,separators=(',',':'))
        return Axis('finite',hashlib.sha256(data.encode()).hexdigest())
    channel=finite('image.input-channels.v1',channel_axis)
    ky=finite('image.kernel-row-offsets.v1',('-1','0','1'))
    kx=finite('image.kernel-column-offsets.v1',('-1','0','1'))
    h=Axis('width','hidden')
    h2=Axis('width','image.level2'); h4=Axis('width','image.level4')
    skip6=Axis('width','image.concat-skip2-bottom4')
    skip3=Axis('width','image.concat-skip1-decoder2')
    layout=((len(channel_axis),w,channel,h),(w,2*w,h,h2),
            (2*w,4*w,h2,h4),(6*w,2*w,skip6,h2),(3*w,w,skip3,h))
    declarations={}; shapes={}
    for name,(a,b,ia,oa) in zip(blocks,layout):
        for index,cin,i_axis in ((0,a,ia),(2,b,oa)):
            prefix=name+'.'+str(index); conv=modules[prefix]
            if (conv.in_channels,conv.out_channels,conv.kernel_size,conv.stride,
                conv.padding,conv.dilation,conv.groups,conv.padding_mode,
                conv.transposed,conv.output_padding)!=(cin,b,(3,3),(1,1),(1,1),(1,1),1,'zeros',False,(0,0)):
                raise ValueError('native image convolution configuration drift')
            declarations[prefix+'.weight']=ParameterRole('custom',(oa,i_axis,ky,kx))
            declarations[prefix+'.bias']=ParameterRole('vector',(oa,))
            shapes[prefix+'.weight']=(b,cin,3,3); shapes[prefix+'.bias']=(b,)
    parameters=dict(model.named_parameters(remove_duplicate=False))
    if set(parameters)!=set(shapes) or any(tuple(parameters[n].shape)!=s for n,s in shapes.items()):
        raise ValueError('native image parameter ownership or shape drift')
    return declarations


def instance_head_parameter_roles(model) -> dict[str, ParameterRole]:
    """Local independent mask-slot axes; no convolution scaling qualification.

    Slot ordinals are exchangeable output coordinates, not biological identities
    or source instance IDs. Spatial singleton axes are explicit. This inventories
    registered metadata only; arbitrary hooks/forward replacement are outside its
    scope. No forward, target access, state mutation or admission is performed.
    """
    import hashlib
    import json
    from .instance_affinity import OverlappingInstanceHead
    if type(model) is not OverlappingInstanceHead:
        raise ValueError('exact native overlapping instance head required')
    w, s = model.channels, model.slots
    policy = model.slot_policy
    full=policy in('source-envelope-v3','source-layer-list-v4')
    if full:
        from .raster_source_envelope import RasterSourceEnvelope
        if type(model.source_envelope)is not RasterSourceEnvelope:raise ValueError('explicit source instance envelope required')
        model.get_extra_state()
    if (policy not in ('bounded-max64-v1','extended-max512-v2','source-envelope-v3','source-layer-list-v4')
            or type(w) is not int or w < 1 or type(s) is not int
            or not 1 <= s <= (model.source_envelope.max_instances if full else(512 if policy=='extended-max512-v2'else 64))
            or type(model.max_output_bytes) is not int or model.max_output_bytes < 1):
        raise ValueError('native instance configuration drift')
    modules = dict(model.named_modules(remove_duplicate=False))
    if (set(modules) != {'','projection'}
            or type(modules['projection']) is not nn.Conv2d):
        raise ValueError('unreviewed instance head structure')
    conv = model.projection
    if (conv.in_channels,conv.out_channels,conv.kernel_size,conv.stride,
            conv.padding,conv.dilation,conv.groups,conv.padding_mode,
            conv.transposed,conv.output_padding) != (w,s,(1,1),(1,1),(0,0),(1,1),1,'zeros',False,(0,0)):
        raise ValueError('instance projection configuration drift')
    shapes = {'projection.weight':(s,w,1,1),'projection.bias':(s,)}
    params = dict(model.named_parameters(remove_duplicate=False))
    if set(params) != set(shapes) or any(tuple(params[n].shape) != sh for n,sh in shapes.items()):
        raise ValueError('instance projection parameter ownership or shape drift')
    def finite(namespace, values):
        payload=json.dumps([namespace,values],separators=(',',':'))
        return Axis('finite',hashlib.sha256(payload.encode()).hexdigest())
    slots=(finite('source-instance.exchangeable-slot-ordinals.v2'if policy=='source-envelope-v3'else'source-mask.ordered-layer-ordinals.v1',
        (tuple(range(s)),model.source_envelope.binding_for_slot_policy(policy)))
        if full else finite('instance.exchangeable-slot-ordinals.v1',tuple(range(s))))
    row=finite('instance.kernel-row-offsets.v1',(0,))
    column=finite('instance.kernel-column-offsets.v1',(0,))
    return {'projection.weight':ParameterRole('custom',(slots,Axis('width','hidden'),row,column)),
            'projection.bias':ParameterRole('vector',(slots,))}


def instance_parameter_ledger(model, *, channel_axis: tuple[str, ...],
                              identifier_axis: tuple[str, ...]) -> ParameterLedger:
    """Complete native RGB instance owners sharing one identity-equal backbone.

    Encoder and slot parameters may alias; conflicting declarations refuse.
    Local/context concatenation order is explicit. Pooling, broadcast, packed
    attention and convolutions remain unqualified custom scaling boundaries.
    Labels are caller declarations, not source or training-admission evidence.
    No forward, target access, initialization or checkpoint mutation occurs.
    """
    from .color_semantic import ColorInstancePredictor
    from .instance_affinity import OverlappingInstanceHead
    roots = tuple(model.items()) if type(model) is nn.ModuleDict else (('',model),)
    if not roots or any(type(m) is not ColorInstancePredictor for _,m in roots):
        raise ValueError('exact nonempty native color instance owners required')
    combined = {}; backbone = None
    h = Axis('width','hidden')
    for prefix, owner in roots:
        w = owner.width
        if (type(w) is not int or w < 2 or w % 2
                or owner.alpha_policy != 'require-opaque'
                or owner.input_policy not in ('native-max512-v1','native-max1000-v1','source-envelope-v2')
                or type(owner._modules.get('head')) is not OverlappingInstanceHead
                or set(owner._modules) != ({'pixels','backbone','head','count'}if owner.head.slot_policy=='source-layer-list-v4'else{'pixels','backbone','head'})):
            raise ValueError('native instance owner configuration or structure drift')
        pixels = image_encoder_parameter_roles(owner.pixels,channel_axis=channel_axis)
        if len(channel_axis) != 3 or owner.pixels.enc0[0].out_channels != w:
            raise ValueError('native RGB encoder width/channel mismatch')
        if backbone is None: backbone = owner.backbone
        elif owner.backbone is not backbone:
            raise ValueError('one shared native image backbone required')
        native = tokengt_parameter_ledger(owner.backbone,
            feature_axis=tuple('hidden-coordinate:'+str(i) for i in range(w)),
            identifier_axis=identifier_axis)
        if owner.backbone.feature.out_features != w:
            raise ValueError('native image backbone width mismatch')
        head = instance_head_parameter_roles(owner.head)
        if owner.head.channels != 2*w:
            raise ValueError('native local/context concatenation width mismatch')
        weight = head['projection.weight']
        head['projection.weight'] = ParameterRole(weight.role,
            (weight.axes[0],Axis('width','image.concat-local-context'),*weight.axes[2:]))
        lead = prefix+'.' if prefix else ''
        combined.update({lead+'pixels.'+n:r for n,r in pixels.items()})
        combined.update({lead+'head.'+n:r for n,r in head.items()})
        if owner.head.slot_policy=='source-layer-list-v4':
            owner.get_extra_state();count=owner.count
            if (type(count)is not nn.Linear or count._modules or count._buffers
                    or(count.in_features,count.out_features)!=(w,owner.head.slots+1)):
                raise ValueError('exact source-supplied-layer categorical count readout required')
            import hashlib,json
            axis=Axis('finite',hashlib.sha256(json.dumps(['source-mask-layer-count.v1',tuple(range(owner.head.slots+1)),
                owner.source_envelope.binding_for_slot_policy(owner.head.slot_policy)],sort_keys=True,separators=(',',':')).encode()).hexdigest())
            combined.update({lead+'count.weight':ParameterRole('output',(axis,h)),lead+'count.bias':ParameterRole('vector',(axis,))})

        for entry in native.parameters:
            role = (ParameterRole('hidden',(h,h)) if 'feature.weight' in entry.aliases
                    else ParameterRole(entry.role,entry.axes))
            for alias in entry.aliases: combined[lead+'backbone.'+alias] = role
    return parameter_ledger(model,combined)


def conditioned_instance_offset_parameter_roles(model) -> dict[str, ParameterRole]:
    """Local task-context to ordered-slot offset only, without scaling approval.

    Does not inventory the registered color/context owners or authenticate source
    data. Shared backbone/payload composition requires a separate complete map.
    """
    from .color_conditioning import ConditionedColorInstance
    from .color_semantic import ColorInstancePredictor
    if type(model) is not ConditionedColorInstance or type(model.color) is not ColorInstancePredictor:
        raise ValueError('exact native conditioned instance owner required')
    w=model.color.width
    head=instance_head_parameter_roles(model.color.head)
    slots=model.color.head.slots
    if (type(w) is not int or w<2 or w%2 or model.color.head.channels!=2*w
            or model.config != dict(version=1,architecture='task-graph-per-slot-logit-offset-v1',
                width=w,id_width=model.config.get('id_width'),slots=slots,
                pooling='existing-positional-payload',training_admission=False)):
        raise ValueError('conditioned instance offset configuration drift')
    linear=model.offset
    if (type(linear) is not nn.Linear or set(dict(linear.named_modules()))!={''}
            or linear.in_features!=w or linear.out_features!=slots):
        raise ValueError('native conditioned offset structure drift')
    params=dict(linear.named_parameters(remove_duplicate=False))
    shapes={'weight':(slots,w),'bias':(slots,)}
    if set(params)!=set(shapes) or any(tuple(params[n].shape)!=s for n,s in shapes.items()):
        raise ValueError('native conditioned offset parameter drift')
    axis=head['projection.weight'].axes[0]
    return {'weight':ParameterRole('output',(axis,Axis('width','hidden'))),
            'bias':ParameterRole('vector',(axis,))}


def semantic_parameter_ledger(model, *, channel_axis: tuple[str, ...],
                              identifier_axis: tuple[str, ...]) -> ParameterLedger:
    """Exact native nuclear-foreground owner; no affinity/instance inference.

    RGB and image identifier names are explicit ordered declarations. Inventories
    metadata only, without forward, scaling, source admission or state changes.
    Convolutions/pooling/attention remain unqualified custom operators.
    """
    import hashlib
    import json
    from .color_semantic import ColorNuclearSemantic
    if type(model) is not ColorNuclearSemantic:
        raise ValueError('exact native color semantic owner required')
    w=model.width
    if (type(w) is not int or w<2 or w%2 or model.alpha_policy!='require-opaque'
            or model.input_policy not in ('native-max512-v1','native-max1000-v1','source-envelope-v2')
            or type(channel_axis) is not tuple or len(channel_axis)!=3
            or type(identifier_axis) is not tuple or len(identifier_axis)!=1
            or set(model._modules)!={'pixels','backbone','head'}
            or any(m._buffers for m in model.modules())):
        raise ValueError('native semantic configuration or ownership drift')
    pixels=image_encoder_parameter_roles(model.pixels,channel_axis=channel_axis)
    if model.pixels.enc0[0].out_channels!=w:
        raise ValueError('native semantic encoder width drift')
    native=tokengt_parameter_ledger(model.backbone,
        feature_axis=tuple('hidden-coordinate:'+str(i) for i in range(w)),
        identifier_axis=identifier_axis)
    if model.backbone.feature.out_features!=w:
        raise ValueError('native semantic backbone width drift')
    conv=model.head
    if (type(conv) is not nn.Conv2d or set(conv._modules)
            or (conv.in_channels,conv.out_channels,conv.kernel_size,conv.stride,
                conv.padding,conv.dilation,conv.groups,conv.padding_mode,
                conv.transposed,conv.output_padding)!=(2*w,1,(1,1),(1,1),(0,0),(1,1),1,'zeros',False,(0,0))):
        raise ValueError('native semantic projection configuration drift')
    params=dict(conv.named_parameters(remove_duplicate=False))
    shapes={'weight':(1,2*w,1,1),'bias':(1,)}
    if set(params)!=set(shapes) or any(tuple(params[n].shape)!=sh for n,sh in shapes.items()):
        raise ValueError('native semantic projection parameter drift')
    def finite(namespace,values):
        return Axis('finite',hashlib.sha256(json.dumps([namespace,values],separators=(',',':')).encode()).hexdigest())
    output=finite('semantic.output.v1',('nuclear-foreground-logit',))
    h=Axis('width','hidden')
    combined={'pixels.'+n:r for n,r in pixels.items()}
    combined.update({'head.weight':ParameterRole('custom',(output,
        Axis('width','image.concat-local-context'),finite('instance.kernel-row-offsets.v1',(0,)),
        finite('instance.kernel-column-offsets.v1',(0,)))),
        'head.bias':ParameterRole('vector',(output,))})
    for entry in native.parameters:
        role=ParameterRole('hidden',(h,h)) if 'feature.weight' in entry.aliases else ParameterRole(entry.role,entry.axes)
        for alias in entry.aliases:combined['backbone.'+alias]=role
    return parameter_ledger(model,combined)


def conditioned_semantic_parameter_ledger(model, *, channel_axis: tuple[str, ...],
                                          identifier_axis: tuple[str, ...]) -> ParameterLedger:
    """Exact task-selected semantic owner with explicit shared context aliases.

    No target/source access, scaling, optimizer or checkpoint mutation. Semantic
    output is a single foreground logit, not an affinity or instance-ID axis.
    """
    return _conditioned_image_parameter_ledger(model,channel_axis=channel_axis,
        identifier_axis=identifier_axis,semantic=True)


def conditioned_instance_parameter_ledger(model, *, channel_axis: tuple[str, ...],
                                         identifier_axis: tuple[str, ...]) -> ParameterLedger:
    """Complete native conditioned RGB owner, including shared graph aliases.

    Construction inventory only. Positional pooling, byte/length concatenation,
    incidence and image operators retain unqualified custom scaling boundaries.
    No source admission, initialization, forward or checkpoint change occurs.
    """
    return _conditioned_image_parameter_ledger(model,channel_axis=channel_axis,
        identifier_axis=identifier_axis)


def _conditioned_image_parameter_ledger(model, *, channel_axis, identifier_axis, semantic=False):
    import hashlib
    import json
    from .geometry_conditioning import GeometryConditioner
    from .payload import PayloadEncoder
    from mora3.tokenization.graph_payload import GraphPayloadCodec
    from mora3.tokenization.shortest_path import ByteTokenizer
    if semantic:
        from .color_conditioning import ConditionedColorSemantic
        if (type(model) is not ConditionedColorSemantic
                or any(m._buffers for m in model.modules())):
            raise ValueError('exact native conditioned semantic owner required')
        color_ledger=semantic_parameter_ledger(model.color,channel_axis=channel_axis,identifier_axis=('image',))
        w=model.color.width
        if (model.config!=dict(version=1,architecture='task-graph-uniform-logit-offset-v1',
                width=w,id_width=model.config.get('id_width'),pooling='existing-positional-payload',
                training_admission=False) or type(model.offset) is not nn.Linear
                or model.offset.in_features!=w or model.offset.out_features!=1
                or model.offset._modules):
            raise ValueError('native semantic offset configuration drift')
        params=dict(model.offset.named_parameters(remove_duplicate=False))
        if (set(params)!={'weight','bias'} or tuple(params['weight'].shape)!=(1,w)
                or tuple(params['bias'].shape)!=(1,)):
            raise ValueError('native semantic offset parameter drift')
        output=next(e for e in color_ledger.parameters if 'head.bias' in e.aliases).axes[0]
        offset={'weight':ParameterRole('output',(output,Axis('width','hidden'))),
                'bias':ParameterRole('vector',(output,))}
    else:
        offset=conditioned_instance_offset_parameter_roles(model)
        color_ledger=instance_parameter_ledger(model.color,channel_axis=channel_axis,identifier_axis=('image',))
    if set(model._modules)!={'color','context','offset'}:
        raise ValueError('conditioned owner structure drift')
    c=model.context; w=model.color.width
    if type(c) is not GeometryConditioner or set(c._modules)!={'payload','graph'}:
        raise ValueError('native geometry context required')
    rank=len(identifier_axis)
    expected=dict(version=1,architecture='native-full-geometry-shared-v1',id_width=rank,
        feature_width=w,width=w,shared_layers=len(model.color.backbone.layers),
        identifiers='exact-identity-zero-padded',pooling='scalar-positional-then-graph-token',
        discrete_geometry_gradient=False)
    if c.config!=expected or model.config['id_width']!=rank:
        raise ValueError('geometry configuration drift')
    for name in ('feature','role','graph_token','layers','final_norm'):
        if getattr(c.graph,name) is not getattr(model.color.backbone,name):
            raise ValueError('geometry must share exact native transforms')
    payload=c.payload; tokenizer=ByteTokenizer(controls=('pad','node','edge'))
    if (type(payload) is not PayloadEncoder or set(payload._modules)!={'embedding','projection'}
            or type(payload.embedding) is not nn.Embedding or type(payload.projection) is not nn.Linear
            or type(payload.codec) is not GraphPayloadCodec
            or type(payload.codec.tokenizer) is not ByteTokenizer
            or payload.codec.tokenizer.vocabulary!=tokenizer.vocabulary
            or payload.codec.tokenizer._controls!=tokenizer._controls
            or payload.offset!=tokenizer.size or payload.pooling!='positional'
            or type(payload.max_tokens) is not int or payload.max_tokens<=0):
        raise ValueError('native byte payload configuration drift')
    shapes={'embedding.weight':(517,w),'projection.weight':(w,w+1),'projection.bias':(w,)}
    params=dict(payload.named_parameters(remove_duplicate=False))
    if (set(params)!=set(shapes) or any(tuple(params[n].shape)!=s for n,s in shapes.items())
            or payload.embedding.num_embeddings!=517 or payload.embedding.embedding_dim!=w
            or payload.projection.in_features!=w+1 or payload.projection.out_features!=w):
        raise ValueError('native payload shape drift')
    combined={}
    def include(prefix,ledger):
        for entry in ledger.parameters:
            role=ParameterRole(entry.role,entry.axes)
            if 'feature.weight' in entry.aliases:
                role=ParameterRole('hidden',(Axis('width','hidden'),)*2)
            for alias in entry.aliases: combined[prefix+alias]=role
    include('color.',color_ledger)
    include('context.graph.',tokengt_parameter_ledger(c.graph,
        feature_axis=tuple('hidden-coordinate:'+str(i) for i in range(w)),identifier_axis=identifier_axis))
    raw=json.dumps(['geometry.byte-payload.v1',tuple(b.hex() for b in tokenizer.vocabulary),
                    tuple(tokenizer._controls),tuple(range(258))],separators=(',',':'))
    tokens=Axis('finite',hashlib.sha256(raw.encode()).hexdigest()); h=Axis('width','hidden')
    combined.update({'context.payload.embedding.weight':ParameterRole('embedding',(tokens,h)),
        'context.payload.projection.weight':ParameterRole('custom',(h,Axis('width','payload-hidden-plus-length'))),
        'context.payload.projection.bias':ParameterRole('vector',(h,))})
    combined.update({'offset.'+n:r for n,r in offset.items()})
    return parameter_ledger(model,combined)


def compose_owner_parameter_ledgers(model: nn.Module,
                                    owners: Mapping[str, ParameterLedger]) -> ParameterLedger:
    """Compose complete immediate-child inventories against the current owner.

    Child ledgers remain explicit caller declarations, not source authentication
    or scaling certificates. Revalidate every local alias/shape/dtype/gradient
    flag and count before merging; the final inventory checks cross-owner ties.
    Root parameters and uncovered children refuse rather than acquire inferred
    axes. No tensor reads, forward calls or state changes are required.
    """
    if not isinstance(model, nn.Module) or not isinstance(owners, Mapping):
        raise ValueError('module and complete child ledgers required')
    children = dict(model._modules)
    if (set(owners) != set(children) or any(c is None for c in children.values())
            or any(v is not None for v in model._parameters.values())):
        raise ValueError('complete child ownership without root parameters required')
    combined = {}
    for name, child in children.items():
        ledger = owners[name]
        if type(ledger) is not ParameterLedger or ledger.schema != 'mora3.parameter-alias-inventory.v1':
            raise ValueError('supported child ledger required')
        declarations = {}
        for entry in ledger.parameters:
            for alias in entry.aliases:
                if alias in declarations:
                    raise ValueError('duplicate child alias')
                declarations[alias] = ParameterRole(entry.role, entry.axes)
        if parameter_ledger(child, declarations) != ledger:
            raise ValueError('stale child inventory')
        combined.update({name+'.'+alias:role for alias,role in declarations.items()})
    return parameter_ledger(model, combined)


def morphology_parameter_ledger(model) -> ParameterLedger:
    """Standalone native rank-one grayscale owner inventory."""
    return _morphology_parameter_ledger(model, identifier_axis=('image',))


def _morphology_parameter_ledger(model, *, identifier_axis) -> ParameterLedger:
    """Complete native grayscale foreground/affinity owner construction map.

    Fixed output coordinates preserve foreground/right/down semantics. Spatial
    convolutions and mean/broadcast context remain unqualified custom operators.
    No source authentication, forward, initialization or checkpoint mutation.
    """
    import hashlib
    import json
    from .morphology import MorphologyForeground, MorphologyInstance
    if type(model) not in (MorphologyForeground, MorphologyInstance):
        raise ValueError('exact native morphology owner required')
    if (set(model._modules) != {'pixels','backbone','head'}
            or model.normalization not in ('unsigned-dtype-max','image-minmax')):
        raise ValueError('native morphology configuration drift')
    w=model.backbone.feature.out_features
    if type(w) is not int or w<2 or w%2:
        raise ValueError('even native morphology width required')
    def finite(namespace,names):
        return Axis('finite',hashlib.sha256(json.dumps([namespace,names],
            separators=(',',':')).encode()).hexdigest())
    h=Axis('width','hidden'); shapes={}; roles={}
    def conv(prefix,module,cin,cout,k,padding,ia,oa):
        if (type(module) is not nn.Conv2d or set(dict(module.named_modules()))!={''}
                or (module.in_channels,module.out_channels,module.kernel_size,module.stride,
                    module.padding,module.dilation,module.groups,module.padding_mode,
                    module.transposed,module.output_padding)!=(cin,cout,(k,k),(1,1),
                        (padding,padding),(1,1),1,'zeros',False,(0,0))):
            raise ValueError('native morphology convolution drift')
        offsets=tuple(str(i) for i in range(-padding,k-padding))
        ky=finite('image.kernel-row-offsets.v1',offsets)
        kx=finite('image.kernel-column-offsets.v1',offsets)
        roles[prefix+'weight']=ParameterRole('custom',(oa,ia,ky,kx))
        roles[prefix+'bias']=ParameterRole('vector',(oa,))
        shapes[prefix+'weight']=(cout,cin,k,k); shapes[prefix+'bias']=(cout,)
    if type(model) is MorphologyInstance:
        if model.width!=w:
            raise ValueError('native instance width drift')
        pixels=image_encoder_parameter_roles(model.pixels,channel_axis=('intensity',))
        if model.pixels.enc0[0].out_channels!=w:
            raise ValueError('native pixel width mismatch')
        roles.update({'pixels.'+n:r for n,r in pixels.items()})
        labels=('foreground','right','down')
    else:
        modules=dict(model.pixels.named_modules(remove_duplicate=False))
        expected={'':nn.Sequential,'0':nn.Conv2d,'1':nn.GELU,'2':nn.Conv2d,'3':nn.GELU}
        if set(modules)!=set(expected) or any(type(modules[n]) is not t for n,t in expected.items()):
            raise ValueError('native foreground pixel structure drift')
        channel=finite('image.input-channels.v1',('intensity',))
        conv('pixels.0.',modules['0'],1,w,3,1,channel,h)
        conv('pixels.2.',modules['2'],w,w,3,1,h,h)
        labels=('foreground',)
    conv('head.',model.head,2*w,len(labels),1,0,
        Axis('width','image.concat-local-context'),finite('morphology.outputs.v1',labels))
    native=tokengt_parameter_ledger(model.backbone,
        feature_axis=tuple('hidden-coordinate:'+str(i) for i in range(w)),
        identifier_axis=identifier_axis)
    for entry in native.parameters:
        role=ParameterRole('hidden',(h,h)) if 'feature.weight' in entry.aliases else ParameterRole(entry.role,entry.axes)
        for alias in entry.aliases: roles['backbone.'+alias]=role
    params=dict(model.named_parameters(remove_duplicate=False))
    if any(n not in params or tuple(params[n].shape)!=s for n,s in shapes.items()):
        raise ValueError('native morphology parameter shape drift')
    return parameter_ledger(model,roles)


def cooperative_grayscale_parameter_ledger(model) -> ParameterLedger:
    """Complete explicit native shared image/response construction inventory.

    Rank-two padded image IDs reuse the response identifier namespace. This is
    alias/count metadata only; convolution and pooled context scaling remain
    unqualified. No source eligibility, pairing or runtime execution is inferred.
    """
    from .cooperative_training import CooperativeModel
    from .sciplex import SciPlexSharedModel
    from .morphology import MorphologyInstance
    from mora3.data.sciplex_admission import POLICY
    if (type(model) is not CooperativeModel
            or set(model._modules) != {'response','image'}
            or type(model.response) is not SciPlexSharedModel
            or type(model.image) is not MorphologyInstance
            or model.image.backbone is not model.response.backbone):
        raise ValueError('exact native shared grayscale cooperative owner required')
    n=model.response; w=n.edge.numel()
    native=dict(version=1,axis_ids=n.axis_ids,width=w,attention='vanilla',
        policy=POLICY,transform='log1p-counts',graph='context-to-observation-v1')
    expected=dict(version=1,architecture='native-shared-tokengt-v1',width=w,
        axis_ids=n.axis_ids,attention='vanilla',
        image_objective='union-bce+local-affinity-bce',response=native,
        identifier_policy='exact-one-node-zero-pad-to-two',shared_layers=2,
        pairing='none-inferred')
    if (n.config!=native or model.config!=expected
            or model.image.normalization!='image-minmax'
            or len(n.backbone.layers)!=2):
        raise ValueError('native cooperative configuration drift')
    return compose_owner_parameter_ledgers(model,{
        'response':response_parameter_ledger(n),
        'image':_morphology_parameter_ledger(model.image,
            identifier_axis=('context','observation'))})


def geometry_fusion_local_parameter_roles(model, *, identifier_axis: tuple[str, ...]) -> dict[str, ParameterRole]:
    """Local native geometry payload/incidence map, excluding shared transforms.

    Not a complete fusion owner inventory or a qualified scaling rule. Ordered
    identifier names are caller declarations, not biological identity. Positional
    pooling and hidden-plus-length concatenation remain custom. No forward,
    initialization, checkpoint conversion or source/task admission occurs.
    """
    import hashlib
    import json
    from .geometry_pixel_fusion import GeometryPixelFusion
    from .geometry_conditioning import GeometryConditioner
    from .payload import PayloadEncoder
    from .morphology import MorphologyInstance
    from mora3.tokenization.graph_payload import GraphPayloadCodec
    from mora3.tokenization.shortest_path import ByteTokenizer
    if (type(model) is not GeometryPixelFusion or set(model._modules)!={'image','geometry'}
            or type(model.image) is not MorphologyInstance
            or type(model.geometry) is not GeometryConditioner
            or set(model.geometry._modules)!={'payload','graph'}):
        raise ValueError('exact native geometry fusion structure required')
    if (type(identifier_axis) is not tuple or not identifier_axis
            or any(type(n) is not str or not n.strip() for n in identifier_axis)
            or len(set(identifier_axis))!=len(identifier_axis)):
        raise ValueError('immutable unique ordered identifier names required')
    c=model.geometry; w=model.image.width
    # Validate complete image construction using its actual finite incidence rank.
    rank=model.image.backbone.incidence.in_features//2
    _morphology_parameter_ledger(model.image,
        identifier_axis=tuple('image-coordinate:'+str(i) for i in range(rank)))
    expected=dict(version=1,architecture='native-full-geometry-shared-v1',
        id_width=len(identifier_axis),feature_width=w,width=w,
        shared_layers=len(model.image.backbone.layers),identifiers='exact-identity-zero-padded',
        pooling='scalar-positional-then-graph-token',discrete_geometry_gradient=False)
    if (c.config!=expected or model.config!=dict(version=1,
            architecture='native-additive-geometry-pixel-v1',geometry=expected,
            geometry_origin='frozen-native-checkpoint',training_admission=False)):
        raise ValueError('native geometry fusion configuration drift')
    for name in ('feature','role','graph_token','layers','final_norm'):
        if getattr(c.graph,name) is not getattr(model.image.backbone,name):
            raise ValueError('exact native shared transforms required')
    graph=tokengt_parameter_ledger(c.graph,
        feature_axis=tuple('hidden-coordinate:'+str(i) for i in range(w)),
        identifier_axis=identifier_axis)
    p=c.payload; t=ByteTokenizer(controls=('pad','node','edge'))
    if (type(p) is not PayloadEncoder or set(p._modules)!={'embedding','projection'}
            or type(p.embedding) is not nn.Embedding or type(p.projection) is not nn.Linear
            or type(p.codec) is not GraphPayloadCodec or type(p.codec.tokenizer) is not ByteTokenizer
            or p.codec.tokenizer.vocabulary!=t.vocabulary or p.codec.tokenizer._controls!=t._controls
            or p.offset!=t.size or p.pooling!='positional'
            or type(p.max_tokens) is not int or p.max_tokens<=0
            or p.embedding.num_embeddings!=517 or p.embedding.embedding_dim!=w
            or p.projection.in_features!=w+1 or p.projection.out_features!=w):
        raise ValueError('native geometry payload configuration drift')
    shapes={'embedding.weight':(517,w),'projection.weight':(w,w+1),'projection.bias':(w,)}
    params=dict(p.named_parameters(remove_duplicate=False))
    if set(params)!=set(shapes) or any(tuple(params[n].shape)!=s for n,s in shapes.items()):
        raise ValueError('native geometry payload parameter drift')
    raw=json.dumps(['geometry.byte-payload.v1',tuple(b.hex() for b in t.vocabulary),
                    tuple(t._controls),tuple(range(258))],separators=(',',':'))
    tokens=Axis('finite',hashlib.sha256(raw.encode()).hexdigest()); h=Axis('width','hidden')
    incidence=next(e for e in graph.parameters if 'incidence.weight' in e.aliases)
    return {'geometry.payload.embedding.weight':ParameterRole('embedding',(tokens,h)),
        'geometry.payload.projection.weight':ParameterRole('custom',(h,Axis('width','payload-hidden-plus-length'))),
        'geometry.payload.projection.bias':ParameterRole('vector',(h,)),
        'geometry.graph.incidence.weight':ParameterRole(incidence.role,incidence.axes)}


def geometry_fusion_parameter_ledger(model, *, identifier_axis: tuple[str, ...],
                                     image_identifier_axis: tuple[str, ...] | None = None) -> ParameterLedger:
    """Complete native fusion inventory with explicit shared-transform aliases.

    Image incidence defaults to its standalone coordinate namespace. An explicit
    image_identifier_axis declares parent coordinates for cooperative aliases;
    geometry retains the caller's separate ordered object namespace. Shared transforms have exactly the
    image owner's roles. Construction metadata does not qualify custom scaling,
    source eligibility, training or discrete geometry gradients.
    """
    local = geometry_fusion_local_parameter_roles(model, identifier_axis=identifier_axis)
    rank = model.image.backbone.incidence.in_features // 2
    if image_identifier_axis is None:
        image_identifier_axis = tuple('image-coordinate:'+str(i) for i in range(rank))
    if (type(image_identifier_axis) is not tuple or len(image_identifier_axis) != rank
            or any(type(n) is not str or not n.strip() for n in image_identifier_axis)
            or len(set(image_identifier_axis)) != rank):
        raise ValueError('immutable unique ordered image identifier names matching rank required')
    image = _morphology_parameter_ledger(model.image,
        identifier_axis=image_identifier_axis)
    roles = {}
    for entry in image.parameters:
        for alias in entry.aliases:
            roles['image.'+alias] = ParameterRole(entry.role, entry.axes)
            if alias.startswith('backbone.') and alias != 'backbone.incidence.weight':
                roles['geometry.graph.'+alias[len('backbone.'):]] = ParameterRole(entry.role, entry.axes)
    roles.update(local)
    return parameter_ledger(model, roles)


def molecule_profile_parameter_ledger(model, *, identifier_axis: tuple[str, ...]) -> ParameterLedger:
    """Complete native molecule/profile owner map with ordered assay axes.

    Source names are caller declarations, not admission/authentication. The
    existing lossy profile adapter and payload/graph custom operators remain
    unqualified for scaling. No forward, state, initialization or optimizer change.
    """
    import hashlib
    import json
    from .molecule_profile import SharedMoleculeProfileModel
    from .annotation_multihead import SharedAnnotationModel
    from .physical_trainer import SharedPhysicalModel
    if (type(model) is not SharedMoleculeProfileModel
            or set(model._modules) != {'shared', 'profile_input', 'output'}
            or type(model.shared) not in (SharedAnnotationModel, SharedPhysicalModel)
            or type(model.axes) is not dict
            or set(model.axes) != {'CellPainting', 'L1000'}):
        raise ValueError('exact native molecule/profile owner required')
    owner_ledger = (physical_parameter_ledger if type(model.shared) is SharedPhysicalModel
                    else annotation_parameter_ledger)
    native = owner_ledger(model.shared, identifier_axis=identifier_axis)
    roles = {'shared.'+alias: ParameterRole(entry.role, entry.axes)
             for entry in native.parameters for alias in entry.aliases}
    w = model.shared.payload.embedding.embedding_dim
    hidden = Axis('width', 'hidden')
    for group in ('profile_input', 'output'):
        modules = getattr(model, group)
        if type(modules) is not nn.ModuleDict or set(modules) != set(model.axes):
            raise ValueError('complete native profile projections required')
        for name, size in (('CellPainting', 1783), ('L1000', 977)):
            axis = model.axes[name]
            if (type(axis) is not tuple or len(axis) != size
                    or any(type(x) is not str or not x for x in axis)
                    or len(set(axis)) != size):
                raise ValueError('complete ordered unique profile axis required')
            digest = hashlib.sha256(json.dumps(['molecule-profile.source-axis.v1', name, axis],
                separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()
            finite = Axis('finite', digest)
            projection = modules[name]
            incoming = group == 'profile_input'
            out_size, in_size = (w, size) if incoming else (size, w)
            if (type(projection) is not nn.Linear or projection._modules
                    or set(projection._parameters) != {'weight', 'bias'}
                    or projection.in_features != in_size or projection.out_features != out_size
                    or projection.weight is None or projection.bias is None
                    or tuple(projection.weight.shape) != (out_size, in_size)
                    or tuple(projection.bias.shape) != (out_size,)):
                raise ValueError('native profile linear configuration/shape drift')
            axes = (hidden, finite) if incoming else (finite, hidden)
            prefix = group+'.'+name
            roles[prefix+'.weight'] = ParameterRole('input' if incoming else 'output', axes)
            roles[prefix+'.bias'] = ParameterRole('vector', (axes[0],))
    return parameter_ledger(model, roles)


def selfies_decoder_local_parameter_roles(model) -> dict[str, ParameterRole]:
    """Local native SELFIES vocabulary/readout and packed GRU declarations.

    Registered profile owner is excluded and requires separate composition.
    Reset/update/new packing has a distinct 3*hidden axis, not a hidden axis.
    Recurrent gates remain custom and refuse width-family qualification. Token
    names are declarations, not source authentication or generation admission.
    No forward, initialization, checkpoint, gradient or optimizer mutation.
    """
    import hashlib
    import json
    from .profile_decoder import ProfileSelfiesDecoder, SelfiesCodec
    from .molecule_profile import SharedMoleculeProfileModel
    from .annotation_multihead import SharedAnnotationModel
    from .physical_trainer import SharedPhysicalModel

    if (type(model) is not ProfileSelfiesDecoder
            or type(model.profiles) is not SharedMoleculeProfileModel
            or type(model.profiles.shared) not in (SharedAnnotationModel, SharedPhysicalModel)
            or type(model.codec) is not SelfiesCodec
            or type(model.max_tokens) is not int or model.max_tokens < 3):
        raise ValueError('exact native SELFIES decoder and profile owner required')
    modules = {n:m for n,m in model.named_modules(remove_duplicate=False)
               if n != 'profiles' and not n.startswith('profiles.')}
    expected = {'':ProfileSelfiesDecoder,'embedding':nn.Embedding,'rnn':nn.GRU,'output':nn.Linear}
    if set(modules) != set(expected) or any(type(modules[n]) is not t for n,t in expected.items()):
        raise ValueError('native decoder local module ownership required')
    c = model.codec
    if (type(c.symbols) is not tuple or len(c.symbols) < 4
            or any(type(s) is not str or not s for s in c.symbols)
            or len(set(c.symbols)) != len(c.symbols)
            or c.symbols[:3] != ('<PAD>','<BOS>','<EOS>')
            or type(c.size) is not int or c.size != len(c.symbols)
            or type(c.index) is not dict or c.index != {s:i for i,s in enumerate(c.symbols)}
            or any(type(v) is not int for v in c.index.values())
            or any(type(v) is not int for v in (c.pad,c.bos,c.eos))
            or (c.pad,c.bos,c.eos) != (0,1,2)):
        raise ValueError('complete immutable ordered SELFIES vocabulary required')
    w = model.profiles.shared.payload.embedding.embedding_dim
    e,r,o = model.embedding,model.rnn,model.output
    if (type(w) is not int or w < 1 or e.num_embeddings != c.size or e.embedding_dim != w
            or e.padding_idx != 0 or e.max_norm is not None or e.norm_type != 2.
            or e.scale_grad_by_freq or e.sparse
            or r.input_size != w or r.hidden_size != w or r.num_layers != 1
            or r.bias is not True or r.batch_first is not True or r.bidirectional is not False
            or r.dropout != 0 or r.mode != 'GRU'
            or o.in_features != w or o.out_features != c.size):
        raise ValueError('native SELFIES embedding/GRU/readout configuration drift')
    h = Axis('width','hidden')
    gate = Axis('width','selfies-gru-reset-update-new-packed')
    v = Axis('finite',hashlib.sha256(json.dumps(
        ['selfies.decoder-vocabulary.v1',c.symbols],ensure_ascii=True,
        separators=(',',':')).encode()).hexdigest())
    roles = {'embedding.weight':ParameterRole('embedding',(v,h)),
             'output.weight':ParameterRole('output',(v,h)),
             'output.bias':ParameterRole('vector',(v,))}
    shapes = {'embedding.weight':(c.size,w),'output.weight':(c.size,w),'output.bias':(c.size,)}
    for name in ('weight_ih_l0','weight_hh_l0','bias_ih_l0','bias_hh_l0'):
        matrix = name.startswith('weight')
        roles['rnn.'+name] = ParameterRole('custom',(gate,h) if matrix else (gate,))
        shapes['rnn.'+name] = (3*w,w) if matrix else (3*w,)
    params = {n:p for n,p in model.named_parameters(remove_duplicate=False)
              if not n.startswith('profiles.')}
    if (set(params) != set(shapes)
            or any(tuple(params[n].shape) != shape for n,shape in shapes.items())
            or any(m._buffers for m in modules.values())):
        raise ValueError('native decoder local parameter shape/ownership drift')
    # Check tied local objects now; composing the excluded owner is a separate gate.
    parameter_ledger(nn.ModuleDict({'embedding':e,'rnn':r,'output':o}),roles)
    return roles


def selfies_decoder_parameter_ledger(model, *, identifier_axis: tuple[str, ...]) -> ParameterLedger:
    """Complete native decoder/profile inventory with identity-based alias checks.

    Ordered assay/vocabulary identities remain declarations, not source admission.
    Recurrent/custom operators remain unqualified for width scaling. This reads
    metadata only and does not alter inference, initialization or training state.
    """
    roles = selfies_decoder_local_parameter_roles(model)
    if model.profiles._buffers:
        raise ValueError('native profile root buffer ownership drift')
    native = molecule_profile_parameter_ledger(model.profiles, identifier_axis=identifier_axis)
    roles.update({'profiles.'+alias: ParameterRole(entry.role, entry.axes)
                  for entry in native.parameters for alias in entry.aliases})
    return parameter_ledger(model, roles)


def tox21_parameter_ledger(model, *, identifier_axis: tuple[str, ...]) -> ParameterLedger:
    """Complete native Tox21 owner, distinct ordered value/mask and call axes.

    Metadata inventory only: no source admission, scaling, initialization or
    computation changes. Shared custom operators remain unqualified.
    """
    import hashlib
    import json
    from .tox21 import SharedTox21Model
    from mora3.data.tox21 import ASSAYS
    from .annotation_multihead import SharedAnnotationModel
    from .physical_trainer import SharedPhysicalModel
    if (type(model) is not SharedTox21Model
            or set(model._modules) != {'shared', 'call_input', 'output'}
            or model._buffers or model._parameters
            or type(model.shared) not in (SharedAnnotationModel, SharedPhysicalModel)):
        raise ValueError('exact native Tox21 owner required')
    owner = (physical_parameter_ledger if type(model.shared) is SharedPhysicalModel
             else annotation_parameter_ledger)
    native = owner(model.shared, identifier_axis=identifier_axis)
    roles = {'shared.'+a: ParameterRole(e.role, e.axes)
             for e in native.parameters for a in e.aliases}
    def finite(value):
        return Axis('finite', hashlib.sha256(json.dumps(value,
            separators=(',', ':'), ensure_ascii=True).encode()).hexdigest())
    calls = finite(['tox21.source-call.v1', ASSAYS])
    conditions = finite(['tox21.condition-value-mask.v1',
        tuple(('value', a) for a in ASSAYS)+tuple(('known', a) for a in ASSAYS)])
    hidden = Axis('width', 'hidden')
    w = model.shared.payload.embedding.embedding_dim
    for name, incoming, n, axes in (
            ('call_input', True, 24, (hidden, conditions)),
            ('output', False, 12, (calls, hidden))):
        layer = getattr(model, name)
        out_size, in_size = (w, n) if incoming else (n, w)
        if (type(layer) is not nn.Linear or layer._modules or layer._buffers
                or set(layer._parameters) != {'weight', 'bias'}
                or layer.in_features != in_size or layer.out_features != out_size
                or layer.weight is None or layer.bias is None
                or tuple(layer.weight.shape) != (out_size, in_size)
                or tuple(layer.bias.shape) != (out_size,)):
            raise ValueError('native Tox21 projection configuration/shape drift')
        roles[name+'.weight'] = ParameterRole('input' if incoming else 'output', axes)
        roles[name+'.bias'] = ParameterRole('vector', (axes[0],))
    return parameter_ledger(model, roles)


def nucleotide_decoder_local_parameter_roles(model) -> dict[str, ParameterRole]:
    """Local native nucleotide vocabulary/readout and packed GRU declarations.

    Registered annotation owner is excluded and requires separate composition.
    Reset/update/new packing has a distinct 3*hidden axis, not a hidden axis.
    Recurrent gates remain custom and refuse width-family qualification. Token
    names are declarations, not source authentication or generation admission.
    No forward, initialization, checkpoint, gradient or optimizer mutation.
    """
    return _nucleotide_decoder_local_parameter_roles(model, rfam=False)


def _nucleotide_decoder_local_parameter_roles(model, *, rfam):
    """Shared exact validator for the two explicitly supported native roots."""
    import hashlib
    import json
    from .nucleotide_decoder import NucleotideDecoder, NucleotideCodec
    from .rfam_family import RfamFamilyModel
    from .annotation_multihead import SharedAnnotationModel
    from .physical_trainer import SharedPhysicalModel
    from mora3.data.nucleotide_prior import ALPHABET

    if (type(model) is not (RfamFamilyModel if rfam else NucleotideDecoder)
            or type(model.shared) not in (SharedAnnotationModel, SharedPhysicalModel)
            or type(model.codec) is not NucleotideCodec):
        raise ValueError('exact native nucleotide decoder and annotation owner required')
    modules = {n:m for n,m in model.named_modules(remove_duplicate=False)
               if n != 'shared' and not n.startswith('shared.')
               and not (rfam and (n == 'family_output' or n.startswith('family_output.')))}
    expected = {'':RfamFamilyModel if rfam else NucleotideDecoder,'embedding':nn.Embedding,'rnn':nn.GRU,'output':nn.Linear}
    if set(modules) != set(expected) or any(type(modules[n]) is not t for n,t in expected.items()):
        raise ValueError('native decoder local module ownership required')
    c = model.codec
    if (type(c.symbols) is not tuple or len(c.symbols) < 4
            or any(type(s) is not str or not s for s in c.symbols)
            or len(set(c.symbols)) != len(c.symbols)
            or c.symbols != ('<PAD>','<BOS>','<EOS>') + tuple(ALPHABET)
            or type(c.size) is not int or c.size != len(c.symbols)
            or type(c.index) is not dict or c.index != {s:i for i,s in enumerate(c.symbols)}
            or any(type(v) is not int for v in c.index.values())
            or any(type(v) is not int for v in (c.pad,c.bos,c.eos))
            or (c.pad,c.bos,c.eos) != (0,1,2)):
        raise ValueError('complete immutable ordered nucleotide vocabulary required')
    w = model.shared.payload.embedding.embedding_dim
    e,r,o = model.embedding,model.rnn,model.output
    if (type(w) is not int or w < 1 or e.num_embeddings != c.size or e.embedding_dim != w
            or e.padding_idx != 0 or e.max_norm is not None or e.norm_type != 2.
            or e.scale_grad_by_freq or e.sparse
            or r.input_size != w or r.hidden_size != w or r.num_layers != 1
            or r.bias is not True or r.batch_first is not True or r.bidirectional is not False
            or r.dropout != 0 or r.mode != 'GRU'
            or o.in_features != w or o.out_features != c.size):
        raise ValueError('native nucleotide embedding/GRU/readout configuration drift')
    h = Axis('width','hidden')
    gate = Axis('width','nucleotide-gru-reset-update-new-packed')
    v = Axis('finite',hashlib.sha256(json.dumps(
        ['nucleotide.decoder-vocabulary.v1',c.symbols],ensure_ascii=True,
        separators=(',',':')).encode()).hexdigest())
    roles = {'embedding.weight':ParameterRole('embedding',(v,h)),
             'output.weight':ParameterRole('output',(v,h)),
             'output.bias':ParameterRole('vector',(v,))}
    shapes = {'embedding.weight':(c.size,w),'output.weight':(c.size,w),'output.bias':(c.size,)}
    for name in ('weight_ih_l0','weight_hh_l0','bias_ih_l0','bias_hh_l0'):
        matrix = name.startswith('weight')
        roles['rnn.'+name] = ParameterRole('custom',(gate,h) if matrix else (gate,))
        shapes['rnn.'+name] = (3*w,w) if matrix else (3*w,)
    params = {n:p for n,p in model.named_parameters(remove_duplicate=False)
              if not n.startswith('shared.') and not (rfam and n.startswith('family_output.'))}
    if (set(params) != set(shapes)
            or any(tuple(params[n].shape) != shape for n,shape in shapes.items())
            or any(m._buffers for m in modules.values())):
        raise ValueError('native decoder local parameter shape/ownership drift')
    # Check tied local objects now; composing the excluded owner is a separate gate.
    parameter_ledger(nn.ModuleDict({'embedding':e,'rnn':r,'output':o}),roles)
    return roles



def nucleotide_decoder_parameter_ledger(model, *, identifier_axis: tuple[str, ...]) -> ParameterLedger:
    """Complete native nucleotide/annotation owner inventory, without scaling.

    Both exact annotation owners retain their full native validators. Identity-
    equal tensors count once and all alias declarations must agree. GRU and
    other custom operators remain unqualified for width-family transfer. This
    metadata-only inventory supplies no source admission or neural computation.
    """
    from .physical_trainer import SharedPhysicalModel
    roles = nucleotide_decoder_local_parameter_roles(model)
    if any(module._buffers for module in model.shared.modules()):
        raise ValueError('native nucleotide shared buffer ownership drift')
    owner = (physical_parameter_ledger if type(model.shared) is SharedPhysicalModel
             else annotation_parameter_ledger)
    native = owner(model.shared, identifier_axis=identifier_axis)
    roles.update({'shared.'+alias: ParameterRole(entry.role, entry.axes)
                  for entry in native.parameters for alias in entry.aliases})
    return parameter_ledger(model, roles)


def rfam_family_local_parameter_roles(model, *, family_axis: tuple[str, ...],
                                      manifest_sha256: str) -> dict[str, ParameterRole]:
    """Declare only the exact native Rfam readout, bound to a caller registry.

    Excludes inherited nucleotide/annotation parameters; complete owner and
    cross-owner aliases require separate composition. The manifest digest is a
    declaration, not source authentication, admission or scaling qualification.
    Reads metadata only; no forward, initialization or state mutation.
    """
    import hashlib
    import json
    from .rfam_family import RfamFamilyModel
    if (type(family_axis) is not tuple or not family_axis
            or any(type(x) is not str or re.fullmatch(r'RF[0-9]{5}', x) is None
                   for x in family_axis)
            or family_axis != tuple(sorted(set(family_axis)))
            or type(manifest_sha256) is not str
            or re.fullmatch('[0-9a-f]{64}', manifest_sha256) is None):
        raise ValueError('complete ordered family axis and manifest digest required')
    if (type(model) is not RfamFamilyModel
            or type(model.family_axis) is not tuple or model.family_axis != family_axis
            or model.manifest_sha256 != manifest_sha256
            or set(model._modules) != {'shared','embedding','rnn','output','family_output'}
            or model._parameters or model._buffers):
        raise ValueError('exact native Rfam local owner/registry required')
    layer = model.family_output
    w = model.embedding.embedding_dim
    n = len(family_axis)
    if (type(w) is not int or w < 1 or type(layer) is not nn.Linear
            or layer._modules or layer._buffers
            or set(layer._parameters) != {'weight','bias'}
            or layer.in_features != w or layer.out_features != n
            or layer.weight is None or layer.bias is None
            or tuple(layer.weight.shape) != (n,w) or tuple(layer.bias.shape) != (n,)):
        raise ValueError('native Rfam local projection configuration/shape drift')
    axis = Axis('finite', hashlib.sha256(json.dumps(
        ['rfam.source-family-readout.v1',manifest_sha256,family_axis],
        separators=(',', ':'),ensure_ascii=True).encode()).hexdigest())
    roles = {'family_output.weight':ParameterRole('output',(axis,Axis('width','hidden'))),
             'family_output.bias':ParameterRole('vector',(axis,))}
    parameter_ledger(nn.ModuleDict({'family_output':layer}),roles)
    return roles


def rfam_family_parameter_ledger(model, *, identifier_axis: tuple[str, ...],
                                 family_axis: tuple[str, ...],
                                 manifest_sha256: str) -> ParameterLedger:
    """Complete exact Rfam/nucleotide/annotation inventory, without computation.

    Explicit registry declarations bind the readout; source authentication and
    admission remain external. All inherited native checks and identity-based
    alias conflicts remain active. Custom operators do not qualify for scaling.
    """
    from .physical_trainer import SharedPhysicalModel
    roles = rfam_family_local_parameter_roles(
        model, family_axis=family_axis, manifest_sha256=manifest_sha256)
    roles.update(_nucleotide_decoder_local_parameter_roles(model, rfam=True))
    if any(module._buffers for module in model.shared.modules()):
        raise ValueError('native Rfam shared buffer ownership drift')
    owner = (physical_parameter_ledger if type(model.shared) is SharedPhysicalModel
             else annotation_parameter_ledger)
    native = owner(model.shared, identifier_axis=identifier_axis)
    roles.update({'shared.'+alias: ParameterRole(entry.role, entry.axes)
                  for entry in native.parameters for alias in entry.aliases})
    return parameter_ledger(model, roles)


def goa_local_parameter_roles(model, *, axis_ids: tuple[str, ...],
                              source_binding: dict) -> dict[str, ParameterRole]:
    """Exact native GOA local readout axes, including ordered pool/policy identity.

    The caller supplies an independent full binding snapshot. This metadata-only
    declaration does not authenticate source files, admit packets, execute the
    head or qualify width scaling. Unknown/quarantined labels retain the native
    objective semantics; no annotation-conditioned generation is introduced.
    """
    import hashlib
    import json
    from .uniprot_go import NativeGOAHead
    if (type(axis_ids) is not tuple or not axis_ids
            or any(type(x) is not str or re.fullmatch(r'GO:[0-9]{7}',x) is None
                   for x in axis_ids) or len(set(axis_ids)) != len(axis_ids)
            or type(source_binding) is not dict
            or set(source_binding) != {'version','policy','axis_ids','packets','generation_eligible'}
            or type(source_binding['version']) is not int or source_binding['version'] != 1
            or source_binding['policy'] != NativeGOAHead.policy
            or type(source_binding['axis_ids']) is not tuple
            or source_binding['axis_ids'] != axis_ids
            or source_binding['generation_eligible'] is not False
            or type(source_binding['packets']) is not list or not source_binding['packets']):
        raise ValueError('complete native ordered GOA axis and source policy binding required')
    for receipt in source_binding['packets']:
        if (type(receipt) is not dict
                or set(receipt) != {'packet_sha256','subject','status','objective'}
                or type(receipt['packet_sha256']) is not str
                or re.fullmatch('[0-9a-f]{64}',receipt['packet_sha256']) is None
                or type(receipt['subject']) is not str
                or not receipt['subject'].startswith('UniProtKB:')
                or not receipt['subject'][len('UniProtKB:'):]
                or receipt['status'] not in ('applicable','quarantined_polarity_conflict')
                or type(receipt['objective']) is not dict
                or receipt['objective'].get('policy') != NativeGOAHead.policy
                or receipt['objective'].get('status') != receipt['status']
                or receipt['objective'].get('unknown_is_negative') is not False):
            raise ValueError('native GOA packet receipt/policy required')
    if (type(model) is not NativeGOAHead or model.policy != NativeGOAHead.policy
            or type(model.axis_ids) is not tuple or model.axis_ids != axis_ids
            or model.get_extra_state() != source_binding
            or type(model._packet_hashes) is not frozenset
            or model._packet_hashes != frozenset(r['packet_sha256'] for r in source_binding['packets'])
            or set(model._modules) != {'output'} or model._parameters or model._buffers):
        raise ValueError('exact native GOA local owner/source binding required')
    layer = model.output
    n = len(axis_ids)
    if (type(layer) is not nn.Linear or layer._modules or layer._buffers
            or set(layer._parameters) != {'weight','bias'}
            or type(layer.in_features) is not int or layer.in_features < 1
            or type(layer.out_features) is not int or layer.out_features != n
            or layer.weight is None or layer.bias is None
            or tuple(layer.weight.shape) != (n,layer.in_features)
            or tuple(layer.bias.shape) != (n,)):
        raise ValueError('native GOA local projection configuration/shape drift')
    try:
        raw = json.dumps(['goa.source-pool-readout.v1',source_binding],
                         sort_keys=True,separators=(',', ':'),ensure_ascii=True,allow_nan=False)
        live = json.dumps(['goa.source-pool-readout.v1',model.get_extra_state()],
                          sort_keys=True,separators=(',', ':'),ensure_ascii=True,allow_nan=False)
        if live != raw:
            raise ValueError('native GOA binding value/type drift')
    except (TypeError,ValueError) as exc:
        raise ValueError('serializable finite native GOA source binding required') from exc
    axis = Axis('finite',hashlib.sha256(raw.encode()).hexdigest())
    roles = {'output.weight':ParameterRole('output',(axis,Axis('width','hidden'))),
             'output.bias':ParameterRole('vector',(axis,))}
    parameter_ledger(model,roles)
    return roles


def goa_cooperative_parameter_ledger(model, *, owners: Mapping[str, ParameterLedger],
                                     axis_ids: tuple[str, ...],
                                     source_binding: dict) -> ParameterLedger:
    """Compose explicit inherited inventories and the exact native GOA readout.

    Inherited ledgers are caller declarations, revalidated against every current
    alias by the generic composer. They do not authenticate sources or qualify
    scaling. GOA retains its complete local pool/policy validator. This function
    only inventories metadata; admission, objectives and checkpoints are unchanged.
    """
    from .goa_cooperative import GOACooperativeModel
    expected = {'native','annotation','flow','residue','profile','decoder',
                'tox21','nucleotide','rfam'}
    if (type(model) is not GOACooperativeModel or not isinstance(owners, Mapping)
            or set(owners) != expected or set(model._modules) != expected | {'goa'}
            or model._parameters or any(m._buffers for m in model.modules())):
        raise ValueError('exact native GOA root and complete inherited owners required')
    local = parameter_ledger(model.goa, goa_local_parameter_roles(
        model.goa, axis_ids=axis_ids, source_binding=source_binding))
    return compose_owner_parameter_ledgers(model, dict(owners, goa=local))


def latent_reconstruction_local_parameter_roles(model, *, latent_axis: tuple[str, ...],
                                                 decoder, configuration: dict) -> dict[str, ParameterRole]:
    """Inventory the cooperative residual W z with explicit finite latent axes.

    The decoder is parent-owned and must be the identical unregistered object.
    Configuration is a caller snapshot, not source/archive authentication. Latent
    coordinates are declarations, not biological features or density estimates.
    No neural execution or scaling qualification is performed.
    """
    import hashlib
    import json
    from .profile_latent_decoder import ProfileLatentSelfiesDecoder
    if (type(model) is not ProfileLatentSelfiesDecoder
            or model.decoder is not decoder
            or set(model._modules) != {'latent_projection'}
            or model._parameters or model._buffers
            or type(model.dimension) is not int or model.dimension < 1
            or type(model.horizon) is not int or model.horizon < 1
            or type(latent_axis) is not tuple or len(latent_axis) != model.dimension
            or any(type(x) is not str or not x for x in latent_axis)
            or len(set(latent_axis)) != len(latent_axis)
            or type(configuration) is not dict):
        raise ValueError('exact shared latent owner, decoder, coordinates and binding required')
    selfies_decoder_local_parameter_roles(decoder)
    policy=model._policy
    expected={'version','dimension','horizon','hidden','context_width','source_binding',
              'profiles','kernels','stopping','normalizer','state_space'}
    if (type(policy) is not dict or set(policy) != expected
            or any(type(policy[k]) is not int or policy[k] < 1
                   for k in ('version','dimension','horizon','hidden','context_width'))
            or policy['version'] != 1 or policy['dimension'] != model.dimension
            or policy['horizon'] != model.horizon
            or type(policy['source_binding']) is not dict or not policy['source_binding']
            or policy['profiles'] != decoder.profiles.get_extra_state()
            or policy['kernels'] != 'concatenated-context-time-tanh-diagonal-Gaussian-v1'
            or policy['stopping'] != 'optional-Bernoulli-forced-origin-and-horizon'
            or policy['normalizer'] != 'conditional-linear-logZ'
            or policy['state_space'] != 'time-layered-Euclidean-latent'):
        raise ValueError('native latent policy configuration required')
    w=decoder.embedding.embedding_dim
    layer=model.latent_projection
    if (policy['context_width'] != w or type(layer) is not nn.Linear
            or layer._modules or layer._buffers
            or set(layer._parameters) != {'weight','bias'} or layer.bias is not None
            or type(layer.in_features) is not int or layer.in_features != model.dimension
            or type(layer.out_features) is not int or layer.out_features != w
            or layer.weight is None or tuple(layer.weight.shape) != (w,model.dimension)):
        raise ValueError('native bias-free latent residual projection required')
    try:
        encode=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
        if encode(configuration) != encode(model.get_extra_state()):
            raise ValueError('latent configuration snapshot drift')
        raw=encode(['shared-latent-residual-coordinates.v1',latent_axis,
                    policy['dimension'],policy['horizon'],policy['source_binding']])
    except (TypeError,ValueError) as exc:
        raise ValueError('finite serializable latent configuration required') from exc
    axis=Axis('finite',hashlib.sha256(raw.encode()).hexdigest())
    roles={'latent_projection.weight':ParameterRole('input',(Axis('width','hidden'),axis))}
    parameter_ledger(model,roles)
    return roles


def reconstruction_cooperative_parameter_ledger(model, *, owners: Mapping[str, ParameterLedger],
        latent_axis: tuple[str, ...], configuration: dict,
        geometry_identifier_axis: tuple[str, ...] | None = None,
        image_identifier_axis: tuple[str, ...] | None = None) -> ParameterLedger:
    """Compose exact reconstruction roots with explicit inherited declarations.

    The branch validates its parent-owned decoder and caller configuration anew.
    Geometry roots require separate object and parent image coordinate declarations;
    incompatible shared aliases still refuse in the generic composer. This is a
    read-only inventory, not archive authentication, admission or scaling proof.
    """
    from .reconstruction_cooperative import (ReconstructionCooperativeModel,
                                             GeometryReconstructionCooperativeModel)
    geometry = type(model) is GeometryReconstructionCooperativeModel
    inherited = {'native','annotation','flow','residue','profile','decoder',
                 'tox21','nucleotide','rfam','goa'}
    children = inherited | {'reconstruction'} | ({'geometry_fusion'} if geometry else set())
    architecture = ('unpaired-native-fourteen-objective-geometry-interface-v1' if geometry
                    else 'unpaired-native-GOA-fixed-profile-reconstruction-interface-v1')
    if (type(model) not in (ReconstructionCooperativeModel,GeometryReconstructionCooperativeModel)
            or not isinstance(owners, Mapping) or set(owners) != inherited
            or set(model._modules) != children or model._parameters
            or any(m._buffers for m in model.modules())
            or type(model.configuration) is not dict
            or model.configuration.get('architecture') != architecture
            or model.configuration.get('reconstruction_ownership') !=
                'parent-decoder-profile;branch-projection-only-v1'):
        raise ValueError('exact reconstruction root and complete inherited owners required')
    local = parameter_ledger(model.reconstruction,
        latent_reconstruction_local_parameter_roles(model.reconstruction,
            latent_axis=latent_axis,decoder=model.decoder,configuration=configuration))
    complete = dict(owners,reconstruction=local)
    if geometry:
        if (geometry_identifier_axis is None or image_identifier_axis is None
                or model.geometry_fusion.image is not model.native.image
                or model.configuration.get('geometry_schedule') != 'interface-only-no-adoption'
                or model.configuration.get('geometry_pairing') != 'none-inferred'
                or model.configuration.get('geometry') != model.geometry_fusion.get_extra_state()):
            raise ValueError('explicit shared geometry owner and coordinates required')
        complete['geometry_fusion'] = geometry_fusion_parameter_ledger(
            model.geometry_fusion,identifier_axis=geometry_identifier_axis,
            image_identifier_axis=image_identifier_axis)
    elif geometry_identifier_axis is not None or image_identifier_axis is not None:
        raise ValueError('geometry coordinates require exact geometry reconstruction root')
    return compose_owner_parameter_ledgers(model,complete)


def admet_parameter_ledger(model, *, identifier_axis: tuple[str, ...]) -> ParameterLedger:
    """Read-only shared ADMET inventory with finite native partition/output axes.

    Contract identities include endpoint definition, assay, chemistry class and
    native unit. Gaussian rows are mean then raw scale (before softplus), not
    calibrated uncertainties. This grants neither source admission nor muP
    qualification; existing custom operators remain unqualified for scaling.
    """
    import hashlib
    import json
    from .admet_model import SharedAdmetModel
    from .admet_objectives import AdmetObjectiveContract
    from .annotation_multihead import SharedAnnotationModel
    from .physical_trainer import SharedPhysicalModel
    if (type(model) is not SharedAdmetModel
            or set(model._modules) != {'shared', 'heads'}
            or model._parameters or model._buffers
            or type(model.shared) not in (SharedAnnotationModel, SharedPhysicalModel)
            or type(model.contracts) is not dict or not model.contracts
            or type(model.heads) is not nn.ModuleDict
            or model.heads._parameters or model.heads._buffers
            or tuple(model.heads) != tuple(model.contracts)):
        raise ValueError('exact native ADMET owner and ordered partitions required')
    owner_ledger = (physical_parameter_ledger if type(model.shared) is SharedPhysicalModel
                    else annotation_parameter_ledger)
    native = owner_ledger(model.shared, identifier_axis=identifier_axis)
    roles = {'shared.'+alias: ParameterRole(entry.role, entry.axes)
             for entry in native.parameters for alias in entry.aliases}
    hidden = Axis('width', 'hidden')
    width = model.shared.payload.embedding.embedding_dim
    for key, contract in model.contracts.items():
        if (type(contract) is not AdmetObjectiveContract
                or contract.molecule_class is None
                or contract.definition.applicability_for(contract.molecule_class) != 'applicable'
                or contract.definition.objective_kind not in ('binary', 'regression')
                or key != contract.partition_key):
            raise ValueError('resolved native endpoint/class/assay/unit partition required')
        contract._check(contract.definition.objective_kind, contract.unit)
        outputs = (('logit',) if contract.definition.objective_kind == 'binary'
                   else ('mean', 'raw_scale'))
        digest = hashlib.sha256(json.dumps(['admet.partition-output.v1', key, outputs],
            separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()
        axis = Axis('finite', digest)
        layer = model.heads[key]
        count = len(outputs)
        if (type(layer) is not nn.Linear or layer._modules or layer._buffers
                or set(layer._parameters) != {'weight', 'bias'}
                or layer.in_features != width or layer.out_features != count
                or layer.weight is None or layer.bias is None
                or tuple(layer.weight.shape) != (count, width)
                or tuple(layer.bias.shape) != (count,)):
            raise ValueError('native ADMET head configuration/shape drift')
        roles['heads.'+key+'.weight'] = ParameterRole('output', (axis, hidden))
        roles['heads.'+key+'.bias'] = ParameterRole('vector', (axis,))
    return parameter_ledger(model, roles)


def lincs_parameter_ledger(model, *, identifier_axis: tuple[str, ...]) -> ParameterLedger:
    """Exact complete native LINCS owners, with source-bound finite coordinates.

    Source axes are validated against the reader's pinned complete schemas. This
    is parameter metadata only: no source/task admission, scaling qualification,
    forward call, checkpoint migration or state/RNG mutation is performed.
    Observation masks and values share ordered coordinates but remain separate
    parameters. Compatible aliases are counted once by the generic ledger.
    """
    import hashlib
    import json
    from mora3.data.lincs_native import _axis
    from .lincs_model import SharedLincsModel
    from .annotation_multihead import SharedAnnotationModel
    from .physical_trainer import SharedPhysicalModel
    modalities = {'CellPainting', 'L1000'}
    if (type(model) is not SharedLincsModel
            or set(model._modules) != {'shared','profile_input','profile_mask','output'}
            or model._parameters or model._buffers
            or type(model.shared) not in (SharedAnnotationModel, SharedPhysicalModel)
            or type(model.axes) is not dict or set(model.axes) != modalities
            or type(model.source_axes) is not dict or set(model.source_axes) != modalities):
        raise ValueError('exact native LINCS owner and complete source axes required')
    owner_ledger = (physical_parameter_ledger if type(model.shared) is SharedPhysicalModel
                    else annotation_parameter_ledger)
    native = owner_ledger(model.shared, identifier_axis=identifier_axis)
    roles = {'shared.'+alias: ParameterRole(entry.role, entry.axes)
             for entry in native.parameters for alias in entry.aliases}
    hidden = Axis('width','hidden')
    width = model.shared.payload.embedding.embedding_dim
    for name in ('profile_input','profile_mask','output'):
        container = getattr(model,name)
        if (type(container) is not nn.ModuleDict or set(container) != modalities
                or container._parameters or container._buffers):
            raise ValueError('exact native LINCS affine containers required')
    for modality in sorted(modalities):
        axis, _ = _axis(model.source_axes[modality], modality)
        if type(model.axes[modality]) is not tuple or model.axes[modality] != axis:
            raise ValueError('ordered full native LINCS axis drift')
        digest = hashlib.sha256(json.dumps(
            ['lincs.native-source-axis.v1',modality,model.source_axes[modality]],
            sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
        finite = Axis('finite',digest)
        count = len(axis)
        for name in ('profile_input','profile_mask','output'):
            layer = getattr(model,name)[modality]
            output = name == 'output'
            a,b = (width,count) if output else (count,width)
            bias = name != 'profile_mask'
            if (type(layer) is not nn.Linear or layer._modules or layer._buffers
                    or set(layer._parameters) != {'weight','bias'}
                    or (layer.in_features,layer.out_features) != (a,b)
                    or layer.weight is None or tuple(layer.weight.shape) != (b,a)
                    or (bias and (layer.bias is None or tuple(layer.bias.shape) != (b,)))
                    or (not bias and layer.bias is not None)):
                raise ValueError('native LINCS affine configuration/shape drift')
            prefix = name+'.'+modality
            roles[prefix+'.weight'] = ParameterRole('output' if output else 'input',
                (finite,hidden) if output else (hidden,finite))
            if bias:
                roles[prefix+'.bias'] = ParameterRole('vector',(finite if output else hidden,))
    return parameter_ledger(model,roles)


def lincs_cooperative_parameter_ledger(model, *, owners: Mapping[str, ParameterLedger],
        identifier_axis: tuple[str, ...], latent_axis: tuple[str, ...], configuration: dict,
        geometry_identifier_axis: tuple[str, ...],
        image_identifier_axis: tuple[str, ...]) -> ParameterLedger:
    """Explicit native LINCS extension of the complete geometry/reconstruction root.

    Inherited inventories remain caller declarations revalidated by composition.
    New branch inventories bind native source coordinates and exact shared owner
    identity. No source admission, numerical execution, scaling qualification or
    historical checkpoint adoption is implied by this read-only metadata API.
    """
    from .lincs_cooperative import LincsCooperativeModel
    from .lincs_model import SharedLincsModel
    inherited = {'native','annotation','flow','residue','profile','decoder',
                 'tox21','nucleotide','rfam','goa'}
    if (type(model) is not LincsCooperativeModel
            or not isinstance(owners, Mapping) or set(owners) != inherited
            or set(model._modules) != inherited | {'reconstruction','geometry_fusion','lincs'}
            or model._parameters or any(m._buffers for m in model.modules())
            or type(model.configuration) is not dict
            or model.configuration.get('architecture') != 'unpaired-native-seventeen-objective-LINCS-v1'
            or model.configuration.get('reconstruction_ownership') !=
                'parent-decoder-profile;branch-projection-only-v1'
            or type(model.lincs) is not SharedLincsModel
            or model.lincs.shared is not model.annotation
            or model.configuration.get('lincs_pairing') !=
                'independent-source-observations-no-image-join'):
        raise ValueError('exact native LINCS cooperative root and shared inherited owners required')
    # Validate native structure/axes before comparing its source configuration.
    native = lincs_parameter_ledger(model.lincs, identifier_axis=identifier_axis)
    if (model.configuration.get('lincs') != model.lincs.get_extra_state()
            or model.configuration.get('reconstruction') != configuration):
        raise ValueError('native LINCS/reconstruction source configuration drift')
    local = parameter_ledger(model.reconstruction,
        latent_reconstruction_local_parameter_roles(model.reconstruction,
            latent_axis=latent_axis,decoder=model.decoder,configuration=configuration))
    if (geometry_identifier_axis is None or image_identifier_axis is None
            or model.geometry_fusion.image is not model.native.image
            or model.configuration.get('geometry_schedule') != 'interface-only-no-adoption'
            or model.configuration.get('geometry_pairing') != 'none-inferred'
            or model.configuration.get('geometry') != model.geometry_fusion.get_extra_state()):
        raise ValueError('explicit shared geometry owner and coordinates required')
    geometry = geometry_fusion_parameter_ledger(model.geometry_fusion,
        identifier_axis=geometry_identifier_axis,image_identifier_axis=image_identifier_axis)
    return compose_owner_parameter_ledgers(model,dict(owners,reconstruction=local,
        geometry_fusion=geometry,lincs=native))


def color_cooperative_parameter_ledger(model, *, owners: Mapping[str, ParameterLedger],
        identifier_axis: tuple[str, ...], latent_axis: tuple[str, ...], configuration: dict,
        geometry_identifier_axis: tuple[str, ...], image_identifier_axis: tuple[str, ...],
        channel_axis: tuple[str, ...], color_identifier_axis: tuple[str, ...]) -> ParameterLedger:
    """Read-only complete native semantic/LINCS cooperative owner composition.

    Parent source/axis declarations are revalidated, not inferred from the saved
    wrapper configuration. Shared transforms require exact object identity; image
    and context incidence coordinates remain independent. This inventory changes
    no computation, source admission, scaling or historical checkpoint policy.
    """
    from .color_cooperative import ColorCooperativeModel
    from .lincs_cooperative import LincsCooperativeModel
    if (type(model) is not ColorCooperativeModel
            or set(model._modules) != {'parent','color'} or model._parameters
            or any(m._buffers for m in model.modules())
            or type(model.parent) is not LincsCooperativeModel
            or type(model.configuration) is not dict
            or type(channel_axis) is not tuple or channel_axis != ('R','G','B')):
        raise ValueError('exact cooperative semantic root and RGB coordinates required')
    parent = lincs_cooperative_parameter_ledger(model.parent,owners=owners,
        identifier_axis=identifier_axis,latent_axis=latent_axis,configuration=configuration,
        geometry_identifier_axis=geometry_identifier_axis,
        image_identifier_axis=image_identifier_axis)
    color = conditioned_semantic_parameter_ledger(model.color,channel_axis=channel_axis,
        identifier_axis=color_identifier_axis)
    native = model.color.color
    expected = dict(version=1,architecture='native-cooperative-color-semantic-attachment-v1',
        parent=model.parent.configuration,color=model.color.get_extra_state(),
        pairing='independent-source-pools-no-image-response-join',training_admission=False)
    if native.input_policy != 'native-max512-v1':
        expected.update(version=2,native_color=native.get_extra_state())
    if model.configuration != expected:
        raise ValueError('cooperative semantic source/configuration drift')
    backbone = model.parent.annotation.backbone
    for name in ('feature','role','graph_token','layers','final_norm'):
        if (getattr(native.backbone,name) is not getattr(backbone,name)
                or getattr(model.color.context.graph,name) is not getattr(backbone,name)):
            raise ValueError('cooperative semantic exact shared transforms required')
    return compose_owner_parameter_ledgers(model,{'parent':parent,'color':color})


def instance_cooperative_parameter_ledger(model, *, owners: Mapping[str, ParameterLedger],
        identifier_axis: tuple[str, ...], latent_axis: tuple[str, ...], configuration: dict,
        geometry_identifier_axis: tuple[str, ...], image_identifier_axis: tuple[str, ...],
        channel_axis: tuple[str, ...], color_identifier_axis: tuple[str, ...]) -> ParameterLedger:
    """Read-only native overlapping-instance/semantic/LINCS owner composition.

    Revalidates complete parent and slot declarations, exact shared modules and
    the non-registering context references. No forward, scaling, source access,
    optimizer, admission or checkpoint conversion is performed.
    """
    from .instance_cooperative import InstanceCooperativeModel
    if (type(model) is not InstanceCooperativeModel
            or set(model._modules) != {'parent','instance'} or model._parameters
            or any(m._buffers for m in model.modules())
            or type(model.configuration) is not dict):
        raise ValueError('exact cooperative instance root required')
    parent = color_cooperative_parameter_ledger(model.parent,owners=owners,
        identifier_axis=identifier_axis,latent_axis=latent_axis,configuration=configuration,
        geometry_identifier_axis=geometry_identifier_axis,
        image_identifier_axis=image_identifier_axis,channel_axis=channel_axis,
        color_identifier_axis=color_identifier_axis)
    instance = conditioned_instance_parameter_ledger(model.instance,
        channel_axis=channel_axis,identifier_axis=color_identifier_axis)
    expected = dict(version=1,architecture='native-cooperative-overlapping-nuclear-instance-v1',
        parent=model.parent.get_extra_state(),instance=model.instance.get_extra_state(),
        instance_color=model.instance.color.get_extra_state(),
        instance_head=model.instance.color.head.get_extra_state(),
        pairing='independent-source-pools-no-image-response-join',training_admission=False)
    if model.configuration != expected:
        raise ValueError('cooperative instance source/configuration drift')
    native = model.parent.color.color
    references = model._context_owners
    if (type(references) is not tuple or len(references) != 2
            or model.instance.context.graph.incidence is not references[0]
            or model.instance.context.payload is not references[1]
            or model.instance.color.pixels is not native.pixels
            or model.instance.color.backbone is not native.backbone
            or model.instance.context is not model.parent.color.context):
        raise ValueError('cooperative instance exact shared owners required')
    return compose_owner_parameter_ledgers(model,{'parent':parent,'instance':instance})
