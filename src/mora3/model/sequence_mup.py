"""Explicit fresh-only native recurrent head width adaptation.

Source-reviewed exact owner registries qualify only explicitly selected policies.
The optimizer's global shared-storage refusal remains unchanged. New recurrent
modules avoid persistent cuDNN parameter coalescing; native math/state names and
legacy modules keep their existing paths. Backend temporary packing is separate
and must be included in the full training resource measurement.
"""
import copy
import math

from torch import nn

from .mixed_payload_projection import exact_wire


class IndependentParameterGRU(nn.GRU):
    """Native GRU math with identity-preserving, independent parameter storages.

    RNNBase calls this hook at construction, device conversion and pointer
    refresh. Returning leaves the registered parameter allocations separate.
    Native GRU.forward still selects its ordinary device/backend implementation.
    """
    def flatten_parameters(self):
        return


def recurrent_parameterization(shared,policy):
    if policy not in ('legacy','mup-sequence-v1'):
        raise ValueError('known explicit native recurrent parameterization required')
    cfg=shared.get_extra_state()
    if policy=='legacy':
        if cfg.get('version')!=1:
            raise ValueError('new shared owner requires explicit recurrent parameterization')
        return None
    shared.validate_parameterization_configuration()
    return dict(schema='native-recurrent-parameterization-v1',policy=policy,
        width=cfg['width'],base_width=cfg['mup']['base_width'],output_mult=cfg['mup']['output_mult'],
        gates='native-reset-update-new-three-equal-slices',
        initialization='native-GRU-uniform-weights-fresh-bias-times-sqrt-width-over-base',
        recurrent_adam_lr='base-over-width',
        embedding='native-normal-with-exact-PAD-row',
        readout='native-linear-fresh-times-sqrt-width-over-base-input-divided-by-width-over-base',
        parameter_storage='independent-registered-parameters-no-persistent-cudnn-flatten')


def initialize_recurrent_head(owner,spec):
    """Apply only after a new head's normal constructors, never at state load."""
    if spec is None:return
    if type(owner.rnn) is not IndependentParameterGRU:
        raise ValueError('explicit independent-parameter native GRU required')
    import torch
    ratio=math.sqrt(spec['width']/spec['base_width'])
    with torch.no_grad():
        owner.rnn.bias_ih_l0.mul_(ratio);owner.rnn.bias_hh_l0.mul_(ratio)
        owner.output.weight.mul_(ratio);owner.output.bias.mul_(ratio)


def recurrent_readout(owner,hidden):
    spec=owner._recurrent_parameterization
    if spec is not None:
        if not exact_wire(spec,owner._recurrent_owned_spec):
            raise ValueError('native recurrent readout parameterization drift')
        hidden=hidden*(spec['output_mult']/(spec['width']/spec['base_width']))
    return owner.output(hidden)


def validate_recurrent_head(owner,*,codec_size,padding_idx):
    spec=owner._recurrent_parameterization
    if spec is None:raise ValueError('explicit applied native recurrent policy required')
    w=spec['width'];e,r,o=owner.embedding,owner.rnn,owner.output
    if (type(e) is not nn.Embedding or type(r) is not IndependentParameterGRU or type(o) is not nn.Linear
            or (e.num_embeddings,e.embedding_dim,e.padding_idx,e.max_norm,e.norm_type,e.scale_grad_by_freq,e.sparse)
                !=(codec_size,w,padding_idx,None,2.0,False,False)
            or (r.input_size,r.hidden_size,r.num_layers,r.bias,r.batch_first,r.dropout,r.bidirectional,r.proj_size,r.mode)
                !=(w,w,1,True,True,0,False,0,'GRU')
            or (o.in_features,o.out_features)!=(w,codec_size)
            or any(m._buffers or m._modules for m in (e,r,o))):
        raise ValueError('native recurrent embedding/gate/readout operator drift')
    expected={'embedding.weight':(codec_size,w),'rnn.weight_ih_l0':(3*w,w),
        'rnn.weight_hh_l0':(3*w,w),'rnn.bias_ih_l0':(3*w,),'rnn.bias_hh_l0':(3*w,),
        'output.weight':(codec_size,w),'output.bias':(codec_size,)}
    live={prefix+'.'+name:p for prefix,module in (('embedding',e),('rnn',r),('output',o))
          for name,p in module.named_parameters(remove_duplicate=False)}
    if set(live)!=set(expected) or any(tuple(live[n].shape)!=shape for n,shape in expected.items()):
        raise ValueError('native recurrent packed parameter shape/ownership drift')


def recurrent_checkpoint_preflight(owner,state_dict,prefix):
    """Versioned configurations before any descendant tensor copy.

    Tensor/state transactional validation belongs to the existing trainer. This
    preflight authenticates all non-tensor child policy declarations only.
    """
    import torch
    for key,value in owner.state_dict().items():
        if not isinstance(value,torch.Tensor) and not exact_wire(state_dict.get(prefix+key),value):
            raise ValueError('native recurrent checkpoint child configuration/policy mismatch')



def profile_mup_parameter_roles(model,*,identifier_axis):
    from .molecule_profile import SharedMoleculeProfileModel
    from .annotation_mup import annotation_mup_parameter_roles
    from .parameterization import molecule_profile_parameter_ledger,ParameterRole,parameter_ledger
    from .mu_optimizer import _storage_signature
    if type(model) is not SharedMoleculeProfileModel or model.parameterization_policy!='mup-molecule-profile-v1':
        raise ValueError('exact applied native molecule/profile owner required')
    model.validate_parameterization_configuration()
    native=annotation_mup_parameter_roles(model.shared,identifier_axis=identifier_axis)
    ledger=molecule_profile_parameter_ledger(model,identifier_axis=identifier_axis)
    roles={}
    for entry in ledger.parameters:
        for name in entry.aliases:
            if name.startswith('shared.'):role=native[name[len('shared.'):]]
            else:role=ParameterRole(entry.role,entry.axes)
            roles[name]=role
    _storage_signature(model);parameter_ledger(model,roles)
    return roles


def profile_mup_width_family(base,delta,target,*,identifier_axis):
    from .parameterization import parameter_ledger,width_family
    from .annotation_mup import annotation_mup_width_family
    from .morphology_mup import _without_member_dimensions
    models=(base,delta,target)
    ledgers=[parameter_ledger(m,profile_mup_parameter_roles(m,identifier_axis=identifier_axis)) for m in models]
    annotation_mup_width_family(*(m.shared for m in models),identifier_axis=identifier_axis)
    expected=None
    for m in models:
        cfg=_without_member_dimensions(m.get_extra_state())
        if expected is None:expected=cfg
        if not exact_wire(cfg,expected):raise ValueError('profile source finite axis/base/readout policy drift')
    return width_family(*ledgers)


def sequence_mup_parameter_roles(model,*,identifier_axis):
    from .nucleotide_decoder import NucleotideDecoder,NucleotideCodec
    from .profile_decoder import ProfileSelfiesDecoder,SelfiesCodec
    from .rfam_family import RfamFamilyModel
    from .pubchem_decoder import PubChemDecoder,PubChemByteCodec
    from .annotation_mup import annotation_mup_parameter_roles
    from .parameterization import Axis,ParameterRole,parameter_ledger
    from .mu_optimizer import _storage_signature
    import hashlib,json
    rfam=type(model) is RfamFamilyModel
    pubchem=type(model) is PubChemDecoder
    if type(model) in (NucleotideDecoder,RfamFamilyModel):
        if type(model.codec) is not NucleotideCodec:raise ValueError('exact native nucleotide codec required')
        from mora3.data.nucleotide_prior import ALPHABET
        if model.codec.symbols!=('<PAD>','<BOS>','<EOS>')+tuple(ALPHABET):
            raise ValueError('exact complete ordered native nucleotide symbols required')
        prefix='shared';namespace='nucleotide.decoder-vocabulary.v1';gate_name='nucleotide'
        owner=annotation_mup_parameter_roles(model.shared,identifier_axis=identifier_axis)
    elif type(model) is ProfileSelfiesDecoder:
        if type(model.codec) is not SelfiesCodec:raise ValueError('exact native SELFIES codec required')
        prefix='profiles';namespace='selfies.decoder-vocabulary.v1';gate_name='selfies'
        owner=profile_mup_parameter_roles(model.profiles,identifier_axis=identifier_axis)
    elif pubchem:
        if type(model.codec) is not PubChemByteCodec:raise ValueError('exact native PubChem byte codec required')
        from mora3.tokenization.shortest_path import ByteTokenizer
        c=model.codec;t=c.tokenizer
        if (type(t) is not ByteTokenizer or t.vocabulary[:256]!=tuple(bytes([i]) for i in range(256))
                or any(type(i) is not int for i in (c.pad,c.bos,c.eos,c.size))
                or (t.control_id('PAD'),t.control_id('BOS'),t.control_id('EOS'))!=(c.pad,c.bos,c.eos)
                or c.size!=len(t.vocabulary)+3):
            raise ValueError('complete native byte vocabulary and exact control IDs required')
        prefix='shared';namespace='pubchem.native-byte-decoder-vocabulary.v1';gate_name='pubchem-byte'
        owner=annotation_mup_parameter_roles(model.shared,identifier_axis=identifier_axis)
    else:raise ValueError('exact source-reviewed native recurrent owner required')
    model.get_extra_state()
    validate_recurrent_head(model,codec_size=model.codec.size,padding_idx=model.codec.pad)
    expected_modules={prefix,'embedding','rnn','output'}|({'family_output'} if rfam else set())
    if set(model._modules)!=expected_modules:raise ValueError('native recurrent owner module drift')
    if pubchem:
        descriptor=model.codec.configuration()
    else:
        if (type(model.codec.symbols) is not tuple or len(set(model.codec.symbols))!=len(model.codec.symbols)
                or model.codec.symbols[:3]!=('<PAD>','<BOS>','<EOS>')
                or type(model.codec.index) is not dict or model.codec.index!={x:i for i,x in enumerate(model.codec.symbols)}
                or any(type(i) is not int for i in model.codec.index.values())
                or any(type(i) is not int for i in (model.codec.pad,model.codec.bos,model.codec.eos,model.codec.size))
                or (model.codec.pad,model.codec.bos,model.codec.eos)!=(0,1,2)
                or model.codec.size!=len(model.codec.symbols)):
            raise ValueError('native recurrent ordered vocabulary/control drift')
        descriptor=model.codec.symbols
    roles={prefix+'.'+n:r for n,r in owner.items()}
    h=Axis('width','hidden');gate=Axis('width',gate_name+'-gru-reset-update-new-packed')
    v=Axis('finite',hashlib.sha256(json.dumps([namespace,descriptor],ensure_ascii=True,separators=(',',':')).encode()).hexdigest())
    roles.update({'embedding.weight':ParameterRole('embedding',(v,h)),
                  'output.weight':ParameterRole('output',(v,h)),'output.bias':ParameterRole('vector',(v,))})
    for name in ('weight_ih_l0','weight_hh_l0','bias_ih_l0','bias_hh_l0'):
        matrix=name.startswith('weight')
        roles['rnn.'+name]=ParameterRole('hidden' if matrix else 'vector',(gate,h) if matrix else (gate,))
    if rfam:
        if model.parameterization_policy!='mup-rfam-v1':raise ValueError('explicit applied Rfam source family readout required')
        f=Axis('finite',hashlib.sha256(json.dumps(['rfam.source-family-axis.v1',model.family_axis,model.manifest_sha256],
            ensure_ascii=True,separators=(',',':')).encode()).hexdigest())
        roles.update({'family_output.weight':ParameterRole('output',(f,h)),
                      'family_output.bias':ParameterRole('vector',(f,))})
    _storage_signature(model);parameter_ledger(model,roles)
    return roles


def sequence_mup_width_family(base,delta,target,*,identifier_axis):
    from .parameterization import parameter_ledger,width_family
    from .annotation_mup import annotation_mup_width_family
    from .profile_decoder import ProfileSelfiesDecoder
    from .morphology_mup import _without_member_dimensions
    models=(base,delta,target)
    if any(type(m) is not type(base) for m in models):raise ValueError('one exact recurrent owner class required across widths')
    ledgers=[parameter_ledger(m,sequence_mup_parameter_roles(m,identifier_axis=identifier_axis)) for m in models]
    if type(base) is ProfileSelfiesDecoder:
        profile_mup_width_family(*(m.profiles for m in models),identifier_axis=identifier_axis)
    else:annotation_mup_width_family(*(m.shared for m in models),identifier_axis=identifier_axis)
    expected=None
    for m in models:
        cfg=_without_member_dimensions(m.get_extra_state())
        if expected is None:expected=cfg
        if not exact_wire(cfg,expected):raise ValueError('native recurrent codec/base/gate/source policy drift')
    return width_family(*ledgers)
