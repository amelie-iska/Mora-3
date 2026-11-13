"""Source-frozen token embeddings, learned node/slot projection and typed decoding.

Nearest-codebook refinement is a local Mora-3 adaptation, not an assertion that
pooled reasoning embeddings invert to tokens. Only supported, versioned codebook
exports and aligned token labels can qualify production use. Source validity,
semantic correctness and corpus admission are not supplied by token syntax.
"""
from dataclasses import dataclass,asdict
from hashlib import sha256
import copy,math
import torch
from torch import nn
from torch.nn import functional as F
from mora3.tokenization.forest_tokens import ForestTokenAxis,serialize_token_forest,parse_token_forest
from mora3.reasoning.flow_forest_refinement import ForestState,DecodedForest,DecodeRefusal,forest_fingerprint
from .forest_flow import NativeForestFlowHead,_sha,_wire,_linear,_parameter_versions,forest_context_sha256,forest_flow_parameter_roles
from .mixed_payload_projection import exact_wire


def _tensor_sha(x):
    return sha256(_wire([list(x.shape),str(x.dtype)])+x.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class ForestTokenCodebook:
    axis: ForestTokenAxis
    values: torch.Tensor
    source_snapshot_sha256: str
    coordinate_sha256: str
    encoder_sha256: str
    export_sha256: str
    evidence_sha256: str

    def __post_init__(self):
        if type(self.axis)is not ForestTokenAxis:raise ValueError('Exact versioned forest token axis required')
        self.axis.__post_init__()
        for pin in(self.source_snapshot_sha256,self.coordinate_sha256,self.encoder_sha256,self.export_sha256,self.evidence_sha256):_sha(pin)
        x=self.values
        if(not isinstance(x,torch.Tensor)or x.ndim!=2 or x.shape[0]!=self.axis.tokenizer().size
            or not 1<=x.shape[1]<=1024 or x.numel()>16_777_216 or x.layout!=torch.strided
            or x.is_meta or x.dtype not in(torch.float32,torch.float64)or x.requires_grad
            or not torch.isfinite(x).all()):
            raise ValueError('Complete finite detached FP32/FP64 vocabulary codebook required')

    def binding(self):
        self.__post_init__()
        return dict(schema='Mora3.ForestTokenCodebook/v1',axis_sha256=self.axis.sha256,
            source_snapshot_sha256=self.source_snapshot_sha256,coordinate_sha256=self.coordinate_sha256,
            encoder_sha256=self.encoder_sha256,export_sha256=self.export_sha256,evidence_sha256=self.evidence_sha256,
            values_sha256=_tensor_sha(self.values),shape=list(self.values.shape),dtype=str(self.values.dtype),
            source_admission=False)


@dataclass(frozen=True)
class ForestDecoderSpec:
    max_slots: int
    temperature: float=1.
    max_score_elements: int=8_000_000
    max_embedding_elements: int=8_000_000
    max_output_tokens: int=262144
    max_output_bytes: int=4*1024**2

    def __post_init__(self):
        for value,maximum in((self.max_slots,4096),(self.max_score_elements,2**31),
            (self.max_output_tokens,262144),(self.max_output_bytes,64*1024**2),(self.max_embedding_elements,2**31)):
            if type(value)is not int or not 1<=value<=maximum:raise ValueError('Explicit positive bounded decoder budgets required')
        if type(self.temperature)not in(int,float)or not math.isfinite(self.temperature)or self.temperature<=0:
            raise ValueError('Positive finite decoder temperature required')


@dataclass(frozen=True)
class KnownForestPayloads:
    context_sha256: str
    coordinate_sha256: str
    payloads: tuple[tuple[int,bytes],...]
    evidence_sha256: str

    def __post_init__(self):
        for pin in(self.context_sha256,self.coordinate_sha256,self.evidence_sha256):_sha(pin)
        if(type(self.payloads)is not tuple or len(self.payloads)>512
            or any(type(row)is not tuple or len(row)!=2 or type(row[0])is not int or row[0]<0 or type(row[1])is not bytes for row in self.payloads)
            or len({i for i,_ in self.payloads})!=len(self.payloads)
            or sum(len(raw)for _,raw in self.payloads)>4*1024**2):
            raise ValueError('Explicit unique bounded source condition payloads required')


class NativeForestTokenDecoder(nn.Module):
    def __init__(self,field,codebook,spec):
        super().__init__()
        if(type(field)is not NativeForestFlowHead or type(codebook)is not ForestTokenCodebook or type(spec)is not ForestDecoderSpec):
            raise ValueError('Exact native field, codebook and decoder specification required')
        field.get_extra_state();codebook.__post_init__();spec.__post_init__()
        if(codebook.source_snapshot_sha256!=field.spec.source_snapshot_sha256
            or codebook.coordinate_sha256!=field.spec.coordinate_sha256
            or codebook.values.dtype!=field.readout.weight.dtype):
            raise ValueError('Decoder source/coordinate/dtype must match the native field')
        if (spec.max_score_elements<codebook.values.shape[0]
                or spec.max_slots*max(field.width,codebook.values.shape[1])>spec.max_embedding_elements):
            raise ValueError('Decoder score budget must fit at least one complete vocabulary row')
        self.field=field;self._field_reference=(field,);self.axis=codebook.axis;self.spec=spec
        self.source_codebook=codebook;self._codebook_binding=codebook.binding()
        self.register_buffer('codebook',codebook.values.detach().clone().to(field.readout.weight.device),persistent=True)
        self.slot_embedding=nn.Embedding(spec.max_slots,field.width)
        self.readout=nn.Linear(field.width,codebook.values.shape[1])
        with torch.no_grad():
            scale=math.sqrt(field.width/field.base_width)
            self.readout.weight.mul_(scale);self.readout.bias.mul_(scale)
        for layer in(self.slot_embedding,self.readout):
            layer.to(dtype=field.readout.weight.dtype,device=field.readout.weight.device)
        self.stop_token=self.axis.tokenizer().control_id('<MORA:NODE_END>')
        self.allowed_token_ids=tuple(range(len(self.axis.tokenizer().vocabulary)))+(self.stop_token,)
        self._configuration=self._current_configuration()
        self._topology=tuple((name,type(module))for name,module in self.named_modules(remove_duplicate=False))

    def _current_configuration(self):
        return dict(schema='Mora3.NativeForestTokenDecoder/v1',field=self.field.get_extra_state(),
            spec=asdict(self.spec),codebook=self._codebook_binding,
            projection='silu(node_hidden+finite_slot_embedding);MuReadout-to-frozen-codebook',
            scoring='2qE-E2 over temperature; query norm cancels in softmax',
            stop_token=self.stop_token,allowed_token_ids=self.allowed_token_ids,
            missing_stop='explicit-incomplete-refusal',serialization='known-skeleton-typed-grammar-v1',
            probabilities='conditional-token-slots-not-flow-density',scoring_memory='bounded-row-chunks-with-nonreentrant-CE-recomputation',training_admission=False)

    def get_extra_state(self):
        self.spec.__post_init__()
        if(_tensor_sha(self.codebook)!=self._codebook_binding['values_sha256']
            or not exact_wire(self.source_codebook.binding(),self._codebook_binding)):
            raise ValueError('Frozen decoder codebook changed')
        if(self.field is not self._field_reference[0]
            or type(self.slot_embedding)is not nn.Embedding
            or tuple(self.slot_embedding.weight.shape)!=(self.spec.max_slots,self.field.width)
            or not _linear(self.readout,self.field.width,self.codebook.shape[1])
            or self._topology!=tuple((n,type(m))for n,m in self.named_modules(remove_duplicate=False))
            or not exact_wire(self._configuration,self._current_configuration())):
            raise ValueError('Native decoder source/axis/operator configuration drift')
        return copy.deepcopy(self._configuration)

    def set_extra_state(self,state):
        if not exact_wire(state,self.get_extra_state()):raise ValueError('Decoder checkpoint configuration mismatch')

    def load_state_dict(self,state_dict,strict=True,assign=False):
        if strict is not True or assign is not False:raise ValueError('Strict non-assignment decoder recovery required')
        return super().load_state_dict(state_dict,strict=True,assign=False)

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        self.get_extra_state();expected=self.state_dict()
        if(local_metadata.get('assign_to_params_buffers',False)is not False
            or {k for k in state_dict if k.startswith(prefix)}!={prefix+k for k in expected}):
            raise ValueError('Exact complete decoder checkpoint keys required')
        for name,live in expected.items():
            incoming=state_dict[prefix+name]
            if isinstance(live,torch.Tensor):
                if(not isinstance(incoming,torch.Tensor)or incoming.is_meta or incoming.shape!=live.shape
                    or incoming.dtype!=live.dtype or incoming.layout!=live.layout):
                    raise ValueError('Decoder checkpoint tensor specification mismatch')
            elif not exact_wire(live,incoming):raise ValueError('Decoder checkpoint source/operator sidecar mismatch')
        if not torch.equal(state_dict[prefix+'codebook'],self.codebook):raise ValueError('Frozen codebook cannot change on load')
        aliases={}
        for name,p in self.named_parameters(remove_duplicate=False):
            value=state_dict[prefix+name]
            if id(p)in aliases and not torch.equal(value,aliases[id(p)]):raise ValueError('Decoder tied aliases differ')
            aliases[id(p)]=value
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def token_embeddings(self,record,task,state,template):
        self.get_extra_state()
        if(not isinstance(state,torch.Tensor)or state.ndim!=3
            or state.shape[0]*state.shape[1]*self.spec.max_slots*max(self.field.width,self.codebook.shape[1])>self.spec.max_embedding_elements):
            raise ValueError('Decoder exceeds declared embedding capacity; no truncation')
        hidden=self.field.encode_forest(record,task,state,state.new_ones(len(state)),template)
        slots=F.silu(hidden[:,:,None,:]+self.slot_embedding.weight[None,None,:,:])
        projected=self.readout(slots*(self.field.readout_mult/(self.field.width/self.field.base_width)))
        if not torch.isfinite(projected).all():raise ValueError('Nonfinite projected token embedding')
        return projected

    def _scores(self,projected,*,codebook=None):
        book=self.codebook if codebook is None else codebook
        if projected.numel()//projected.shape[-1]*len(book)>self.spec.max_score_elements:
            raise ValueError('Dense decoder score tensor exceeds the declared bound')
        score=(2*torch.matmul(projected,book.T)-book.square().sum(1))/self.spec.temperature
        allowed=torch.zeros(len(book),device=score.device,dtype=torch.bool)
        allowed[list(self.allowed_token_ids)]=True
        if not torch.isfinite(score).all():raise ValueError('Nonfinite nearest-codebook score')
        return score.masked_fill(~allowed,float('-inf'))

    def forward(self,record,task,state,template):
        if not isinstance(state,torch.Tensor)or state.ndim!=3:raise ValueError('Complete forest state tensor required')
        if state.shape[0]*state.shape[1]*self.spec.max_slots*len(self.codebook)>self.spec.max_score_elements:
            raise ValueError('Dense decoder score tensor exceeds the declared bound; use bounded decoding')
        return self._scores(self.token_embeddings(record,task,state,template))

    @torch.no_grad()
    def nearest_token_ids(self,record,task,state,template):
        values=self.token_embeddings(record,task,state,template)
        flat=values.reshape(-1,values.shape[-1]);chunk=self.spec.max_score_elements//len(self.codebook)
        ids=[self._scores(flat[i:i+chunk]).argmax(-1)for i in range(0,len(flat),chunk)]
        return torch.cat(ids).reshape(values.shape[:-1])

    def bind_context(self,record,task,known):
        return BoundForestDecoder(self,record,task,known)


class BoundForestDecoder:
    def __init__(self,decoder,record,task,known):
        if type(decoder)is not NativeForestTokenDecoder or type(known)is not KnownForestPayloads:
            raise ValueError('Exact learned decoder and known condition payloads required')
        known.__post_init__();decoder.get_extra_state()
        self.decoder=decoder;self.record=record;self.task=task;self.known=known
        self.context_sha256=forest_context_sha256(record,task,source_snapshot_sha256=decoder.field.spec.source_snapshot_sha256)
        if known.context_sha256!=self.context_sha256 or known.coordinate_sha256!=decoder.field.spec.coordinate_sha256:
            raise ValueError('Known payloads must bind exact source context and coordinates')
        self._versions=_parameter_versions(decoder);self._configuration=_wire(decoder.get_extra_state())
        h=sha256(self._configuration)
        for name,p in decoder.named_parameters():h.update(_wire([name,list(p.shape),str(p.dtype),_tensor_sha(p)]))
        self.sha256=sha256(_wire(dict(schema='Mora3.BoundForestDecoder/v1',weights=h.hexdigest(),
            context=self.context_sha256,known=[(i,x.hex())for i,x in known.payloads],evidence=known.evidence_sha256))).hexdigest()

    @torch.no_grad()
    def __call__(self,forest):
        d=self.decoder;d.get_extra_state();forest_fingerprint(forest)
        if(self._versions!=_parameter_versions(d)or self._configuration!=_wire(d.get_extra_state())
            or self.context_sha256!=forest_context_sha256(self.record,self.task,source_snapshot_sha256=d.field.spec.source_snapshot_sha256)):
            raise ValueError('Decoder model/context drift; bind a fresh snapshot')
        if forest.context_sha256!=self.context_sha256 or tuple(sorted(i for i,_ in self.known.payloads))!=forest.clamped:
            raise ValueError('Known payload indices must be exactly the clamped conditions')
        tokens=d.nearest_token_ids(self.record,self.task,forest.states[None],forest)[0]
        payloads=[];known=dict(self.known.payloads);tokenizer=d.axis.tokenizer()
        for i,row in enumerate(tokens):
            if i in known:payloads.append(known[i]);continue
            ends=(row==d.stop_token).nonzero().flatten()
            if not len(ends):
                return DecodeRefusal(forest_fingerprint(forest),self.context_sha256,self.sha256,
                    'token-slot-budget-without-stop',self.known.evidence_sha256,True)
            payloads.append(tokenizer.decode(row[:int(ends[0])].tolist()))
        encoded=serialize_token_forest(d.axis,forest,tuple(payloads),max_tokens=d.spec.max_output_tokens,max_bytes=d.spec.max_output_bytes)
        rows=parse_token_forest(d.axis,encoded,max_nodes=d.field.spec.max_nodes,max_tokens=d.spec.max_output_tokens,max_bytes=d.spec.max_output_bytes)
        identity=sha256(_wire([dict(row,payload=row['payload'].hex())for row in rows])).hexdigest()
        return DecodedForest(encoded,identity,forest_fingerprint(forest),self.context_sha256,self.sha256)


def prepare_forest_token_targets(axis,forests,payloads,*,max_slots):
    if type(axis)is not ForestTokenAxis or type(forests)is not tuple or not forests or type(payloads)is not tuple or len(payloads)!=len(forests):
        raise ValueError('Exact axis, complete forests and source-aligned payload tuples required')
    if type(max_slots)is not int or not 1<=max_slots<=4096:raise ValueError('Explicit bounded token slots required')
    n=len(forests[0].parents);device=forests[0].states.device
    if len(forests)*n*max_slots>16_777_216:raise ValueError('Token target allocation exceeds budget')
    tok=axis.tokenizer();stop=tok.control_id('<MORA:NODE_END>')
    target=torch.full((len(forests),n,max_slots),-100,dtype=torch.long,device=device);mask=torch.zeros_like(target,dtype=torch.bool)
    for b,(forest,row)in enumerate(zip(forests,payloads)):
        if type(forest)is not ForestState or not forest.is_complete or len(forest.parents)!=n or forest.states.device!=device:
            raise ValueError('Complete same-axis forest batch required')
        forest.__post_init__()
        if type(row)is not tuple or len(row)!=n:raise ValueError('One explicit optional payload per node required')
        for i,payload in enumerate(row):
            if payload is None:continue
            if i in forest.clamped or type(payload)is not bytes:raise ValueError('Only known free-node targets are supervised')
            if len(payload)>4*1024**2:raise ValueError('Source payload exceeds byte budget')
            encoded=tok.encode(payload)+(stop,)
            if len(encoded)>max_slots:raise ValueError('Complete target exceeds slot budget; no truncation')
            target[b,i,:len(encoded)]=torch.tensor(encoded,device=device);mask[b,i,:len(encoded)]=True
    return target,mask


def forest_token_loss(decoder,record,task,state,template,target,available):
    if type(decoder)is not NativeForestTokenDecoder:raise ValueError('Exact native token decoder required')
    decoder.get_extra_state()
    if not isinstance(state,torch.Tensor)or state.ndim!=3:raise ValueError('Complete forest state tensor required')
    decoder.field._validate(record,task,state,state.new_ones(len(state)),template)
    shape=(len(state),state.shape[1],decoder.spec.max_slots)
    if(not isinstance(target,torch.Tensor)or target.shape!=shape or target.dtype!=torch.long or target.device!=state.device
        or not isinstance(available,torch.Tensor)or available.shape!=shape or available.dtype!=torch.bool or available.device!=state.device):
        raise ValueError('Exact token-label and availability tensors required')
    if available[:,list(template.clamped)].any():raise ValueError('Known clamped payloads cannot be target labels')
    if not available.any():return dict(loss=None,supervised_tokens=0,training_admission=False)
    allowed=set(decoder.allowed_token_ids)
    for b in range(len(state)):
        for i in range(state.shape[1]):
            count=int(available[b,i].sum())
            if not count:continue
            if not available[b,i,:count].all()or available[b,i,count:].any():raise ValueError('Token target must be a complete contiguous prefix')
            ids=target[b,i,:count].tolist()
            if any(k not in allowed for k in ids)or ids[-1]!=decoder.stop_token or decoder.stop_token in ids[:-1]:
                raise ValueError('Exactly one final stop with permitted payload tokens required')
    if len(state)*state.shape[1]*decoder.spec.max_slots*len(decoder.codebook)<=decoder.spec.max_score_elements:
        loss=F.cross_entropy(decoder(record,task,state,template)[available],target[available])
    else:
        values=decoder.token_embeddings(record,task,state,template)[available];labels=target[available]
        chunk=decoder.spec.max_score_elements//len(decoder.codebook)
        from torch.utils.checkpoint import checkpoint
        frozen_configuration=_wire(decoder.get_extra_state())
        def block_loss(q,book,label):
            if _wire(decoder.get_extra_state())!=frozen_configuration:
                raise ValueError('Decoder configuration changed during backward recomputation')
            return F.cross_entropy(decoder._scores(q,codebook=book),label,reduction='sum')
        # Passing the frozen book explicitly retains PyTorch version checking.
        # The checkpointed function has no RNG or model parameter mutation.
        # Without recomputation, autograd would retain ALL per-chunk logits.
        losses=[]
        for i in range(0,len(values),chunk):
            args=(values[i:i+chunk],decoder.codebook,labels[i:i+chunk])
            losses.append(checkpoint(block_loss,*args,use_reentrant=False,preserve_rng_state=False)
                if torch.is_grad_enabled()and values.requires_grad else block_loss(*args))
        loss=sum(losses)/len(values)
    return dict(loss=loss,supervised_tokens=int(available.sum()),training_admission=False)


def forest_decoder_parameter_roles(model,*,identifier_axis):
    from .parameterization import Axis,ParameterRole,parameter_ledger
    from .mu_optimizer import _storage_signature
    if type(model)is not NativeForestTokenDecoder:raise ValueError('Exact native token decoder required')
    model.get_extra_state()
    roles={'field.'+n:r for n,r in forest_flow_parameter_roles(model.field,identifier_axis=identifier_axis).items()}
    hidden=Axis('width','hidden')
    slots=Axis('finite',sha256(_wire(['forest-token-slots',model.spec.max_slots])).hexdigest())
    output=Axis('finite',sha256(_wire(model._codebook_binding)).hexdigest())
    roles['slot_embedding.weight']=ParameterRole('embedding',(slots,hidden))
    roles['readout.weight']=ParameterRole('output',(output,hidden));roles['readout.bias']=ParameterRole('vector',(output,))
    _storage_signature(model);parameter_ledger(model,roles)
    return roles


def native_forest_token_objective(owner,*args,**kwargs):
    from .full_owner import SourceBoundOwner
    if type(owner)is not SourceBoundOwner:raise ValueError('Exact source-bound owner required')
    owner.require_training_admission()
    decoder=getattr(owner,'forest_decoder',None)
    if type(decoder)is not NativeForestTokenDecoder:raise ValueError('Factory-registered decoder required')
    return forest_token_loss(decoder,*args,**kwargs)
