"""Positive-observation residue incidence reference, not bond-pair prediction.

Interval annotations supervise covered positions; bonds supervise endpoints only.
Unknown positions/types have no loss. This cannot establish specificity or calibrated
probabilities. Scope/evidence remain in the source-bound batch, never encoder input.
"""
import math
import copy
from .mixed_payload_projection import exact_wire
import torch
from torch.nn import functional as F
from mora3.data.uniprot_residue import residue_batch


def align_token_residues(sequence, tokenizer, token_features):
    """Expand each complete payload token over its ASCII residue span, no pooling.

    Repeated features share gradients; within-token residue discrimination needs a
    positional/contextual encoder in the caller. Controls and targets never enter.
    """
    if not isinstance(sequence,str) or not sequence or any(not ('A'<=c<='Z') for c in sequence):
        raise ValueError('nonempty uppercase ASCII residue sequence required')
    ids=tokenizer.encode(sequence.encode('ascii'))
    if token_features.ndim!=2 or token_features.shape[0]!=len(ids):
        raise ValueError('one feature row per complete sequence token required')
    lengths=[len(tokenizer.vocabulary[i]) for i in ids]
    if tokenizer.decode(ids)!=sequence.encode('ascii'):
        raise ValueError('sequence token roundtrip failed')
    index=torch.repeat_interleave(torch.arange(len(ids),device=token_features.device),
                                 torch.tensor(lengths,device=token_features.device))
    return token_features[index]


def observed_residue_loss(logits, views, *, feature_axis, applicable):
    """Mean softplus(-z) over unique source-observed (record,residue,type) cells.

    Overlapping duplicate assertions do not add weight. Ineligible locations stay
    lossless in the data contract without coordinate loss. Caller applicability is
    explicit task metadata. No logits arithmetic on missing/nonapplicable cells.
    """
    views,logits=tuple(views),tuple(logits)
    batch=residue_batch(views,feature_axis=feature_axis)
    axis=batch.feature_axis
    if len(logits)!=len(views) or not logits:
        raise ValueError('one residue logit matrix per view required')
    device=logits[0].device
    if (not isinstance(applicable,torch.Tensor) or applicable.dtype!=torch.bool
            or applicable.shape!=(len(views),) or applicable.device!=device):
        raise ValueError('Boolean applicability on logits device required')
    selected=[];zeros=[]
    for i,(z,v,rows) in enumerate(zip(logits,views,batch.targets())):
        if (z.ndim!=2 or z.shape!=(len(v.sequence),len(axis)) or z.device!=device
                or not z.is_floating_point() or z.dtype!=logits[0].dtype):
            raise ValueError('residue/feature dimensions, dtype and device must agree')
        zeros.append(z.reshape(-1)[:0].sum())
        if not applicable[i].item():continue
        positions=set()
        for row in rows:
            if row['coordinate_status']!='exact_canonical':continue
            j=axis.index(row['feature_type'])
            if row['interval0'] is not None:
                a,b=row['interval0'];indices=range(a,b)
            elif row['endpoint_indices0'] is not None:
                indices=row['endpoint_indices0']
            else:continue
            positions.update((p,j) for p in indices)
        if positions:
            p,j=zip(*sorted(positions))
            selected.append(z[list(p),list(j)])
    zero=torch.stack(zeros).sum()
    if not selected:return zero
    return F.softplus(-torch.cat(selected)).mean()+zero


class ContextualResidueHead(torch.nn.Module):
    """Experimental local residue head over shared learned features/context.

    Absolute sinusoidal positions distinguish residues within a learned token.
    A width-three convolution supplies local context; the caller supplies the
    shared biological context separately. This adds no independent global
    backbone and does not replace TokenGT. No targets/provenance are inputs.
    """
    def __init__(self, width, feature_axis, *, parameterization_policy="legacy",
                 base_width=None, output_mult=1.0):
        super().__init__()
        axis = tuple(feature_axis)
        if (not isinstance(width, int) or width < 2 or width % 2
                or not axis or any(not isinstance(x, str) or not x for x in axis)
                or len(set(axis)) != len(axis)):
            raise ValueError('even positive width and unique nonempty feature axis required')
        if parameterization_policy not in ('legacy','mup-residue-v1'):
            raise ValueError('known residue parameterization policy required')
        if parameterization_policy=='legacy':
            if base_width is not None or output_mult!=1.0:
                raise ValueError('legacy residue head has no width declaration')
        elif (type(base_width) is not int or base_width<2 or base_width%2 or width<base_width
                or type(output_mult) not in (int,float) or not math.isfinite(output_mult) or output_mult<=0):
            raise ValueError('valid residue base width and finite positive output multiplier required')
        self.parameterization_policy,self.base_width,self.output_mult=parameterization_policy,base_width,float(output_mult)
        self.width, self.feature_axis = width, axis
        self._declaration=(width,axis,parameterization_policy,base_width,float(output_mult))
        self.context = torch.nn.Linear(width, width)
        self.local = torch.nn.Conv1d(width, width, 3, padding=1)
        self.output = torch.nn.Linear(width, len(axis))
        if parameterization_policy!='legacy':
            with torch.no_grad():
                ratio=math.sqrt(width/base_width)
                self.context.bias.mul_(ratio);self.local.bias.mul_(ratio)
                self.output.weight.mul_(ratio);self.output.bias.mul_(ratio)

    def get_extra_state(self):
        if self._declaration!=(self.width,self.feature_axis,self.parameterization_policy,self.base_width,self.output_mult):
            raise ValueError('residue parameterization declaration drift')
        result=dict(version=1, width=self.width, feature_axis=self.feature_axis,
                    position='sinusoidal-10000', context='shared-plus-local3')
        if self.parameterization_policy!='legacy':
            result.update(version=2,mup=dict(schema='residue-parameterization-v1',
                policy=self.parameterization_policy,base_width=self.base_width,output_mult=self.output_mult,
                convolution='dense-fixed-three-offsets-fanin-width-times-three',
                hidden_adam_lr='base-over-width',bias_initialization='native-times-sqrt-width-over-base',
                readout='native-times-sqrt-width-over-base-input-divided-by-width-over-base'))
        return result

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError('residue head configuration/ordered axis mismatch')

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,
                              missing_keys,unexpected_keys,error_msgs):
        if not exact_wire(state_dict.get(prefix+'_extra_state'),self.get_extra_state()):
            raise ValueError('residue head checkpoint configuration mismatch')
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,
                                     missing_keys,unexpected_keys,error_msgs)

    def forward(self, residue_features, shared_context):
        features=self.encode_features(residue_features,shared_context)
        if self.parameterization_policy!='legacy':
            features=features*(self.output_mult/(self.width/self.base_width))
        return self.output(features)

    def encode_features(self, residue_features, shared_context):
        self.get_extra_state()
        x, c = residue_features, shared_context
        if (x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != self.width
                or c.shape != (self.width,) or not x.is_floating_point()
                or x.dtype != c.dtype or x.device != c.device):
            raise ValueError('nonempty residue features and matching shared context required')
        p = torch.arange(len(x), device=x.device, dtype=x.dtype)[:, None]
        frequency = torch.exp(-torch.arange(0, self.width, 2, device=x.device,
                              dtype=x.dtype) * math.log(10000.) / self.width)
        position = torch.stack((torch.sin(p * frequency), torch.cos(p * frequency)), dim=-1).flatten(1)
        h = x + position + self.context(c)
        return h + F.gelu(self.local(h.T[None])[0].T)


from mora3.model.uniprot_go import annotation_graph_task


class SharedResidueModel(torch.nn.Module):
    """Residue incidence using the native shared sequence graph context.

    The single known protein TokenGT entity supplies global pooled context; exact
    token expansion supplies local features. This is not all-residue attention.
    Annotation heads remain owned once by the shared model and inactive here.
    """
    def __init__(self, shared, feature_axis, *, parameterization_policy="legacy"):
        super().__init__()
        from mora3.model.annotation_multihead import SharedAnnotationModel
        if not isinstance(shared, SharedAnnotationModel):
            raise ValueError('native shared annotation model required')
        if parameterization_policy not in ('legacy','mup-residue-v1'):
            raise ValueError('known shared residue parameterization policy required')
        cfg=shared.get_extra_state()
        if parameterization_policy=='legacy' and cfg['version']!=1:
            raise ValueError('new shared owner requires explicit residue parameterization')
        if parameterization_policy!='legacy':
            shared.validate_parameterization_configuration()
        self.parameterization_policy=parameterization_policy
        self.shared = shared
        self._shared_reference=(shared,)
        spec=cfg.get('mup',{})
        self.head = ContextualResidueHead(shared.payload.embedding.embedding_dim, feature_axis,
            parameterization_policy=parameterization_policy,base_width=spec.get('base_width'),
            output_mult=spec.get('output_mult',1.0))
        self.head.to(device=shared.payload.embedding.weight.device,
                     dtype=shared.payload.embedding.weight.dtype)

    def get_extra_state(self):
        if self.parameterization_policy!='legacy':
            if self.shared is not self._shared_reference[0] or self.head.parameterization_policy!=self.parameterization_policy:
                raise ValueError('residue owner policy/reference drift')
            self.shared.validate_parameterization_configuration()
        return dict(version=1 if self.parameterization_policy=='legacy' else 2,
                    objective='observed-residue-incidence-v1',
                    context='native-single-protein-TokenGT',
                    shared=self.shared.get_extra_state(),
                    payload=self.shared.payload.get_extra_state(),
                    head=self.head.get_extra_state())

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError('shared residue configuration mismatch')

    def _validate_parameterization_checkpoint(self,state_dict,prefix):
        if not exact_wire(state_dict.get(prefix+'_extra_state'),self.get_extra_state()):
            raise ValueError('shared residue checkpoint configuration mismatch')
        if not exact_wire(state_dict.get(prefix+'head._extra_state'),self.head.get_extra_state()):
            raise ValueError('residue head checkpoint configuration mismatch')
        owner=self.shared
        if not exact_wire(state_dict.get(prefix+'shared._extra_state'),owner.get_extra_state()):
            raise ValueError('residue shared owner checkpoint configuration mismatch')
        if self.parameterization_policy!='legacy':
            lead=prefix+'shared.'
            owner.payload._validate_parameterization_checkpoint(state_dict,lead+'payload.')
            owner.backbone._validate_initialization_checkpoint(state_dict,lead+'backbone.')
            for name in ('payload','go','ec')+(('physical',) if hasattr(owner,'physical') else ()):
                if not exact_wire(state_dict.get(lead+name+'._extra_state'),getattr(owner,name).get_extra_state()):
                    raise ValueError('residue shared child checkpoint configuration mismatch')

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,
                              missing_keys,unexpected_keys,error_msgs):
        self._validate_parameterization_checkpoint(state_dict,prefix)
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,
                                     missing_keys,unexpected_keys,error_msgs)

    def forward(self, views, *, prepared=None, applicable=None):
        views = tuple(views)
        if prepared is not None:
            self._validate_prepared(prepared, views, applicable)
        if not views:
            raise ValueError('nonempty prediction views required')
        graphs=[]
        for view in views:
            record,task=annotation_graph_task(view)
            graphs.append(self.shared.payload.forward_task(record,task=task,
                          identifiers=self.shared.sequence_identifiers()))
        contexts=self.shared.backbone(graphs)[0][:,0]
        tokenizer=self.shared.payload.codec.tokenizer
        result=[]
        for i,(v,c) in enumerate(zip(views,contexts)):
            token_ids = (tokenizer.encode(v.sequence.encode('ascii')) if prepared is None
                         else prepared.token_ids[i])
            ids=torch.tensor(token_ids,
                             device=c.device,dtype=torch.long)
            x=self.shared.payload.embedding(ids)
            aligned = (align_token_residues(v.sequence,tokenizer,x) if prepared is None
                       else prepared.align(i,x))
            result.append(self.head(aligned,c))
        return tuple(result)

    def encode_sequences(self,sequences):
        """Condition-request features only, with no source annotation adapter.

        Request content hashes identify supplied input bytes; they do not assert
        source provenance, rights, experimental truth, or corpus admission.
        Sparse known-site coordinates and decoder target history are absent.
        """
        import hashlib
        from mora3.data.schema import GraphRecord,Node,Measurement
        from mora3.data.provenance import Provenance
        sequences=tuple(sequences)
        if (not sequences or any(type(s) is not str or not s
                or any(c not in 'ACDEFGHIKLMNPQRSTVWYBXZJUO' for c in s) for s in sequences)):
            raise ValueError('complete canonical protein condition sequences required')
        self.get_extra_state()
        graphs=[]
        for sequence in sequences:
            sha=hashlib.sha256(sequence.encode('ascii')).hexdigest()
            record=GraphRecord('provided-sequence-condition:'+sha,
                (Node('protein',(Measurement('sequence',sequence,None,'observed',True),)),),(),
                (Provenance('provided-sequence-condition','request-content-sha256-v1',
                    sha,sha,'no-rights-inferred'),))
            graphs.append(self.shared.payload(record,input_fields={('node',0,'sequence')},
                identifiers=self.shared.sequence_identifiers()))
        contexts=self.shared.backbone(graphs)[0][:,0]
        tokenizer=self.shared.payload.codec.tokenizer
        result=[]
        for sequence,context in zip(sequences,contexts):
            ids=torch.tensor(tokenizer.encode(sequence.encode('ascii')),device=context.device,dtype=torch.long)
            aligned=align_token_residues(sequence,tokenizer,self.shared.payload.embedding(ids))
            result.append(self.head.encode_features(aligned,context))
        return tuple(result)

    def _validate_prepared(self, prepared, views, applicable):
        from .residue_preparation import PreparedResidues
        device=self.shared.payload.embedding.weight.device
        if (not isinstance(prepared, PreparedResidues)
                or not isinstance(applicable, torch.Tensor)
                or applicable.dtype != torch.bool or applicable.device != device
                or applicable.shape != (len(views),)):
            raise ValueError('prepared residues and Boolean applicability on model device required')
        prepared.validate(views, self.shared.payload.codec.tokenizer,
                          feature_axis=self.head.feature_axis,
                          applicable=tuple(applicable.tolist()))

    def objective(self, logits, views, *, applicable, prepared=None):
        views=tuple(views)
        if prepared is not None:
            self._validate_prepared(prepared, views, applicable)
            loss=prepared.loss(logits)
        else:
            loss=observed_residue_loss(logits,views,feature_axis=self.head.feature_axis,
                                       applicable=applicable)
        return dict(residue=loss,total=loss)
