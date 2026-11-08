"""Native-symbol autoregressive prior on shared ConvexTok/TokenGT context.

Train-only unpaired reference priors; no translation, paired response inference,
long-context feasibility or biological generation-quality claim. Target sequence
is decoded in native order, separately from the known single condition node.
"""
import copy
import json
import torch
from torch import nn
from torch.nn import functional as F
from mora3.data.nucleotide_prior import ALPHABET, SOURCES, NucleotidePriorView
from mora3.data.rnacentral_prior import DOMAIN as RNACENTRAL_DOMAIN
from mora3.data.schema import GraphRecord, Node, Measurement
from mora3.data.provenance import Provenance
from mora3.data.tasks import ConditionalTask
from .annotation_multihead import SharedAnnotationModel
from .mixed_payload_projection import exact_wire
from .sequence_mup import (IndependentParameterGRU,recurrent_parameterization,initialize_recurrent_head,
    recurrent_readout,validate_recurrent_head,recurrent_checkpoint_preflight)


def _is_rnacentral_view(view):
    p = view.provenance()
    return (view.inputs().get('prior_domain') == RNACENTRAL_DOMAIN
            or p.get('schema') == 'mora3.RNAcentralPriorAdmission/v1'
            or p.get('source_family') == 'RNAcentral-reference-priors')


def _validate_rnacentral_view(view):
    """Replay the bounded admitted reader before source-observed model entry.

    Provenance flags alone are not admission. Exact returned native/context/
    receipt/group bytes must match a fresh pinned file-reader result. No raw
    FASTA or full-corpus scan occurs; this correctness reference does not cache.
    """
    from pathlib import Path
    import hashlib
    from mora3.data.rnacentral_prior import MASTER_SHA256, SOURCE, read_rnacentral_priors
    try:
        p = view.provenance()
        if (p.get('schema') != 'mora3.RNAcentralPriorAdmission/v1'
                or p.get('source_family') != SOURCE
                or view.inputs() != dict(prior_domain=RNACENTRAL_DOMAIN, source_organism=None)):
            raise ValueError('source-bound RNAcentral prior required')
        original = p['source_ref']['original_FASTA']
        if type(original) is not str or not Path(original).is_absolute() or '/outputs/' not in original:
            raise ValueError('retained native source root required')
        root = Path(original.split('/outputs/', 1)[0]).resolve()
        manifest = Path('outputs/data-intake/20261101-PubChem-corpus-pilot-v1/'
                        'independent-master-union-v20/master-corpus-manifest-v20-PubChem-native-codec-pilot-v1.json')
        raw = (root / manifest).read_bytes()
        if hashlib.sha256(raw).hexdigest() != MASTER_SHA256:
            raise ValueError('authoritative RNAcentral pilot manifest mismatch')
        master = json.loads(raw)
        shards = [s['path'] for s in master['shards']
                  if s.get('source_family') == SOURCE and s['sha256'] == p['corpus_binding']['shard_sha256']]
        if len(shards) != 1:
            raise ValueError('one admitted pilot shard required')
        checked, = read_rnacentral_priors(manifest, root=root, expected_manifest_sha256=MASTER_SHA256,
            shard_paths=shards, task_ids=[p['task_id']], source_status=p['source_ref']['source_status'], limit=1)
        if checked.canonical_bytes() != view.canonical_bytes():
            raise ValueError('exact RNAcentral source/task/receipt/type/group binding mismatch')
    except (KeyError, TypeError, OSError) as exc:
        raise ValueError('complete pinned RNAcentral admission required') from exc


class NucleotideCodec:
    pad, bos, eos = 0, 1, 2
    symbols = ('<PAD>', '<BOS>', '<EOS>') + tuple(ALPHABET)
    size = len(symbols)
    index = {s:i for i,s in enumerate(symbols)}

    def encode(self, sequence):
        if not isinstance(sequence,str) or not sequence or set(sequence)-set(ALPHABET):
            raise ValueError('nonempty native nucleotide symbols required')
        return [self.bos]+[self.index[c] for c in sequence]+[self.eos]

    def decode(self, ids):
        ids=list(ids)
        if (not ids or ids[0]!=self.bos or self.eos not in ids
                or any(type(i)!=int for i in ids)):
            raise ValueError('native BOS/EOS sequence required')
        end=ids.index(self.eos)
        if any(i<3 or i>=self.size for i in ids[1:end]) or any(i!=self.pad for i in ids[end+1:]):
            raise ValueError('invalid native symbol or padding')
        return ''.join(self.symbols[i] for i in ids[1:end])


class NucleotideDecoder(nn.Module):
    def __init__(self, shared, *, parameterization_policy="legacy"):
        super().__init__()
        if not isinstance(shared,SharedAnnotationModel):raise ValueError('native shared model required')
        spec=recurrent_parameterization(shared,parameterization_policy)
        self.shared=shared;self.codec=NucleotideCodec()
        self._recurrent_parameterization=spec
        self._recurrent_owned_spec=copy.deepcopy(spec)
        self._recurrent_shared_reference=(shared,)
        self._recurrent_codec=(self.codec.symbols,self.codec.size,self.codec.pad,self.codec.bos,self.codec.eos,
                               tuple(self.codec.index.items()))
        w=shared.payload.embedding.embedding_dim
        self.embedding=nn.Embedding(self.codec.size,w,padding_idx=self.codec.pad)
        self.rnn=(nn.GRU if spec is None else IndependentParameterGRU)(w,w,batch_first=True)
        self.output=nn.Linear(w,self.codec.size)
        initialize_recurrent_head(self,spec)
        p=next(shared.parameters());self.to(device=p.device,dtype=p.dtype)

    def _validate_recurrent_parameterization(self):
        spec=self._recurrent_parameterization
        if (not exact_wire(spec,self._recurrent_owned_spec)
                or not exact_wire(spec,recurrent_parameterization(self.shared,'mup-sequence-v1'))
                or self.shared is not self._recurrent_shared_reference[0]
                or self._recurrent_codec!=(self.codec.symbols,self.codec.size,self.codec.pad,self.codec.bos,self.codec.eos,
                                            tuple(self.codec.index.items()))):
            raise ValueError('native recurrent source/codec/parameterization drift')
        validate_recurrent_head(self,codec_size=self.codec.size,padding_idx=self.codec.pad)

    def get_extra_state(self):
        spec=getattr(self,'_recurrent_parameterization',None)
        if spec is not None:self._validate_recurrent_parameterization()
        result=dict(version=1,symbols=self.codec.symbols,shared=self.shared.get_extra_state(),
            payload=self.shared.payload.get_extra_state(),architecture='native-symbol-conditional-initial-state-GRU',
            objective='mean-observed-lexical-next-symbol-and-EOS-CE-excluding-padding',
            condition_keys=('prior_domain','source_organism'),target_truncation=False)
        if spec is not None:result.update(version=2,mup=copy.deepcopy(spec))
        return copy.deepcopy(result)

    def set_extra_state(self,state):
        if state!=self.get_extra_state():raise ValueError('nucleotide configuration mismatch')

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        if getattr(self,'_recurrent_parameterization',None) is not None:
            recurrent_checkpoint_preflight(self,state_dict,prefix)
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def _readout(self,hidden):
        if getattr(self,'_recurrent_parameterization',None) is None:return self.output(hidden)
        return recurrent_readout(self,hidden)

    def condition(self,conditions):
        """Target-free public interface. No target length, topology or audit metadata."""
        conditions=tuple(conditions)
        if not conditions:raise ValueError('nonempty native conditions required')
        graphs=[]
        for c in conditions:
            if (not isinstance(c,dict) or set(c)!={'prior_domain','source_organism'}
                    or c['prior_domain'] not in ({v[0] for v in SOURCES.values()} | {RNACENTRAL_DOMAIN})
                    or (c['prior_domain'] == RNACENTRAL_DOMAIN and c['source_organism'] is not None)
                    or (c['source_organism'] is not None and
                        (not isinstance(c['source_organism'],str) or not c['source_organism']))):
                raise ValueError('native condition allowlist/missingness mismatch')
            record=GraphRecord('input',(Node('nucleotide-prior',(
                Measurement('condition',json.dumps(c,sort_keys=True),None,'observed',True),
                Measurement('requested_output',None,None,'missing',False))),),(),
                (Provenance('native-input','v1','input','0'*64,'source-admission-external'),))
            task=ConditionalTask(((('node',0,'condition'),'condition'),
                                  (('node',0,'requested_output'),'target')),topology_known=True)
            graphs.append(self.shared.payload.forward_task(record,task=task,
                identifiers=self.shared.sequence_identifiers()))
        return self.shared.backbone(graphs)[0][:,0]

    def teacher_batch(self,views):
        views=tuple(views)
        if not views or any(not isinstance(v,NucleotidePriorView) for v in views):
            raise ValueError('nonempty source-bound nucleotide views required')
        for v in views:
            if _is_rnacentral_view(v):
                _validate_rnacentral_view(v)
        ids=[self.codec.encode(v.sequence) for v in views]
        p=next(self.parameters());n=max(len(x)-1 for x in ids)
        prefix=torch.full((len(ids),n),self.codec.pad,device=p.device,dtype=torch.long)
        target=prefix.clone()
        for i,x in enumerate(ids):
            prefix[i,:len(x)-1]=prefix.new_tensor(x[:-1]);target[i,:len(x)-1]=target.new_tensor(x[1:])
        return prefix,target

    def forward(self,conditions,prefix):
        if getattr(self,'_recurrent_parameterization',None) is not None:self._validate_recurrent_parameterization()
        conditions=tuple(conditions)
        if (not isinstance(prefix,torch.Tensor) or prefix.dtype!=torch.long or prefix.ndim!=2
                or prefix.shape[0]!=len(conditions) or prefix.shape[1]<1
                or prefix.device!=self.embedding.weight.device
                or bool(((prefix<0)|(prefix>=self.codec.size)).any())):
            raise ValueError('valid ordered native prefixes required')
        h=self.condition(conditions).unsqueeze(0).contiguous()
        hidden,_=self.rnn(self.embedding(prefix),h)
        return self._readout(hidden)

    def objective(self,views):
        views=tuple(views);prefix,target=self.teacher_batch(views)
        return F.cross_entropy(self(tuple(v.inputs() for v in views),prefix).reshape(-1,self.codec.size),
            target.reshape(-1),ignore_index=self.codec.pad)

    @torch.no_grad()
    def generate(self,conditions,*,max_new_symbols):
        """Explicit caller budget, never target-derived. Nontermination is retained."""
        if type(max_new_symbols)!=int or max_new_symbols<1:raise ValueError('positive generation budget required')
        conditions=tuple(conditions);h=self.condition(conditions).unsqueeze(0).contiguous()
        token=torch.full((len(conditions),1),self.codec.bos,device=h.device,dtype=torch.long)
        sequences=[[self.codec.bos] for _ in conditions];done=[False]*len(conditions)
        for _ in range(max_new_symbols):
            z,h=self.rnn(self.embedding(token),h);logits=self._readout(z[:,0])
            logits[:,:2]=-torch.inf
            token=logits.argmax(-1)[:,None]
            for i,t in enumerate(token[:,0].tolist()):
                if not done[i]:sequences[i].append(t);done[i]=t==self.codec.eos
            if all(done):break
        return dict(evidence_class='model_produced',token_ids=tuple(tuple(x) for x in sequences),
            terminated=tuple(done),sequences=tuple(self.codec.decode(x) if d else None for x,d in zip(sequences,done)),
            partial_sequences=tuple(''.join(self.codec.symbols[t] for t in x[1:] if t>=3) for x in sequences))
