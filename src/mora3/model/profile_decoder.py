"""Versioned profile-conditioned SELFIES autoregressive reference component.

Source SMILES are converted by the installed SELFIES codec; no claims about
chemical-group generalization, causal response, or biomedical validity follow.
This decoder does not replace flow matching or other modality generators.
"""
from .configuration_snapshot import configuration_snapshot
from .mixed_payload_projection import exact_wire
from .sequence_mup import (IndependentParameterGRU,recurrent_parameterization,initialize_recurrent_head,
    recurrent_readout,validate_recurrent_head,recurrent_checkpoint_preflight)
import copy
import selfies
import torch
from torch import nn
from torch.nn import functional as F


class SelfiesCodec:
    pad, bos, eos = 0, 1, 2

    def __init__(self, smiles):
        symbols=set()
        for text in smiles:
            encoded=selfies.encoder(text,strict=True)
            symbols.update(selfies.split_selfies(encoded))
        if not symbols: raise ValueError('nonempty molecular vocabulary required')
        self.symbols=('<PAD>','<BOS>','<EOS>')+tuple(sorted(symbols))
        self.index={s:i for i,s in enumerate(self.symbols)}
        self.size=len(self.symbols)

    @classmethod
    def from_ordered_source_symbols(cls,symbols):
        """Restore an exact immutable source-built finite axis without refitting."""
        symbols=tuple(symbols)
        if (len(symbols)<4 or symbols[:3]!=('<PAD>','<BOS>','<EOS>')
                or any(type(s)is not str for s in symbols) or len(set(symbols))!=len(symbols)
                or symbols[3:]!=tuple(sorted(symbols[3:]))):
            raise ValueError('exact ordered source SELFIES controls and unique lexical axis required')
        try:
            valid=all(tuple(selfies.split_selfies(s))==(s,)for s in symbols[3:])
        except(Exception,):valid=False
        if not valid:raise ValueError('exact single source SELFIES symbols required')
        codec=cls.__new__(cls);codec.symbols=symbols
        codec.index={s:i for i,s in enumerate(symbols)};codec.size=len(symbols)
        return codec

    def encode(self, smiles, *, max_tokens):
        if not isinstance(smiles,str) or not smiles.strip(): raise ValueError('nonempty molecule required')
        symbols=tuple(selfies.split_selfies(selfies.encoder(smiles,strict=True)))
        if len(symbols)+2>max_tokens: raise ValueError('sequence length exceeds bound; no truncation')
        if any(s not in self.index for s in symbols): raise ValueError('unknown SELFIES vocabulary symbol')
        return [self.bos]+[self.index[s] for s in symbols]+[self.eos]

    def decode(self, ids):
        ids=list(ids)
        if not ids or ids[0]!=self.bos or self.eos not in ids: raise ValueError('BOS/EOS required')
        end=ids.index(self.eos)
        if any(type(i)!=int or i<3 or i>=self.size for i in ids[1:end]):
            raise ValueError('invalid molecular token')
        if any(i!=self.pad for i in ids[end+1:]): raise ValueError('nonpadding after EOS')
        return selfies.decoder(''.join(self.symbols[i] for i in ids[1:end]))


class ProfileSelfiesDecoder(nn.Module):
    def __init__(self, profiles, codec, *, max_tokens,parameterization_policy="legacy"):
        super().__init__()
        if type(max_tokens)!=int or max_tokens<3: raise ValueError('length bound must allow BOS/symbol/EOS')
        spec=recurrent_parameterization(profiles.shared,parameterization_policy)
        if spec is not None:profiles.validate_parameterization_configuration()
        self.profiles=profiles;self.codec=codec;self.max_tokens=max_tokens
        self._recurrent_parameterization=spec;self._recurrent_owned_spec=copy.deepcopy(spec)
        self._recurrent_profiles_reference=(profiles,)
        self._recurrent_codec=(codec.symbols,codec.size,codec.pad,codec.bos,codec.eos,tuple(codec.index.items()),max_tokens)
        w=profiles.shared.payload.embedding.embedding_dim
        self.embedding=nn.Embedding(codec.size,w,padding_idx=codec.pad)
        self.rnn=(nn.GRU if spec is None else IndependentParameterGRU)(w,w,batch_first=True)
        self.output=nn.Linear(w,codec.size)
        initialize_recurrent_head(self,spec)
        p=next(profiles.parameters());self.to(device=p.device,dtype=p.dtype)

    def _validate_recurrent_parameterization(self):
        if (not exact_wire(self._recurrent_parameterization,self._recurrent_owned_spec)
                or not exact_wire(self._recurrent_parameterization,recurrent_parameterization(self.profiles.shared,'mup-sequence-v1'))
                or self.profiles is not self._recurrent_profiles_reference[0]
                or self._recurrent_codec!=(self.codec.symbols,self.codec.size,self.codec.pad,self.codec.bos,self.codec.eos,
                                            tuple(self.codec.index.items()),self.max_tokens)):
            raise ValueError('SELFIES recurrent source/codec/parameterization drift')
        self.profiles.validate_parameterization_configuration()
        validate_recurrent_head(self,codec_size=self.codec.size,padding_idx=self.codec.pad)

    def get_extra_state(self):
        if self._recurrent_parameterization is not None:self._validate_recurrent_parameterization()
        result=dict(version=1,representation='SELFIES',selfies_version=selfies.__version__,
            symbols=self.codec.symbols,max_tokens=self.max_tokens,profiles=self.profiles.get_extra_state(),
            objective='mean-nonpadding-next-token-cross-entropy',architecture='conditional-initial-state-GRU')
        if self._recurrent_parameterization is not None:result.update(version=2,mup=copy.deepcopy(self._recurrent_parameterization))
        return configuration_snapshot(result)

    def set_extra_state(self,state):
        if state!=self.get_extra_state():raise ValueError('decoder configuration mismatch')

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        if self._recurrent_parameterization is not None:recurrent_checkpoint_preflight(self,state_dict,prefix)
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def _readout(self,hidden):
        return recurrent_readout(self,hidden)

    def _views(self,views):
        views=self.profiles._validate(views)
        if any(v.direction!='profile_to_molecule' for v in views):raise ValueError('inverse direction required')
        return views

    def teacher_batch(self,views):
        views=self._views(views)
        ids=[self.codec.encode(v.targets()['canonical_isomeric_smiles'],max_tokens=self.max_tokens) for v in views]
        p=next(self.parameters());n=max(len(x)-1 for x in ids)
        prefix=torch.full((len(ids),n),self.codec.pad,device=p.device,dtype=torch.long)
        target=prefix.clone()
        for i,x in enumerate(ids):
            prefix[i,:len(x)-1]=prefix.new_tensor(x[:-1]);target[i,:len(x)-1]=target.new_tensor(x[1:])
        return prefix,target

    def forward(self,views,prefix):
        if self._recurrent_parameterization is not None:self._validate_recurrent_parameterization()
        views=self._views(views)
        if (prefix.dtype!=torch.long or prefix.ndim!=2 or prefix.shape[0]!=len(views) or
            not 1<=prefix.shape[1]<self.max_tokens or (prefix<0).any() or (prefix>=self.codec.size).any()):
            raise ValueError('bounded valid prefix required')
        context=self.profiles.condition(views)
        hidden,_=self.rnn(self.embedding(prefix),context.unsqueeze(0).contiguous())
        return self._readout(hidden)

    def objective(self,views):
        views=self._views(views);prefix,target=self.teacher_batch(views)
        return F.cross_entropy(self(views,prefix).reshape(-1,self.codec.size),target.reshape(-1),ignore_index=self.codec.pad)

    @torch.no_grad()
    def generate(self,views):
        """Greedy target-free decoding; unresolved sequences are explicit, never cut silently."""
        views=self._views(views);context=self.profiles.condition(views)
        h=context.unsqueeze(0).contiguous();token=torch.full((len(views),1),self.codec.bos,device=context.device,dtype=torch.long)
        sequences=[[self.codec.bos] for _ in views];done=[False]*len(views)
        for _ in range(self.max_tokens-1):
            z,h=self.rnn(self.embedding(token),h);logits=self._readout(z[:,0])
            logits[:,:3]=-torch.inf # re-enable EOS; BOS and PAD cannot be generated
            logits[:,self.codec.eos]=self._readout(z[:,0])[:,self.codec.eos]
            token=logits.argmax(-1)[:,None]
            for i,t in enumerate(token[:,0].tolist()):
                if not done[i]:sequences[i].append(t);done[i]=t==self.codec.eos
            if all(done):break
        return dict(token_ids=tuple(tuple(x) for x in sequences),terminated=tuple(done),
            smiles=tuple(self.codec.decode(x) if d else None for x,d in zip(sequences,done)))
