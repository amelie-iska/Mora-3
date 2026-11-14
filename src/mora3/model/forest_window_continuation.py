"""Causal within-node forest-token continuation with bounded recurrent windows.

This is an additive, UNADMITTED neural operator. It never uses suffix labels to
score an earlier token. Its native forest context is the existing declared
skeleton/encoder; cross-node future-topology isolation, complete native muP
parameter roles, full-owner release and corpus supervision are NOT established.
"""
from dataclasses import dataclass,asdict
from hashlib import sha256
import copy,math
import torch
from torch import nn
from torch.nn import functional as F
from .forest_token_decoder import NativeForestTokenDecoder,_tensor_sha
from .forest_flow import _wire
from .mixed_payload_projection import exact_wire


@dataclass(frozen=True)
class WindowContinuationSpec:
    max_slots: int
    max_windows: int
    max_total_tokens: int
    max_score_elements: int

    def __post_init__(self):
        for key,bound in (('max_slots',4096),('max_windows',4096),
                          ('max_total_tokens',262144),('max_score_elements',2**31)):
            value=getattr(self,key)
            if type(value) is not int or not 1<=value<=bound:
                raise ValueError('Explicit finite causal window capacity required')
        if self.max_total_tokens>self.max_windows*self.max_slots:
            raise ValueError('Total token capacity exceeds declared window count')


class NativeForestWindowContinuation(nn.Module):
    """An isolated, exact-axis, trainable causal window decoder prototype.

    The native decoder and field are shared once (not duplicated). RNN hidden
    state is carried between <=4096-token windows; the RNN input at position t
    is ONLY token t-1, with a fixed start vector for t=0. No target suffix or
    STOP is fed into the next-token prediction at its own position.
    """
    def __init__(self,decoder,spec):
        super().__init__()
        if type(decoder) is not NativeForestTokenDecoder or type(spec) is not WindowContinuationSpec:
            raise ValueError('Exact native source-bound decoder and window specification required')
        decoder.get_extra_state();spec.__post_init__()
        if spec.max_slots>decoder.spec.max_slots or spec.max_score_elements<len(decoder.codebook):
            raise ValueError('Window or vocabulary budget incompatible with the native decoder')
        self.decoder=decoder
        self.spec=spec
        self.prefix_input=nn.Linear(decoder.codebook.shape[1],decoder.field.width)
        self.causal_gru=nn.GRU(decoder.field.width,decoder.field.width,batch_first=True)
        self.register_buffer('start',torch.zeros(decoder.field.width),persistent=True)
        ref=decoder.readout.weight
        self.prefix_input.to(device=ref.device,dtype=ref.dtype)
        self.causal_gru.to(device=ref.device,dtype=ref.dtype)
        self.start=self.start.to(device=ref.device,dtype=ref.dtype)
        self._frozen=self._current_configuration()
        self._topology=tuple((name,type(mod))for name,mod in self.named_modules(remove_duplicate=False))

    def _current_configuration(self):
        return dict(schema='Mora3.ForestWindowContinuation/v1',native_decoder=self.decoder.get_extra_state(),
                    spec=asdict(self.spec),source_codebook_sha256=_tensor_sha(self.decoder.codebook),
                    semantics='strict-causal-shifted-token-RNN-hidden-carried-between-windows',
                    attention='intra-node-only;cross-node-future-skeleton-not-qualified',
                    mu_parameter_roles_qualified=False,source_admission=False,training_admission=False)

    def get_extra_state(self):
        self.spec.__post_init__()
        if (not exact_wire(self._frozen,self._current_configuration())
                or tuple((n,type(m))for n,m in self.named_modules(remove_duplicate=False))!=self._topology
                or self.prefix_input.in_features!=self.decoder.codebook.shape[1]
                or self.prefix_input.out_features!=self.decoder.field.width
                or self.causal_gru.input_size!=self.decoder.field.width
                or self.causal_gru.hidden_size!=self.decoder.field.width
                or self.causal_gru.num_layers!=1 or self.causal_gru.bidirectional
                or tuple(self.start.shape)!=(self.decoder.field.width,)
                or not torch.equal(self.start,torch.zeros_like(self.start))):
            raise ValueError('Native causal decoder source/width/operator drift')
        return copy.deepcopy(self._frozen)

    def set_extra_state(self,value):
        if not exact_wire(value,self.get_extra_state()):
            raise ValueError('Causal decoder checkpoint source/operator mismatch')

    def load_state_dict(self,state_dict,strict=True,assign=False):
        if strict is not True or assign is not False:
            raise ValueError('Strict non-assignment exact checkpoint recovery required')
        expected=self.state_dict()
        if set(state_dict)!=set(expected):raise ValueError('Incomplete causal continuation checkpoint')
        for name,value in expected.items():
            incoming=state_dict[name]
            if isinstance(value,torch.Tensor):
                if (not isinstance(incoming,torch.Tensor)or incoming.shape!=value.shape
                        or incoming.dtype!=value.dtype or incoming.device!=value.device):
                    raise ValueError('Causal continuation checkpoint tensor mismatch')
                if name=='decoder.codebook' and not torch.equal(incoming,value):
                    raise ValueError('Frozen source codebook cannot change')
                if name=='start' and not torch.equal(incoming,value):
                    raise ValueError('Fixed causal start state cannot change')
            elif not exact_wire(incoming,value):
                raise ValueError('Causal continuation checkpoint source mismatch')
        return super().load_state_dict(state_dict,strict=True,assign=False)

    def last_window_lengths(self,target):
        self._validate_target(target)
        length=len(target);slot=self.spec.max_slots
        return tuple(min(slot,length-i)for i in range(0,length,slot))

    def _validate_target(self,target):
        self.get_extra_state()
        if (type(target)is not torch.Tensor or target.ndim!=1 or target.dtype!=torch.long
                or target.device!=self.decoder.codebook.device or target.is_meta
                or not 1<=len(target)<=self.spec.max_total_tokens
                or math.ceil(len(target)/self.spec.max_slots)>self.spec.max_windows):
            raise ValueError('Complete token target exceeds exact causal window capacity')
        ids=target.tolist()
        allowed=set(self.decoder.allowed_token_ids)
        if (ids[-1]!=self.decoder.stop_token or self.decoder.stop_token in ids[:-1]
                or any(i not in allowed for i in ids)):
            raise ValueError('Exactly one final stop and valid native payload tokens required')

    def _window_scores(self,record,task,state,template,node_index,target):
        self._validate_target(target)
        if (type(state)is not torch.Tensor or state.ndim!=3 or len(state)!=1
                or state.device!=self.decoder.codebook.device or state.dtype!=self.decoder.codebook.dtype
                or type(node_index)is not int or not 0<=node_index<state.shape[1]
                or node_index in template.clamped):
            raise ValueError('One exact native source state and unclamped node required')
        # The existing field validates the native source task and declared
        # skeleton. This interface does not establish global prefix isolation.
        initial=self.decoder.field.encode_forest(record,task,state,state.new_ones(1),template)
        hidden=initial[0,node_index].reshape(1,1,-1)
        W=self.spec.max_slots
        for start in range(0,len(target),W):
            end=min(start+W,len(target))
            prior=target[max(0,start-1):end-1]
            previous=self.prefix_input(self.decoder.codebook.index_select(0,prior))
            if start==0:
                previous=torch.cat((self.start[None],previous),dim=0)
            if len(previous)!=end-start:
                raise ValueError('Causal next-token shift mismatch')
            out,hidden=self.causal_gru(previous[None],hidden)
            slots=self.decoder.slot_embedding.weight[:end-start]
            query=self.decoder.readout(F.silu(out[0]+slots)*(
                self.decoder.field.readout_mult/(self.decoder.field.width/self.decoder.field.base_width)))
            chunk=self.spec.max_score_elements//len(self.decoder.codebook)
            for pos in range(0,len(query),chunk):
                yield self.decoder._scores(query[pos:pos+chunk]),target[start+pos:start+pos+len(query[pos:pos+chunk])]

    @torch.no_grad()
    def generate_token_ids(self,record,task,state,template,node_index,*,max_steps):
        """Greedy, token-causal continuation; never claim a partial result complete.

        No labeled target bytes are provided to this inference path. The original
        source field still requires separate future-node context qualification.
        """
        self.get_extra_state()
        if(type(max_steps)is not int or not 1<=max_steps<=self.spec.max_total_tokens
                or math.ceil(max_steps/self.spec.max_slots)>self.spec.max_windows):
            raise ValueError('Exact inference window and total token capacity required')
        if (type(state)is not torch.Tensor or state.ndim!=3 or len(state)!=1
                or state.device!=self.decoder.codebook.device or state.dtype!=self.decoder.codebook.dtype
                or type(node_index)is not int or not 0<=node_index<state.shape[1]
                or node_index in template.clamped):
            raise ValueError('One exact native source state and unclamped node required')
        initial=self.decoder.field.encode_forest(record,task,state,state.new_ones(1),template)
        hidden=initial[0,node_index].reshape(1,1,-1)
        previous=None
        result=[]
        for step in range(max_steps):
            inp=(self.start if previous is None else
                 self.prefix_input(self.decoder.codebook[previous].unsqueeze(0))[0])
            out,hidden=self.causal_gru(inp.reshape(1,1,-1),hidden)
            slot=self.decoder.slot_embedding.weight[step%self.spec.max_slots]
            query=self.decoder.readout(F.silu(out[0,0]+slot)*(
                self.decoder.field.readout_mult/(self.decoder.field.width/self.decoder.field.base_width)))
            next_id=int(self.decoder._scores(query[None])[0].argmax())
            result.append(next_id)
            if next_id==self.decoder.stop_token:
                return dict(schema='Mora3.CausalWindowDecode/v1',tokens=tuple(result),
                            complete=True,reason='final-stop',training_admission=False)
            previous=next_id
        return dict(schema='Mora3.CausalWindowDecode/v1',tokens=tuple(result),
                    complete=False,reason='token-budget-exhausted',training_admission=False)

    def teacher_logits(self,record,task,state,template,node_index,target):
        self._validate_target(target)
        if len(target)*len(self.decoder.codebook)>self.spec.max_score_elements:
            raise ValueError('Dense causal output exceeds exact score capacity; use streaming loss')
        return torch.cat([scores for scores,_ in self._window_scores(
            record,task,state,template,node_index,target)],dim=0)

    def teacher_loss(self,record,task,state,template,node_index,target):
        total=None
        for score,label in self._window_scores(record,task,state,template,node_index,target):
            part=F.cross_entropy(score,label,reduction='sum')
            total=part if total is None else total+part
        return total/len(target)
