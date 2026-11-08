"""Observed source annotation LIST/EOS primitive; never biological negatives.

A caller supplies already target-excluded sequence context and explicit causal
prefixes. Source-specific readers own membership, source scope, splits, and the
meaning/order of the observed list. This primitive grants no corpus admission.
Its vocabulary is a frozen exact source label axis, followed by EOS; BOS is input
only. Empty observed lists supervise EOS, unobserved lists receive no target.
"""
import copy
import hashlib
import json
import math

import torch
from torch import nn
from torch.nn import functional as F

from .mixed_payload_projection import exact_wire
from .parameterization import Axis,ParameterRole,parameter_ledger,width_family
from .mu_optimizer import _storage_signature


class SourceAnnotationListHead(nn.Module):
    """Named finite-width native GRUCell adaptation for ordered source lists."""
    def __init__(self,width,axis,*,source_namespace,base_width,output_mult=1.0,max_target_length):
        super().__init__()
        if (type(width) is not int or type(base_width) is not int or base_width<1 or width<base_width
                or type(axis) is not tuple or not axis or len(set(axis))!=len(axis)
                or any(type(x) is not str or not x for x in axis)
                or type(source_namespace) is not str or not source_namespace.strip()
                or type(output_mult) not in (int,float) or not math.isfinite(output_mult) or output_mult<=0
                or type(max_target_length) is not int or max_target_length<1):
            raise ValueError('exact source axis/namespace and valid list decoder width/base/budget required')
        self.width,self.base_width,self.output_mult=width,base_width,float(output_mult)
        self.axis,self.source_namespace,self.max_target_length=axis,source_namespace,max_target_length
        self.eos,self.bos=len(axis),len(axis)+1
        self.embedding=nn.Embedding(len(axis)+2,width)
        self.cell=nn.GRUCell(width,width)
        self.output=nn.Linear(width,len(axis)+1)
        with torch.no_grad():
            ratio=math.sqrt(width/base_width)
            self.cell.bias_ih.mul_(ratio);self.cell.bias_hh.mul_(ratio)
            self.output.weight.mul_(ratio);self.output.bias.mul_(ratio)
        self._configuration=self._current_configuration()

    @property
    def axis_sha256(self):
        return hashlib.sha256(json.dumps([self.source_namespace,self.axis,'observed-source-list/EOS/BOS-v1'],
            ensure_ascii=True,separators=(',',':')).encode()).hexdigest()

    def _current_configuration(self):
        return dict(schema='Mora3.SourceAnnotationListHead/v1',width=self.width,base_width=self.base_width,
            output_mult=self.output_mult,source_namespace=self.source_namespace,axis=self.axis,
            axis_sha256=self.axis_sha256,eos=self.eos,bos=self.bos,
            prediction_classes=len(self.axis)+1,input_classes=len(self.axis)+2,
            max_target_length=self.max_target_length,order='source-specific-explicit-list-order-and-repetitions',
            objective='mean-ordered-list-next-label-cross-entropy-including-observed-EOS',
            biological_negatives=False,missing_list='unobserved-no-loss-not-empty',
            source_admission='external-reader-owned-never-created-here',
            gate_initialization='native-uniform-fresh-bias-times-sqrt-width-over-base',gate_adam_lr='base-over-width',
            readout='native-fresh-times-sqrt-width-over-base-input-divided-by-width-over-base')

    def get_extra_state(self):
        if not exact_wire(self._current_configuration(),self._configuration):
            raise ValueError('source list declaration/axis configuration drift')
        modules=dict(self.named_modules(remove_duplicate=False))
        expected={'':SourceAnnotationListHead,'embedding':nn.Embedding,'cell':nn.GRUCell,'output':nn.Linear}
        if (set(modules)!=set(expected) or any(type(modules[n]) is not t for n,t in expected.items())
                or (self.embedding.num_embeddings,self.embedding.embedding_dim,self.embedding.padding_idx,
                    self.embedding.max_norm,self.embedding.norm_type,self.embedding.scale_grad_by_freq,self.embedding.sparse)
                    !=(len(self.axis)+2,self.width,None,None,2.0,False,False)
                or (self.cell.input_size,self.cell.hidden_size,self.cell.bias)!=(self.width,self.width,True)
                or (self.output.in_features,self.output.out_features)!=(self.width,len(self.axis)+1)
                or any(m._buffers for m in modules.values())):
            raise ValueError('source list native operator ownership/configuration drift')
        shapes={'embedding.weight':(len(self.axis)+2,self.width),'cell.weight_ih':(3*self.width,self.width),
            'cell.weight_hh':(3*self.width,self.width),'cell.bias_ih':(3*self.width,),
            'cell.bias_hh':(3*self.width,),'output.weight':(len(self.axis)+1,self.width),'output.bias':(len(self.axis)+1,)}
        params=dict(self.named_parameters(remove_duplicate=False))
        if set(params)!=set(shapes) or any(tuple(params[n].shape)!=shape for n,shape in shapes.items()):
            raise ValueError('source list packed parameter shape drift')
        return copy.deepcopy(self._configuration)

    def set_extra_state(self,state):
        if not exact_wire(state,self.get_extra_state()):raise ValueError('source list checkpoint configuration mismatch')

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        if not exact_wire(state_dict.get(prefix+'_extra_state'),self.get_extra_state()):
            raise ValueError('source list checkpoint source/axis/policy mismatch')
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def _observed_lists(self,observed_lists):
        if (type(observed_lists) is not tuple or not observed_lists
                or any(type(row) is not tuple or len(row)+1>self.max_target_length
                    or any(type(i) is not int or not 0<=i<len(self.axis) for i in row) for row in observed_lists)):
            raise ValueError('explicit complete observed source lists required; missing is not empty')
        return observed_lists

    def teacher_prefixes(self,observed_lists):
        return tuple((self.bos,*row) for row in self._observed_lists(observed_lists))

    def step(self,ids,hidden):
        self.get_extra_state()
        if (not isinstance(ids,torch.Tensor) or ids.dtype!=torch.long or ids.ndim!=1 or not len(ids)
                or not isinstance(hidden,torch.Tensor) or hidden.shape!=(len(ids),self.width)
                or ids.device!=hidden.device or hidden.device!=self.embedding.weight.device
                or hidden.dtype!=self.embedding.weight.dtype or not torch.isfinite(hidden).all()
                or (ids<0).any() or (ids>=len(self.axis)+2).any()):
            raise ValueError('aligned finite source-list context and exact integer prefix IDs required')
        return self._step(ids,hidden)

    def _step(self,ids,hidden):
        hidden=self.cell(self.embedding(ids),hidden)
        return self.output(hidden*(self.output_mult/(self.width/self.base_width))),hidden

    def forward(self,context,prefixes,*,prior_logits=None):
        self.get_extra_state()
        if (not isinstance(context,torch.Tensor) or context.ndim!=2 or context.shape[1]!=self.width
                or not len(context) or context.device!=self.embedding.weight.device
                or context.dtype!=self.embedding.weight.dtype or not torch.isfinite(context).all()
                or type(prefixes) is not tuple or len(prefixes)!=len(context)
                or any(type(row) is not tuple or not 1<=len(row)<=self.max_target_length
                    or type(row[0]) is not int or row[0]!=self.bos
                    or any(type(i) is not int or not 0<=i<len(self.axis) for i in row[1:]) for row in prefixes)):
            raise ValueError('complete context and bounded causal BOS/source-role prefixes required')
        if prior_logits is not None and (not isinstance(prior_logits,torch.Tensor)
                or prior_logits.shape!=(len(context),len(self.axis)+1) or prior_logits.dtype!=context.dtype
                or prior_logits.device!=context.device or not torch.isfinite(prior_logits).all()):
            raise ValueError('complete matching source-next-symbol prior logits required')
        outputs=[]
        for i,(c,row) in enumerate(zip(context,prefixes)):
            hidden=c[None];logits=[]
            for token in row:
                z,hidden=self._step(torch.tensor([token],device=c.device),hidden)
                logits.append(z[0] if prior_logits is None else z[0]+prior_logits[i])
            outputs.append(torch.stack(logits))
        return tuple(outputs)

    def objective(self,logits,observed_lists):
        rows=self._observed_lists(observed_lists)
        if type(logits) is not tuple or len(logits)!=len(rows):raise ValueError('one complete logit list per observed source list required')
        targets=[]
        for z,row in zip(logits,rows):
            if (not isinstance(z,torch.Tensor) or z.shape!=(len(row)+1,len(self.axis)+1)
                    or z.dtype!=self.output.weight.dtype or z.device!=self.output.weight.device
                    or not torch.isfinite(z).all()):raise ValueError('complete finite source-list logits required')
            targets.extend((*row,self.eos))
        return F.cross_entropy(torch.cat(logits),torch.tensor(targets,device=self.output.weight.device))

    @torch.no_grad()
    def generate(self,context,*,max_symbols,prior_logits=None):
        if type(max_symbols) is not int or not 1<=max_symbols<=self.max_target_length:
            raise ValueError('explicit bounded source-list generation budget required')
        if (not isinstance(context,torch.Tensor) or context.shape!=(self.width,)
                or context.device!=self.output.weight.device or context.dtype!=self.output.weight.dtype):
            raise ValueError('one matching source condition context required')
        if prior_logits is not None and (not isinstance(prior_logits,torch.Tensor)
                or prior_logits.shape!=(len(self.axis)+1,) or prior_logits.dtype!=context.dtype
                or prior_logits.device!=context.device or not torch.isfinite(prior_logits).all()):
            raise ValueError('complete matching source-next-symbol generation prior required')
        history=[];hidden=context[None];token=self.bos
        for _ in range(max_symbols):
            logits,hidden=self.step(torch.tensor([token],device=context.device),hidden)
            if prior_logits is not None:logits=logits+prior_logits[None]
            token=int(logits.argmax(-1).item())
            if token==self.eos:
                return dict(annotation_ids=tuple(self.axis[i] for i in history),role_ids=tuple(history),
                    termination='EOS',symbols=len(history)+1,semantics='model-produced-source-annotation-list-hypothesis')
            history.append(token)
        return dict(annotation_ids=tuple(self.axis[i] for i in history),role_ids=tuple(history),
            termination='budget-exhausted',symbols=len(history),semantics='model-produced-source-annotation-list-hypothesis')


def source_annotation_list_parameter_roles(model):
    if type(model) is not SourceAnnotationListHead:raise ValueError('exact reviewed source annotation list primitive required')
    model.get_extra_state()
    h=Axis('width','hidden');g=Axis('width','source-list-gru-reset-update-new-packed')
    inputs=Axis('finite',hashlib.sha256(('source-list-input:'+model.axis_sha256).encode()).hexdigest())
    outputs=Axis('finite',hashlib.sha256(('source-list-output:'+model.axis_sha256).encode()).hexdigest())
    roles={'embedding.weight':ParameterRole('embedding',(inputs,h)),
        'cell.weight_ih':ParameterRole('hidden',(g,h)),'cell.weight_hh':ParameterRole('hidden',(g,h)),
        'cell.bias_ih':ParameterRole('vector',(g,)),'cell.bias_hh':ParameterRole('vector',(g,)),
        'output.weight':ParameterRole('output',(outputs,h)),'output.bias':ParameterRole('vector',(outputs,))}
    _storage_signature(model);parameter_ledger(model,roles)
    return roles


def source_annotation_list_width_family(base,delta,target):
    models=(base,delta,target)
    ledgers=[parameter_ledger(m,source_annotation_list_parameter_roles(m)) for m in models]
    expected=None
    for m in models:
        cfg=m.get_extra_state();cfg.pop('width')
        if expected is None:expected=cfg
        if not exact_wire(cfg,expected):raise ValueError('source list finite axis/namespace/base/policy drift')
    if base.width!=base.base_width:raise ValueError('actual source list base must match declared base width')
    return width_family(*ledgers)
