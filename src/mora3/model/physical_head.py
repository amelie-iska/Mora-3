"""Assertion-query source-entry metadata prediction, never numeric properties.

Sequence latents are shared externally; only parameter type enters the query.
Unit/scope/method targets never enter features. All parent data stays in the
PhysicalBatch binding. Not an assertion generator or canonical property head.
"""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from .mixed_payload_projection import exact_wire


class StructuredPhysicalHead(nn.Module):
    def __init__(self,width,parameter_axis,method_axis,*,parameterization_policy='legacy',
                 base_width=None,output_mult=1.):
        super().__init__()
        if (parameterization_policy not in ('legacy','mup-physical-v1')
                or type(output_mult) not in (int,float) or not math.isfinite(output_mult) or output_mult<=0
                or (parameterization_policy=='legacy' and (base_width is not None or output_mult!=1))):
            raise ValueError('explicit supported physical readout parameterization required')
        new=parameterization_policy!='legacy'
        if new and (type(width) is not int or type(base_width) is not int or base_width<=0 or width<base_width):
            raise ValueError('compatible positive physical base width required')
        self.parameter_axis=tuple(parameter_axis)
        self.method_axis=tuple(method_axis)
        for axis in (self.parameter_axis,self.method_axis):
            if not axis or len(set(axis))!=len(axis) or any(not isinstance(s,str) or not s for s in axis):
                raise ValueError('nonempty unique axis required')
        self.configuration=dict(version=1,width=width,parameters=self.parameter_axis,
                                methods=self.method_axis,objective='observed-slot-CE-sum')
        self.query=nn.Embedding(len(self.parameter_axis),width)
        self.unit=nn.Linear(width,2)
        self.scope=nn.Linear(width,2)
        self.method=nn.Linear(width,len(self.method_axis))
        self._mup_spec=None
        if new:
            self._mup_spec=dict(schema='physical-readout-parameterization-v1',policy=parameterization_policy,
                               base_width=base_width,output_mult=float(output_mult))
            self.configuration.update(version=2,mup=copy.deepcopy(self._mup_spec))
            with torch.no_grad():
                for head in (self.unit,self.scope,self.method):
                    head.weight.mul_(math.sqrt(width/base_width));head.bias.mul_(math.sqrt(width/base_width))

    def validate_parameterization_configuration(self):
        if self._mup_spec is None or not exact_wire(self.configuration.get('mup'),self._mup_spec):
            raise ValueError('physical parameterization configuration drift')

    def get_extra_state(self):
        if self._mup_spec is not None:self.validate_parameterization_configuration()
        return copy.deepcopy(self.configuration)

    def set_extra_state(self,state):
        if state!=self.configuration:
            raise ValueError('physical head configuration mismatch')

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,
                              missing_keys,unexpected_keys,error_msgs):
        incoming=state_dict.get(prefix+'_extra_state')
        if self._mup_spec is not None or (type(incoming) is dict and 'mup' in incoming):
            if not exact_wire(incoming,self.get_extra_state()):
                raise ValueError('physical readout checkpoint parameterization mismatch')
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,
                                     missing_keys,unexpected_keys,error_msgs)

    def _rows(self,targets):
        return [(i,r) for i,group in enumerate(targets) for r in group]

    def forward(self,latent,targets):
        if latent.ndim!=2 or latent.shape!=(len(targets),self.configuration['width']):
            raise ValueError('one latent per source entry required')
        rows=self._rows(targets)
        if any(r['parameter_type'] not in self.parameter_axis for _,r in rows):
            raise ValueError('parameter axis omits assertion')
        indices=torch.tensor([i for i,_ in rows],device=latent.device,dtype=torch.long)
        kinds=torch.tensor([self.parameter_axis.index(r['parameter_type']) for _,r in rows],device=latent.device,dtype=torch.long)
        features=latent[indices]+self.query(kinds)
        if self._mup_spec is not None:
            self.validate_parameterization_configuration();spec=self._mup_spec
            features=features*(spec['output_mult']/(self.configuration['width']/spec['base_width']))
        return {name:getattr(self,name)(features) for name in ('unit','scope','method')}

    def objective(self,logits,targets):
        rows=self._rows(targets)
        if set(logits)!={'unit','scope','method'}:
            raise ValueError('all structured slots required')
        labels={'unit':[],'scope':[],'method':[]}
        for _,r in rows:
            if r['unit_status'] not in ('unknown','explicit_source_unit'):
                raise ValueError('unsupported unit status')
            if r['method'] is not None and r['method'] not in self.method_axis:
                raise ValueError('method axis omits observed method')
            labels['unit'].append(int(r['unit_status']=='explicit_source_unit'))
            labels['scope'].append(int(r['molecule_scope'] is not None))
            labels['method'].append(-100 if r['method'] is None else self.method_axis.index(r['method']))
        total=0
        for name,z in logits.items():
            expected=2 if name!='method' else len(self.method_axis)
            if z.shape!=(len(rows),expected):
                raise ValueError('structured logits shape mismatch')
            y=torch.tensor(labels[name],device=z.device,dtype=torch.long)
            active=y!=-100
            total=total+(F.cross_entropy(z[active],y[active]) if active.any() else z.sum()*0)
        return total
