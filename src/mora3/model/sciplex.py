"""Observational full-axis adapters into the existing shared TokenGT reference.

Learned dense expression projection is explicitly lossy, uses every ordered gene,
not gene-token attention. Context->observation edge denotes assay conditioning,
not a causal or molecular interaction. No chemical structures are inferred.
"""
import copy
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from mora3.data.sciplex_admission import ObservationalSciPlexTask,POLICY
from .tokengt import TokenGT, _validate_native_policy
from .tokengt import _native_mup_declaration
from .sciplex_preparation import PreparedSciPlexTask


def _same_configuration(left,right):
    """Exact wire types prevent bool/int and nested mutable-config aliases."""
    if type(left) is not type(right):return False
    if type(left) is dict:
        return (left.keys()==right.keys()
                and all(_same_configuration(left[k],right[k]) for k in left))
    if type(left) is tuple:
        return len(left)==len(right) and all(_same_configuration(a,b) for a,b in zip(left,right))
    return left==right


def _owner_configuration(axis_ids,width,attention,scaling_policy,attn_mult,
                         initialization_policy='legacy',encoder_var=1.0,
                         parameterization_policy='legacy',base_width=None,readout_mult=1.0):
    """Pure constructor preflight, before any learned-layer initialization."""
    axis_ids=tuple(axis_ids)
    if not axis_ids or any(not isinstance(x,str) or not x.strip() for x in axis_ids):
        raise ValueError('nonempty ordered source gene axis required')
    _validate_native_policy(width,2,width,attention,32,scaling_policy,attn_mult,
                            initialization_policy,encoder_var)
    if (parameterization_policy not in ('legacy','mup-sciplex-v1')
            or type(readout_mult) not in (int,float) or not math.isfinite(readout_mult)
            or readout_mult<=0
            or (parameterization_policy=='legacy' and (base_width is not None or readout_mult!=1))):
        raise ValueError('explicit supported native parameterization/readout declaration required')
    native_mup=_native_mup_declaration(width,2,width,initialization_policy,scaling_policy,
        'mup-tokengt-v1' if parameterization_policy!='legacy' else 'legacy',base_width,base_width)
    config=dict(version=1,axis_ids=axis_ids,width=width,attention=attention,
                policy=POLICY,transform='log1p-counts',graph='context-to-observation-v1')
    if scaling_policy!='legacy':
        config.update(version=2,native_backbone=dict(schema='tokengt-native-configuration-v1',
            feature_width=width,id_width=2,width=width,heads=2,layers=2,ffn_width=width,
            attention=attention,tropical_block_size=32,scaling_policy=scaling_policy,
            attn_mult=float(attn_mult)))
        if initialization_policy!='legacy':
            config['version']=3
            config['native_backbone'].update(schema='tokengt-native-configuration-v2',
                initialization_policy=initialization_policy,encoder_var=float(encoder_var))
    if native_mup is not None:
        config.update(version=4,mup=dict(schema='sciplex-parameterization-v1',policy=parameterization_policy,
            base_width=base_width,output_mult=float(readout_mult),
            readout='finite-output-MuReadout-fresh-SP-rescale-and-input-width-division'))
        config['native_backbone'].update(schema='tokengt-native-configuration-v3',parameterization=native_mup)
    return config


class SciPlexSharedModel(nn.Module):
    def __init__(self, axis_ids, *, width=16, attention='vanilla',
                 scaling_policy='legacy',attn_mult=1.0,initialization_policy='legacy',encoder_var=1.0,
                 parameterization_policy='legacy',base_width=None,readout_mult=1.0):
        super().__init__()
        self.config=_owner_configuration(axis_ids,width,attention,scaling_policy,attn_mult,
            initialization_policy,encoder_var,parameterization_policy,base_width,readout_mult)
        self.preparation_cache=None  # opt-in runtime-only; no learned tensors cached
        self.axis_ids=self.config['axis_ids']
        self.expression=nn.Linear(len(self.axis_ids),width)
        self.intervention=nn.Linear(6,width)
        self.context=nn.Linear(1,width)
        self.edge=nn.Parameter(torch.zeros(width))
        self.direction=nn.Embedding(2,width)
        self.backbone=TokenGT(width,2,width=width,heads=2,layers=2,attention=attention,
            scaling_policy=scaling_policy,attn_mult=attn_mult,
            initialization_policy=initialization_policy,encoder_var=encoder_var,
            parameterization_policy='mup-tokengt-v1' if parameterization_policy!='legacy' else 'legacy',
            base_width=base_width,base_ffn_width=base_width)
        self._owner_configuration = copy.deepcopy(self.config)
        self._backbone_configuration = self.backbone.configuration()
        self.response_head=nn.Linear(width,len(self.axis_ids))
        self.intervention_head=nn.Linear(width,6)
        if parameterization_policy!='legacy':
            # MuReadout._rescale_parameters: one fresh SP->base-variance conversion.
            with torch.no_grad():
                for head in (self.response_head,self.intervention_head):
                    head.weight.mul_(math.sqrt(width/base_width))
                    head.bias.mul_(math.sqrt(width/base_width))

    @classmethod
    def from_configuration(cls,config):
        """Reconstruct exact v1 or opt-in v2; refuse malformed metadata before RNG.

        Tensor state must be loaded strictly afterward. This does not upgrade
        external flow/cooperative schemas or qualify initialization/optimizer
        scaling, source admission or whole-model width transfer.
        """
        keys={'version','axis_ids','width','attention','policy','transform','graph'}
        if (type(config) is not dict or type(config.get('version')) is not int
                or config['version'] not in (1,2,3,4)
                or set(config)!=(keys|{'native_backbone','mup'} if config['version']==4 else
                                 keys|{'native_backbone'} if config['version']>=2 else keys)
                or type(config.get('axis_ids')) is not tuple):
            raise ValueError('exact versioned SciPlex owner configuration required')
        policy,mult,init,var='legacy',1.0,'legacy',1.0
        mup,base,out='legacy',None,1.
        if config['version']>=2:
            native=config['native_backbone']
            if (type(native) is not dict or 'scaling_policy' not in native
                    or 'attn_mult' not in native or native['scaling_policy']!='mup-v1'):
                raise ValueError('explicit opt-in native configuration required')
            policy,mult=native['scaling_policy'],native['attn_mult']
            if config['version']>=3:
                init,var=native.get('initialization_policy'),native.get('encoder_var')
                if init!='mup-transformer-v1':raise ValueError('explicit new native initialization required')
        if config['version']==4:
            spec=config['mup']
            if type(spec) is not dict:
                raise ValueError('exact native readout parameterization configuration required')
            mup,base,out=spec.get('policy'),spec.get('base_width'),spec.get('output_mult')
        expected=_owner_configuration(config['axis_ids'],config['width'],config['attention'],
                                      policy,mult,init,var,mup,base,out)
        if not _same_configuration(config,expected):
            raise ValueError('SciPlex owner/native configuration mismatch')
        return cls(config['axis_ids'],width=config['width'],attention=config['attention'],
            scaling_policy=policy,attn_mult=mult,initialization_policy=init,encoder_var=var,
            parameterization_policy=mup,base_width=base,readout_mult=out)

    def validate_configuration(self):
        """Bind each versioned owner to its exact native attention configuration.

        Metadata checks do not certify task admission or migrate checkpoints.
        Shared image/annotation adapters retain their separately owned schemas.
        """
        try:
            actual = self.backbone.configuration()
        except (ValueError, AttributeError) as exc:
            raise ValueError('SciPlex backbone configuration drift') from exc
        if (not _same_configuration(self.config,self._owner_configuration)
                or self.axis_ids != self._owner_configuration['axis_ids']
                or not _same_configuration(actual,self._backbone_configuration)
                or (self.config['version']>=2
                    and not _same_configuration(self.config['native_backbone'],actual))):
            raise ValueError('SciPlex owner/backbone configuration mismatch')

    def get_extra_state(self):
        self.validate_configuration()
        return copy.deepcopy(self.config)

    def set_extra_state(self,state):
        self.validate_configuration()
        if not _same_configuration(state,self.config):raise ValueError('incompatible axis/model configuration')

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,
                              missing_keys,unexpected_keys,error_msgs):
        incoming=state_dict.get(prefix+'_extra_state')
        # Preserve old v1 state interpretation. Explicit opt-in metadata must be
        # checked before the owner's edge tensor, not only later child loading.
        if self.config['version']>=2 or (type(incoming) is dict and incoming.get('version') in (2,3,4)):
            self.validate_configuration()
            if not _same_configuration(incoming,self.config):
                raise ValueError('SciPlex opt-in checkpoint configuration mismatch')
            self.backbone._validate_initialization_checkpoint(state_dict,prefix+'backbone.')
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,
                                     missing_keys,unexpected_keys,error_msgs)

    def binding(self,tasks):
        return dict(config=self.get_extra_state(),tasks=[t.canonical_bytes() for t in tasks])

    def project(self, task):
        w=self.edge
        if self.preparation_cache is None:
            return task.project(device=w.device,dtype=w.dtype,for_training=True)
        return self.preparation_cache.project(task,axis_ids=self.axis_ids,
            device=w.device,dtype=w.dtype,for_training=True)

    def graph(self,task):
        self.validate_configuration()
        from .response_sampling import ResponseContext
        if isinstance(task,ResponseContext):
            return task.graph(self)
        if not isinstance(task,ObservationalSciPlexTask):
            raise ValueError('observational admission required')
        if isinstance(task, PreparedSciPlexTask):
            direction=task.prepared_direction;axis=task.prepared_axis_ids
        else:
            raw=json.loads(task.task.sidecar_bytes)
            direction=raw['task'];where='targets' if direction=='response' else 'conditions'
            axis=tuple(raw[where]['expression']['axis_ids'])
        if axis!=self.axis_ids:
            raise ValueError('gene order mismatch')
        w=self.edge;p=self.project(task)['inputs']
        if direction=='response':
            z=torch.cat((F.one_hot(p['intervention_index'],5).to(w.dtype),p['dose_coordinate'].reshape(1)))
            observed=self.intervention(z)
        else:observed=self.expression(torch.log1p(p['expression']))
        context=self.context((p['exposure_hours']/24).reshape(1))
        # Exact two-dimensional orthogonal IDs; no RNG or hidden target topology.
        return dict(features=torch.stack((context,observed,self.edge))+
                    self.direction.weight[int(direction=='intervention')],
            endpoints=torch.tensor([[0,0],[1,1],[0,1]],device=w.device),
            roles=torch.tensor([0,0,1],device=w.device),
            identifiers=torch.eye(2,device=w.device,dtype=w.dtype))

    def forward(self,tasks):
        if not tasks:raise ValueError('nonempty homogeneous task batch required')
        directions=[(t.prepared_direction if isinstance(t,PreparedSciPlexTask) else t.input_metadata()['task']) for t in tasks]
        if len(set(directions))!=1:raise ValueError('homogeneous masks required')
        x=self.backbone([self.graph(t) for t in tasks])[0][:,0]
        if self.config['version']==4:
            spec=self.config['mup']
            x=x*(spec['output_mult']/(self.config['width']/spec['base_width']))
        return (self.response_head if directions[0]=='response' else self.intervention_head)(x)

    def loss(self,tasks):
        out=self(tasks);w=self.edge
        ps=[self.project(t) for t in tasks]
        if (tasks[0].prepared_direction if isinstance(tasks[0],PreparedSciPlexTask) else tasks[0].input_metadata()['task'])=='response':
            target=torch.stack([torch.log1p(p['targets']['expression']) for p in ps])
            mask=torch.stack([p['target_masks']['expression'] for p in ps])
            if not mask.all():raise ValueError('complete observed targets required')
            return F.mse_loss(out,target)
        return F.cross_entropy(out[:,:5],torch.stack([p['targets']['intervention_index'] for p in ps]))+F.mse_loss(out[:,5],torch.stack([p['targets']['dose_coordinate'] for p in ps]))
