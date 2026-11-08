"""Observational full-axis profile heads on the existing shared TokenGT.

One known molecule entity, complete SMILES/context payload; not atom/bond-expanded
or a molecular decoder. Profile conditioning is an explicit lossy linear adapter.
No cross-assay joins, image supervision, causal or generation-quality claim.
"""
from .configuration_snapshot import configuration_snapshot
import copy
import math
from .mixed_payload_projection import exact_wire
import json
import torch
from torch import nn
from mora3.data.molecule_profile import MoleculeProfileView
from mora3.data.schema import GraphRecord, Node, Measurement
from mora3.data.provenance import Provenance
from mora3.data.tasks import ConditionalTask
from .annotation_multihead import SharedAnnotationModel


class SharedMoleculeProfileModel(nn.Module):
    def __init__(self, shared, axes, *, parameterization_policy="legacy"):
        super().__init__()
        if not isinstance(shared, SharedAnnotationModel):
            raise ValueError('native shared annotation model required')
        if set(axes) != {'CellPainting','L1000'}:
            raise ValueError('both complete source axes required')
        self.axes = {m:tuple(a) for m,a in axes.items()}
        for m,n in (('CellPainting',1783),('L1000',977)):
            a=self.axes[m]
            if len(a)!=n or len(set(a))!=n or any(not isinstance(x,str) or not x for x in a):
                raise ValueError('complete unique source axis required')
        if parameterization_policy not in ('legacy','mup-molecule-profile-v1'):
            raise ValueError('known explicit molecule/profile parameterization required')
        cfg=shared.get_extra_state()
        if parameterization_policy=='legacy' and cfg['version']!=1:
            raise ValueError('new shared model requires explicit molecule/profile parameterization')
        if parameterization_policy!='legacy':shared.validate_parameterization_configuration()
        self.parameterization_policy=parameterization_policy
        self._parameterization_declaration=parameterization_policy
        self.shared=shared
        self._shared_reference=(shared,)
        self._mup_configuration=None
        width=shared.payload.embedding.embedding_dim
        self.profile_input=nn.ModuleDict({m:nn.Linear(len(a),width) for m,a in self.axes.items()})
        self.output=nn.ModuleDict({m:nn.Linear(width,len(a)) for m,a in self.axes.items()})
        if parameterization_policy!='legacy':
            self._mup_configuration=dict(schema='molecule-profile-parameterization-v1',policy=parameterization_policy,
                width=width,base_width=cfg['mup']['base_width'],output_mult=cfg['mup']['output_mult'],
                axes=copy.deepcopy(self.axes),input='native-full-finite-axis-fan-in-and-adamLR1',
                readout='native-fresh-times-sqrt-width-over-base-input-divided-by-width-over-base')
            with torch.no_grad():
                for head in self.output.values():
                    ratio=math.sqrt(width/cfg['mup']['base_width'])
                    head.weight.mul_(ratio);head.bias.mul_(ratio)
        self._mup_owned_configuration=copy.deepcopy(self._mup_configuration)
        self.to(device=shared.payload.embedding.weight.device,dtype=shared.payload.embedding.weight.dtype)

    @property
    def backbone(self):
        return self.shared.backbone

    def validate_parameterization_configuration(self):
        if self._mup_configuration is None:raise ValueError('explicit applied molecule/profile owner required')
        self.shared.validate_parameterization_configuration();cfg=self.shared.get_extra_state()
        spec=self._mup_configuration
        if (not exact_wire(self._mup_configuration,self._mup_owned_configuration)
                or self.parameterization_policy!=self._parameterization_declaration or self.shared is not self._shared_reference[0]
                or self.parameterization_policy!='mup-molecule-profile-v1' or self.axes!=spec['axes']
                or cfg['width']!=spec['width'] or cfg['mup']['base_width']!=spec['base_width']
                or cfg['mup']['output_mult']!=spec['output_mult']):
            raise ValueError('molecule/profile source/base/readout configuration drift')

    def get_extra_state(self):
        if self._mup_configuration is not None:self.validate_parameterization_configuration()
        result=dict(version=1,axes=self.axes,shared=self.shared.get_extra_state(),
            payload=self.shared.payload.get_extra_state(),representation='single-molecule-complete-SMILES-context-v1',
            profile_condition='complete-axis-linear-plus-context-v1',objective='mean-per-record-full-axis-MSE',
            molecular_decoder=False)
        if self._mup_configuration is not None:result.update(version=2,mup=copy.deepcopy(self._mup_configuration))
        return configuration_snapshot(result)

    def set_extra_state(self,state):
        if state!=self.get_extra_state():
            raise ValueError('molecule/profile configuration mismatch')

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        if self._mup_configuration is not None:
            for key,value in self.state_dict().items():
                if not isinstance(value,torch.Tensor) and not exact_wire(state_dict.get(prefix+key),value):
                    raise ValueError('molecule/profile checkpoint configuration/policy mismatch')
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def _readout_features(self,features):
        if self._mup_configuration is None:return features
        self.validate_parameterization_configuration();spec=self._mup_configuration
        return features*(spec['output_mult']/(spec['width']/spec['base_width']))

    def _validate(self,views):
        views=tuple(views)
        if not views or any(not isinstance(v,MoleculeProfileView) for v in views):
            raise ValueError('nonempty native molecule/profile views required')
        for v in views:
            if v.modality not in self.axes or v.axis!=self.axes[v.modality]:
                raise ValueError('ordered source axis mismatch')
        return views

    def condition(self,views):
        """Inputs only: target/provenance bytes are never consulted by encoding.

        Missing dose/time stay absent; cell-line missingness is explicit JSON null.
        Profile inverse interface yields shared latent conditions, not molecules.
        """
        views=self._validate(views)
        graphs=[]
        for v in views:
            x=v.inputs()
            forward=v.direction=='molecule_to_profile'
            if v.direction not in ('molecule_to_profile','profile_to_molecule'):
                raise ValueError('unsupported direction')
            expected={'assay_modality','cell_line'} | ({'canonical_isomeric_smiles','known_profile_axis'} if forward else {'source_profile_array'})
            if set(x)!=expected or x['assay_modality']!=v.modality:
                raise ValueError('condition allowlist/modality mismatch')
            context={k:x[k] for k in ('assay_modality','cell_line')}
            if forward:
                if tuple(x['known_profile_axis'])!=v.axis: raise ValueError('condition axis mismatch')
                context['canonical_isomeric_smiles']=x['canonical_isomeric_smiles']
            record=GraphRecord('input', (Node('molecule-assay',(
                Measurement('condition',json.dumps(context,sort_keys=True),None,'observed',True),
                Measurement('requested_output',None,None,'missing',False))),),(),
                (Provenance('native-input','v1','input','0'*64,'source-admission-external'),))
            task=ConditionalTask(((('node',0,'condition'),'condition'),
                                   (('node',0,'requested_output'),'target')),topology_known=True)
            graph=self.shared.payload.forward_task(record,task=task,identifiers=self.shared.sequence_identifiers())
            if not forward:
                a=x['source_profile_array']
                if tuple(a['axis_ids'])!=v.axis: raise ValueError('condition axis mismatch')
                y=graph['features'].new_tensor(a['values'])
                if y.shape!=(len(v.axis),) or not torch.isfinite(y).all():
                    raise ValueError('complete finite condition required')
                graph['features']=graph['features']+self.profile_input[v.modality](y)[None]
            graphs.append(graph)
        return self.backbone(graphs)[0][:,0]

    def forward(self,views):
        views=self._validate(views)
        if any(v.direction!='molecule_to_profile' for v in views):
            raise ValueError('prediction direction must be molecule_to_profile')
        context=self.condition(views)
        return tuple(self.output[v.modality](self._readout_features(c)) for c,v in zip(context,views))

    def objective(self,predictions,views):
        views=self._validate(views);predictions=tuple(predictions)
        if len(predictions)!=len(views): raise ValueError('one prediction per record required')
        losses=[]
        for z,v in zip(predictions,views):
            if v.direction!='molecule_to_profile':raise ValueError('prediction direction required')
            a=v.targets()['source_profile_array']
            if tuple(a['axis_ids'])!=v.axis or z.shape!=(len(v.axis),):
                raise ValueError('complete source axis required')
            y=z.new_tensor(a['values'])
            if y.shape!=z.shape or not torch.isfinite(y).all() or not torch.isfinite(z).all():
                raise ValueError('finite full-axis predictions/targets required')
            losses.append((z-y).square().mean())
        return torch.stack(losses).mean()

    def generate(self,views):
        self._validate(views)
        raise ValueError('molecular decoder not implemented; condition() returns latent interface only')
