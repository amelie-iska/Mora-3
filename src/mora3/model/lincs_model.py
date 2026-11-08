"""Native LINCS full-axis heads sharing TokenGT; separate from legacy Rosetta.

Native numeric values are cast to model precision only at the neural boundary.
Missing coordinates use zero storage plus an explicit learned observation mask;
this is not source imputation. Assay units/scales remain separate. Reverse
conditioning returns a latent interface, not a molecule or a trained decoder.
"""
from .configuration_snapshot import configuration_snapshot
import copy
import json
import torch
import math
from .mixed_payload_projection import exact_wire
from torch import nn
from mora3.data.lincs_native import LincsNativeView, NativeProfileArray, _axis, lincs_input_bytes
from mora3.data.schema import GraphRecord, Node, Measurement
from mora3.data.provenance import Provenance
from mora3.data.tasks import ConditionalTask
from .annotation_multihead import SharedAnnotationModel
from .lincs_objective import native_profile_loss


class SharedLincsModel(nn.Module):
    def __init__(self, shared, source_axes, *, parameterization_policy="legacy"):
        super().__init__()
        if not isinstance(shared, SharedAnnotationModel) or set(source_axes)!={'CellPainting','L1000'}:
            raise ValueError('shared annotation model and both native source axes required')
        self.source_axes=copy.deepcopy(source_axes)
        self.axes={m:_axis(a,m)[0] for m,a in self.source_axes.items()}
        if parameterization_policy not in ('legacy','mup-lincs-v1'):
            raise ValueError('known explicit native assay readout policy required')
        cfg=shared.get_extra_state()
        if parameterization_policy=='legacy' and cfg['version']!=1:
            raise ValueError('new shared owner requires explicit native assay parameterization')
        if parameterization_policy!='legacy':shared.validate_parameterization_configuration()
        self.parameterization_policy=parameterization_policy
        self._shared_reference=(shared,)
        self._mup_configuration=None
        self.shared=shared
        width=shared.payload.embedding.embedding_dim
        self.profile_input=nn.ModuleDict({m:nn.Linear(len(a),width) for m,a in self.axes.items()})
        self.profile_mask=nn.ModuleDict({m:nn.Linear(len(a),width,bias=False) for m,a in self.axes.items()})
        self.output=nn.ModuleDict({m:nn.Linear(width,len(a)) for m,a in self.axes.items()})
        if parameterization_policy!='legacy':
            self._mup_configuration=dict(schema='native-assay-readout-parameterization-v1',policy=parameterization_policy,
                width=width,base_width=cfg['mup']['base_width'],output_mult=cfg['mup']['output_mult'],
                axes=copy.deepcopy(self.axes),source_axes=copy.deepcopy(self.source_axes),input='finite-source-value-and-mask-projections-invariant',
                readout='native-fresh-times-sqrt-width-over-base-input-divided-by-width-over-base')
            with torch.no_grad():
                ratio=math.sqrt(width/cfg['mup']['base_width'])
                for head in self.output.values():head.weight.mul_(ratio);head.bias.mul_(ratio)
        self._mup_owned_configuration=copy.deepcopy(self._mup_configuration)
        self.to(device=shared.payload.embedding.weight.device,dtype=shared.payload.embedding.weight.dtype)

    def validate_parameterization_configuration(self):
        if self._mup_configuration is None:raise ValueError('explicit applied native assay readout required')
        self.shared.validate_parameterization_configuration();cfg=self.shared._configuration;spec=self._mup_configuration
        if (not exact_wire(spec,self._mup_owned_configuration) or self.shared is not self._shared_reference[0]
                or self.parameterization_policy!='mup-lincs-v1' or cfg['width']!=spec['width']
                or cfg['mup']['base_width']!=spec['base_width'] or cfg['mup']['output_mult']!=spec['output_mult']
                or self.axes!=spec['axes'] or not exact_wire(self.source_axes,spec['source_axes'])):
            raise ValueError('native assay source/axis/base/readout configuration drift')

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        if self._mup_configuration is not None:
            for key,value in self.state_dict().items():
                if not isinstance(value,torch.Tensor) and not exact_wire(state_dict.get(prefix+key),value):
                    raise ValueError('native assay checkpoint source/axis/policy configuration mismatch')
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def _readout_features(self,features):
        if self._mup_configuration is None:return features
        self.validate_parameterization_configuration();spec=self._mup_configuration
        return features*(spec['output_mult']/(spec['width']/spec['base_width']))

    def get_extra_state(self):
        if self._mup_configuration is not None:self.validate_parameterization_configuration()
        state=dict(version=1,axes=self.axes,source_axes=self.source_axes,
            shared=self.shared.get_extra_state(),payload=self.shared.payload.get_extra_state(),
            condition='native-values-plus-observed-mask-and-source-coordinate-context-v1',
            objective='per-record-observed-coordinate-MSE;endpoint-scales-not-calibrated',
            molecular_decoder=False)
        if self._mup_configuration is not None:state.update(version=2,mup=copy.deepcopy(self._mup_configuration))
        return configuration_snapshot(state)

    def set_extra_state(self,state):
        if state!=self.get_extra_state(): raise ValueError('native LINCS configuration mismatch')

    def _validate(self,views):
        views=tuple(views)
        if not views or any(not isinstance(v,LincsNativeView) for v in views):
            raise ValueError('nonempty source-bound LINCS views required')
        for v in views:
            if v.modality not in self.axes or v.axis!=self.axes[v.modality]:
                raise ValueError('ordered native source axis mismatch')
        return views

    def condition(self,views):
        graphs=[]
        for v in self._validate(views):
            # Validate the input allowlist independently without touching targets.
            raw=lincs_input_bytes({'conditions':v.inputs()},modality=v.modality,
                direction=v.direction,source_axis=self.source_axes[v.modality])
            context=json.loads(raw)
            array=context.pop('source_profile_native_array',None)
            context.pop('known_profile_axis',None)
            context.pop('known_native_feature_types',None)
            record=GraphRecord('input',(Node('molecule-assay',(
                Measurement('condition',json.dumps(context,sort_keys=True),None,'observed',True),
                Measurement('requested_output',None,None,'missing',False))),),(),
                (Provenance('native-input','v1','input','0'*64,'source-admission-external'),))
            task=ConditionalTask(((('node',0,'condition'),'condition'),
                (('node',0,'requested_output'),'target')),topology_known=True)
            graph=self.shared.payload.forward_task(record,task=task,identifiers=self.shared.sequence_identifiers())
            if array is not None:
                a=NativeProfileArray.from_mapping(array,source_axis=self.source_axes[v.modality])
                values=graph['features'].new_tensor([0 if x is None else x for x in a.values])
                mask=graph['features'].new_tensor(a.observed_mask)
                if not torch.isfinite(values).all():raise ValueError('native values exceed model precision')
                graph['features']=graph['features']+(self.profile_input[v.modality](values)
                    +self.profile_mask[v.modality](mask))[None]
            graphs.append(graph)
        return self.shared.backbone(graphs)[0][:,0]

    def forward(self,views):
        views=self._validate(views)
        if any(v.direction!='molecule_to_profile' for v in views):
            raise ValueError('prediction direction must be molecule_to_profile')
        return tuple(self.output[v.modality](self._readout_features(c)) for c,v in zip(self.condition(views),views))

    def objective(self,predictions,views):
        views=self._validate(views); predictions=tuple(predictions)
        if len(predictions)!=len(views): raise ValueError('one prediction per view required')
        losses=[]
        for p,v in zip(predictions,views):
            if v.direction!='molecule_to_profile':raise ValueError('prediction direction required')
            a=NativeProfileArray.from_mapping(v.targets()['source_profile_native_array'],
                source_axis=self.source_axes[v.modality])
            y=p.new_tensor([float('nan') if x is None else x for x in a.values])
            mask=torch.tensor(a.observed_mask,device=p.device,dtype=torch.bool)
            losses.append(native_profile_loss(p,y,mask,modality=v.modality))
        return torch.stack(losses).mean()

    def generate(self,views):
        self._validate(views)
        raise ValueError('native LINCS molecular decoder not implemented; latent conditioning only')
