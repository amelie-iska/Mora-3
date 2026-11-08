"""Native source-call prediction and desired-pattern conditioning on shared TokenGT.

Observed binary source assay calls are priors, not dose-response or clinical risk.
No cross-source joins or molecular decoder are supplied by this component.
"""
import json
import copy
import torch
import math
from .mixed_payload_projection import exact_wire
from torch import nn
from mora3.data.tox21 import ASSAYS, FORWARD, INVERSE, Tox21View, tox21_call_tensors
from mora3.data.schema import GraphRecord, Node, Measurement
from mora3.data.provenance import Provenance
from mora3.data.tasks import ConditionalTask
from .annotation_multihead import SharedAnnotationModel


class SharedTox21Model(nn.Module):
    def __init__(self, shared, *, parameterization_policy="legacy"):
        super().__init__()
        if not isinstance(shared, SharedAnnotationModel):
            raise ValueError('native shared annotation model required')
        if parameterization_policy not in ('legacy','mup-tox21-v1'):
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
        self.call_input=nn.Linear(24,width)
        self.output=nn.Linear(width,12)
        if parameterization_policy!='legacy':
            self._mup_configuration=dict(schema='native-assay-readout-parameterization-v1',policy=parameterization_policy,
                width=width,base_width=cfg['mup']['base_width'],output_mult=cfg['mup']['output_mult'],
                axis=ASSAYS,input='finite-source-value-and-mask-projections-invariant',
                readout='native-fresh-times-sqrt-width-over-base-input-divided-by-width-over-base')
            with torch.no_grad():
                ratio=math.sqrt(width/cfg['mup']['base_width'])
                for head in (self.output,):head.weight.mul_(ratio);head.bias.mul_(ratio)
        self._mup_owned_configuration=copy.deepcopy(self._mup_configuration)
        self.to(device=shared.payload.embedding.weight.device,dtype=shared.payload.embedding.weight.dtype)

    @property
    def backbone(self):return self.shared.backbone

    def validate_parameterization_configuration(self):
        if self._mup_configuration is None:raise ValueError('explicit applied native assay readout required')
        self.shared.validate_parameterization_configuration();cfg=self.shared._configuration;spec=self._mup_configuration
        if (not exact_wire(spec,self._mup_owned_configuration) or self.shared is not self._shared_reference[0]
                or self.parameterization_policy!='mup-tox21-v1' or cfg['width']!=spec['width']
                or cfg['mup']['base_width']!=spec['base_width'] or cfg['mup']['output_mult']!=spec['output_mult']
                or ASSAYS!=spec['axis']):
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
        state=dict(version=1,axis=ASSAYS,shared=self.shared.get_extra_state(),
            payload=self.shared.payload.get_extra_state(),representation='single-molecule-complete-SMILES-v1',
            desired_condition='binary-values-plus-observed-mask-linear-v1',
            objective='observed-call-mean-BCE-with-logits',molecular_decoder=False)
        if self._mup_configuration is not None:state.update(version=2,mup=copy.deepcopy(self._mup_configuration))
        return state

    def set_extra_state(self,state):
        if state!=self.get_extra_state():raise ValueError('Tox21 configuration mismatch')

    def _validate(self,views):
        views=tuple(views)
        if not views or any(not isinstance(v,Tox21View) or v.axis!=ASSAYS or v.direction not in (FORWARD,INVERSE) for v in views):
            raise ValueError('nonempty native ordered-axis Tox21 views required')
        return views

    def condition(self,views):
        """Encode only input bytes; never consume targets, audit or source identifiers."""
        return self._condition_inputs(tuple((view.direction, view.inputs()) for view in self._validate(views)))

    def _condition_inputs(self, rows):
        if not rows or any(direction not in (FORWARD, INVERSE) for direction, _ in rows):
            raise ValueError('Nonempty native Tox21 direction and input fields required')
        graphs=[]
        for direction, x in rows:
            forward=direction==FORWARD
            expected={'known_assay_axis'}|({'canonical_isomeric_smiles'} if forward else {'desired_source_call_values','desired_source_call_mask'})
            if set(x)!=expected or tuple(x['known_assay_axis'])!=ASSAYS:
                raise ValueError('condition allowlist/axis mismatch')
            context=dict(known_assay_axis=ASSAYS,physical_dose=None,exposure_time=None,cell_line=None,
                role='observed_molecule' if forward else 'desired_source_call_pattern')
            if forward:
                if not isinstance(x['canonical_isomeric_smiles'],str) or not x['canonical_isomeric_smiles'].strip():
                    raise ValueError('complete source molecule required')
                context['canonical_isomeric_smiles']=x['canonical_isomeric_smiles']
            record=GraphRecord('input',(Node('molecule-assay',(
                Measurement('condition',json.dumps(context,sort_keys=True),None,'observed',True),
                Measurement('requested_output',None,None,'missing',False))),),(),
                (Provenance('native-input','v1','input','0'*64,'source-admission-external'),))
            task=ConditionalTask(((('node',0,'condition'),'condition'),(('node',0,'requested_output'),'target')),topology_known=True)
            graph=self.shared.payload.forward_task(record,task=task,identifiers=self.shared.sequence_identifiers())
            if not forward:
                vals=x['desired_source_call_values'];mask=x['desired_source_call_mask']
                if (len(vals)!=12 or len(mask)!=12 or any(type(m)is not bool for m in mask) or not any(mask) or
                    any((type(y)is not int or y not in (0,1)) if m else y is not None for y,m in zip(vals,mask))):
                    raise ValueError('observed binary calls and explicit unknown mask required')
                features=graph['features'].new_tensor([y if m else 0 for y,m in zip(vals,mask)]+list(mask))
                graph['features']=graph['features']+self.call_input(features)[None]
            graphs.append(graph)
        return self.backbone(graphs)[0][:,0]

    def forward(self,views):
        views=self._validate(views)
        if any(v.direction!=FORWARD for v in views):raise ValueError('prediction direction required')
        return self.output(self._readout_features(self.condition(views)))

    def objective(self,logits,views):
        views=self._validate(views)
        if logits.shape!=(len(views),12) or not torch.isfinite(logits).all():raise ValueError('finite full-axis logits required')
        for v in views:
            target=v.targets()
            if (set(target)!={'source_call_values','source_call_mask'} or
                tuple(target['source_call_values'])!=v.values or tuple(target['source_call_mask'])!=v.observed_mask):
                raise ValueError('target bytes and source calls disagree')
        values,mask=tox21_call_tensors(views,role='target',device=logits.device)
        # Tensors are reconstructed from immutable source view fields by the reader.
        return nn.functional.binary_cross_entropy_with_logits(logits,values.to(logits.dtype),reduction='none')[mask].mean()

    def generate(self,views):
        self._validate(views)
        raise ValueError('Tox21 molecular decoder not implemented; latent conditioning only')
