"""Endpoint/assay-local ADMET heads on the existing shared TokenGT encoder.

This interface has no observational admission adapter and no molecular decoder.
Gaussian scales are native-coordinate parameters, not calibrated uncertainties.
"""
import copy
import math
from .mixed_payload_projection import exact_wire
from decimal import Decimal
import torch
from torch import nn
from torch.nn import functional as F
from mora3.data.admet import (AdmetPredictionView, AdmetObservation, admet_condition_graph,
                             canonical, DesiredProperty)
from .annotation_multihead import SharedAnnotationModel
from .admet_objectives import (AdmetObjectiveContract, require_observational_admet_admission,
                               require_single_admet_partition, desired_interval_penalty)


class SharedAdmetModel(nn.Module):
    def __init__(self, shared, contracts, *, parameterization_policy="legacy"):
        super().__init__()
        contracts=tuple(contracts)
        if not isinstance(shared,SharedAnnotationModel) or not contracts:
            raise ValueError('shared annotation model and explicit ADMET partitions required')
        for c in contracts:
            if (not isinstance(c,AdmetObjectiveContract) or c.molecule_class is None
                    or c.definition.applicability_for(c.molecule_class)!='applicable'
                    or c.definition.objective_kind not in ('binary','regression')):
                raise ValueError('resolved endpoint and explicitly applicable molecule class required')
            c._check(c.definition.objective_kind,c.unit)
        if len({c.partition_key for c in contracts})!=len(contracts):
            raise ValueError('duplicate ADMET partitions')
        if parameterization_policy not in ('legacy','mup-admet-v1'):
            raise ValueError('known explicit ADMET parameterization required')
        cfg=shared.get_extra_state()
        if parameterization_policy=='legacy' and cfg['version']!=1:
            raise ValueError('new shared owner requires explicit ADMET readout policy')
        if parameterization_policy!='legacy':shared.validate_parameterization_configuration()
        self.parameterization_policy=parameterization_policy
        self._shared_reference=(shared,)
        self._mup_configuration=None
        self.shared=shared
        self.contracts={c.partition_key:c for c in contracts}
        width=shared.payload.embedding.embedding_dim
        self.heads=nn.ModuleDict({c.partition_key:nn.Linear(width,1 if c.definition.objective_kind=='binary' else 2)
                                 for c in contracts})
        if parameterization_policy!='legacy':
            self._mup_configuration=dict(schema='admet-partition-readout-parameterization-v1',policy=parameterization_policy,
                width=width,base_width=cfg['mup']['base_width'],output_mult=cfg['mup']['output_mult'],
                partitions=tuple(self.contracts),readout='native-fresh-times-sqrt-width-over-base-input-divided-by-width-over-base')
            with torch.no_grad():
                ratio=math.sqrt(width/cfg['mup']['base_width'])
                for head in self.heads.values():head.weight.mul_(ratio);head.bias.mul_(ratio)
        self._mup_owned_configuration=copy.deepcopy(self._mup_configuration)
        self.to(device=shared.payload.embedding.weight.device,dtype=shared.payload.embedding.weight.dtype)

    def validate_parameterization_configuration(self):
        if self._mup_configuration is None:raise ValueError('explicit applied ADMET readout required')
        self.shared.validate_parameterization_configuration();cfg=self.shared.get_extra_state();spec=self._mup_configuration
        if (not exact_wire(spec,self._mup_owned_configuration) or self.shared is not self._shared_reference[0]
                or self.parameterization_policy!='mup-admet-v1' or set(self._modules)!={'shared','heads'}
                or tuple(self.contracts)!=spec['partitions'] or type(self.heads) is not nn.ModuleDict
                or set(self.heads)!=set(self.contracts) or cfg['width']!=spec['width']
                or cfg['mup']['base_width']!=spec['base_width'] or cfg['mup']['output_mult']!=spec['output_mult']):
            raise ValueError('ADMET source partition/base/readout configuration drift')
        for key,contract in self.contracts.items():
            head=self.heads[key];outputs=1 if contract.definition.objective_kind=='binary' else 2
            if (key!=contract.partition_key or type(head) is not nn.Linear or head._modules or head._buffers
                    or (head.in_features,head.out_features)!=(spec['width'],outputs)
                    or tuple(head.weight.shape)!=(outputs,spec['width']) or tuple(head.bias.shape)!=(outputs,)):
                raise ValueError('ADMET typed partition/readout operator drift')

    def get_extra_state(self):
        if self._mup_configuration is not None:self.validate_parameterization_configuration()
        result=dict(version=1,partitions=tuple(self.contracts),
            shared=self.shared.get_extra_state(),payload=self.shared.payload.get_extra_state(),
            scale='softplus-plus-dtype-epsilon-native-coordinate',observational_admission=False,molecular_decoder=False)
        if self._mup_configuration is not None:result.update(version=2,mup=copy.deepcopy(self._mup_configuration))
        return copy.deepcopy(result)

    def set_extra_state(self,state):
        if state!=self.get_extra_state():raise ValueError('ADMET partition/configuration mismatch')

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        if self._mup_configuration is not None:
            for key,value in self.state_dict().items():
                if not isinstance(value,torch.Tensor) and not exact_wire(state_dict.get(prefix+key),value):
                    raise ValueError('ADMET checkpoint source/partition/policy configuration mismatch')
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def _readout_features(self,features):
        if self._mup_configuration is None:return features
        self.validate_parameterization_configuration();spec=self._mup_configuration
        return features*(spec['output_mult']/(spec['width']/spec['base_width']))

    def condition(self,inputs):
        inputs=tuple(inputs)
        if not inputs:raise ValueError('nonempty target-free ADMET queries required')
        graphs=[]
        for query in inputs:
            record,task=admet_condition_graph(query)
            graphs.append(self.shared.payload.forward_task(record,task=task,
                          identifiers=self.shared.sequence_identifiers()))
        return self.shared.backbone(graphs)[0][:,0]

    def _validate(self,views,contracts):
        views=tuple(views);contracts=tuple(contracts)
        if not views or len(views)!=len(contracts):raise ValueError('one explicit partition per prediction view required')
        for v,c in zip(views,contracts):
            if (not isinstance(v,AdmetPredictionView) or not isinstance(c,AdmetObjectiveContract)
                    or c.partition_key not in self.contracts):
                raise ValueError('registered ADMET partition and typed prediction view required')
            inputs=v.inputs()
            if (inputs.get('role')!='source_native_ADMET_prediction_query'
                    or canonical(inputs.get('endpoint'))!=canonical(c.definition.query())
                    or canonical(inputs.get('assay_context'))!=c.assay_context.native_bytes
                    or inputs.get('native_entity',{}).get('molecule_class')!=c.molecule_class):
                raise ValueError('endpoint/class/native-unit/assay partition mismatch')
        return views,contracts

    def forward(self,views,contracts):
        views,contracts=self._validate(views,contracts)
        z=self.condition(v.inputs() for v in views)
        result=[]
        for row,c in zip(z,contracts):
            raw=self.heads[c.partition_key](self._readout_features(row))
            result.append(dict(logit=raw[0]) if c.definition.objective_kind=='binary' else
                          dict(mean=raw[0],scale=F.softplus(raw[1])+torch.finfo(raw.dtype).eps))
        return tuple(result)

    def _objective_coordinates(self,predictions,views,contracts):
        """Check homogeneous partition and scalar head output precision without casting."""
        views,contracts=self._validate(views,contracts)
        contract=require_single_admet_partition(contracts)
        predictions=tuple(predictions)
        if len(predictions)!=len(views):
            raise ValueError('one head output per typed view required')
        keys=('logit',) if contract.definition.objective_kind=='binary' else ('mean','scale')
        reference=self.heads[contract.partition_key].weight
        for output in predictions:
            if not isinstance(output,dict) or set(output)!=set(keys):
                raise ValueError('exact endpoint-specific head output required')
            for value in output.values():
                if (not isinstance(value,torch.Tensor) or value.ndim!=0
                        or value.dtype not in (torch.float32,torch.float64)
                        or value.dtype!=reference.dtype or value.device!=reference.device):
                    raise ValueError('scalar head output must preserve model device/precision')
        return contract,{key:torch.stack([p[key] for p in predictions]) for key in keys}

    def reference_objective(self,predictions,views,contracts,*,native_unit,**targets_and_masks):
        """Tensor reference interface ONLY, not source observational admission.

        Explicit synthetic/reference targets exercise shared-head gradients. No
        source cells, Decimal conversion, implicit unit conversion, cross-partition
        pooling or model-supervision eligibility are established by this method.
        Actual source supervision must enter observed_loss and its admission gate.
        """
        contract,coordinates=self._objective_coordinates(predictions,views,contracts)
        if contract.definition.objective_kind=='binary':
            return contract.binary_loss(coordinates['logit'],native_unit=native_unit,**targets_and_masks)
        return contract.censored_loss(coordinates['mean'],coordinates['scale'],
                                      native_unit=native_unit,**targets_and_masks)

    def desired_objective(self,predictions,views,contracts,*,native_unit,lower,upper,active):
        """Separate native-coordinate regression desired-set distance surrogate.

        Bounds are explicit desired coordinates, not experimental observations.
        Binary probability constraints require a separately defined interface.
        """
        contract,coordinates=self._objective_coordinates(predictions,views,contracts)
        contract._check('regression',native_unit)
        return desired_interval_penalty(coordinates['mean'],lower,upper,active=active)

    def assess_desired(self,predictions,views,contracts,requests,*,native_unit,decimal_policy):
        """Typed regression desired-set assessment of candidate predicted means.

        Explicit nearest-model-dtype conversion uses Python float then the model
        dtype; original Decimal literals are retained. Overflow, nonzero-to-zero
        underflow and collapsed intervals reject. Membership is evaluated in that
        declared floating coordinate, not exact Decimal or uncertainty space.
        Squared distance to the closed set is a differentiable surrogate: strict
        boundaries can have zero penalty while satisfaction is false. This is
        model consistency, never observed supervision or independent validation.
        Binary class/probability constraints require a separate typed interface.
        """
        contract,coordinates=self._objective_coordinates(predictions,views,contracts)
        contract._check('regression',native_unit)
        if decimal_policy!='nearest-model-dtype':
            raise ValueError('explicit nearest-model-dtype Decimal conversion policy required')
        requests=tuple(requests);mean=coordinates['mean'];scale=coordinates['scale']
        if len(requests)!=mean.numel():raise ValueError('one typed desired request per candidate required')
        if (not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale<=0).any()):
            raise ValueError('finite candidate mean and positive finite native scale required')

        def convert(literal):
            decimal=Decimal(literal)
            value=mean.new_tensor(float(decimal))
            if not torch.isfinite(value) or (decimal!=0 and value==0):
                raise ValueError('desired bound overflow/underflow at declared model precision')
            return value

        penalties=[];satisfied=[];native=[]
        for p,r in zip(mean,requests):
            if (not isinstance(r,DesiredProperty) or r.endpoint_key!=contract.definition.key
                    or r.assay_context.native_bytes!=contract.assay_context.native_bytes):
                raise ValueError('typed desired endpoint/native-unit/assay/class partition mismatch')
            cell=r.native_constraint;native.append(cell.native_bytes)
            if cell.relation=='interval':
                lo=convert(cell.lower_decimal_literal);hi=convert(cell.upper_decimal_literal)
                if lo>=hi:raise ValueError('desired interval collapsed at declared precision')
                penalties.append(torch.relu(lo-p).square()+torch.relu(p-hi).square())
                satisfied.append((p>=lo if cell.lower_inclusive else p>lo)
                                 & (p<=hi if cell.upper_inclusive else p<hi))
            else:
                bound=convert(cell.decimal_literal);delta=p-bound
                if cell.relation=='=':
                    penalties.append(delta.square());satisfied.append(p==bound)
                elif cell.relation in ('<','<='):
                    penalties.append(torch.relu(delta).square())
                    satisfied.append(p<bound if cell.relation=='<' else p<=bound)
                else:
                    penalties.append(torch.relu(-delta).square())
                    satisfied.append(p>bound if cell.relation=='>' else p>=bound)
        return dict(penalty=torch.stack(penalties).mean(),satisfied=torch.stack(satisfied),
                    native_constraints=tuple(native),decimal_policy=decimal_policy,dtype=str(mean.dtype),
                    evidence_role='model_prediction_desired_consistency',observed_label_count=0)

    def assess_desired_class(self,predictions,views,contracts,requests,*,native_unit):
        """Score exact desired native classes without choosing a class threshold.

        Bernoulli negative log probability is a desired-consistency surrogate,
        not an observed-label loss, calibrated probability or biological evidence.
        Probability constraints require a separately defined interface.
        """
        contract,coordinates=self._objective_coordinates(predictions,views,contracts)
        contract._check('binary',native_unit)
        logits=coordinates['logit'];requests=tuple(requests)
        if len(requests)!=logits.numel() or not torch.isfinite(logits).all():
            raise ValueError('one typed desired class per finite candidate logit required')
        classes=[];native=[]
        for r in requests:
            if (not isinstance(r,DesiredProperty) or r.endpoint_key!=contract.definition.key
                    or r.assay_context.native_bytes!=contract.assay_context.native_bytes):
                raise ValueError('typed desired endpoint/native-unit/assay/class partition mismatch')
            cell=r.native_constraint
            if cell.relation!='=' or Decimal(cell.decimal_literal) not in (Decimal(0),Decimal(1)):
                raise ValueError('exact native zero/one desired class required; no thresholding')
            classes.append(int(Decimal(cell.decimal_literal)));native.append(cell.native_bytes)
        target=logits.new_tensor(classes)
        return dict(penalty=F.binary_cross_entropy_with_logits(logits,target),
                    desired_class_probability=torch.sigmoid((2*target-1)*logits),
                    native_classes=tuple(classes),native_constraints=tuple(native),
                    dtype=str(logits.dtype),probability_semantics='uncalibrated-model-Bernoulli',
                    evidence_role='model_prediction_desired_consistency',observed_label_count=0)

    def observed_loss(self,prediction,view,contract,*,source_admission=None):
        self._validate((view,),(contract,))
        require_observational_admet_admission(source_admission)

    def generate(self,inputs):
        raise ValueError('ADMET molecular decoder not implemented; desired latent conditioning only')
