"""Reusable sequence-conditioned GO/EC model; source admission stays with reader.

Single known protein entity, positional pooled payload reference. No residue
sequence transformer, generation admission, negative imputation or label transfer.
"""
import copy
import hashlib
import math
import torch
from torch import nn
from mora3.model.annotation_vocabulary import AnnotationPayloadEncoder
from mora3.model.uniprot_go import GOAnnotationHead, annotation_graph_task, go_targets, observed_go_loss
from mora3.model.uniprot_ec import ECAnnotationHead, ec_targets
from mora3.model.tokengt import TokenGT, DenseAttention
from mora3.model.tokengt import _validate_native_policy,_native_mup_declaration
from .mixed_payload_projection import exact_wire


class SharedAnnotationModel(nn.Module):
    @classmethod
    def from_registry(cls, vocabulary, registry, **kwargs):
        """Use complete source axes; registry metadata never enters forward inputs."""
        from mora3.data.annotation_axes import AnnotationAxes
        if not isinstance(registry, AnnotationAxes):
            raise ValueError('verified annotation registry required')
        model=cls(vocabulary,registry.axis('GO'),registry.axis('EC'),**kwargs)
        model._configuration['annotation_registry']=registry.binding()
        return model

    def __init__(self, vocabulary, go_axis, ec_axis, *, width=16, heads=2, layers=2,
                 max_tokens=8192,backbone=None,parameterization_policy='legacy',base_width=None,
                 readout_mult=1.,attn_mult=1.,encoder_var=1.):
        super().__init__()
        if (parameterization_policy not in ('legacy','mup-annotation-v1')
                or type(readout_mult) not in (int,float) or not math.isfinite(readout_mult) or readout_mult<=0):
            raise ValueError('explicit supported annotation parameterization/readout required')
        new=parameterization_policy!='legacy'
        if not new and (base_width is not None or readout_mult!=1 or attn_mult!=1 or encoder_var!=1):
            raise ValueError('legacy annotation accepts no new width/readout initialization options')
        if (not new and isinstance(backbone,TokenGT) and len(backbone._native_configuration)!=10):
            raise ValueError('new native initializer requires an explicit annotation owner policy')
        if new:
            _validate_native_policy(width,heads,width,'vanilla',32,'mup-v1',attn_mult,'mup-transformer-v1',encoder_var)
            _native_mup_declaration(width,heads,width,'mup-transformer-v1','mup-v1',
                                   'mup-tokengt-v1',base_width,base_width)
            if type(layers) is not int or layers<=0:raise ValueError('positive native layer count required')
            if backbone is not None:
                cfg=backbone.configuration()
                if (type(backbone) is not TokenGT or cfg['schema']!='tokengt-native-configuration-v3'
                        or cfg['width']!=width or cfg['feature_width']!=width
                        or cfg['heads']!=heads or cfg['layers']!=layers
                        or cfg['parameterization']['base_width']!=base_width
                        or cfg['attn_mult']!=attn_mult or cfg['encoder_var']!=encoder_var):
                    raise ValueError('exact compatible shared native muP backbone required')
        self._configuration=dict(version=1,width=width,heads=heads,layers=layers,
                                 max_tokens=max_tokens,objective='sum-observed-applicable-GO-EC-v1')
        self.payload=AnnotationPayloadEncoder(vocabulary,width,max_tokens=max_tokens,
            parameterization_policy='mup-payload-v1' if new else 'legacy',base_width=base_width)
        if backbone is not None:
            if (not isinstance(backbone,TokenGT)
                    or backbone.feature.in_features!=width
                    or backbone.feature.out_features!=width
                    or len(backbone.layers)!=layers
                    or any(layer.attention.heads!=heads for layer in backbone.layers)
                    or any(not isinstance(layer.attention, DenseAttention) for layer in backbone.layers)):
                raise ValueError('compatible vanilla native backbone required')
            self.backbone=backbone
            self._configuration.update(shared_backbone=True,
                identifier_width=backbone.incidence.in_features//2,
                identifier_policy='exact-one-node-zero-padded')
        else:
            self.backbone=(TokenGT(width,1,width=width,heads=heads,layers=layers,
                scaling_policy='mup-v1',attn_mult=attn_mult,initialization_policy='mup-transformer-v1',
                encoder_var=encoder_var,parameterization_policy='mup-tokengt-v1',
                base_width=base_width,base_ffn_width=base_width) if new else
                TokenGT(width,1,width=width,heads=heads,layers=layers))
        self.go=GOAnnotationHead(width,go_axis)
        self.ec=ECAnnotationHead(width,ec_axis)
        self._mup_owned_configuration=None
        if new:
            with torch.no_grad():
                for head in (self.go.output,self.ec.output):
                    head.weight.mul_(math.sqrt(width/base_width));head.bias.mul_(math.sqrt(width/base_width))
            self._configuration.update(version=2,mup=dict(schema='annotation-parameterization-v1',
                policy=parameterization_policy,base_width=base_width,output_mult=float(readout_mult),
                vocabulary_sha256=hashlib.sha256(vocabulary.specification_json.encode()).hexdigest()),
                native_backbone=self.backbone.configuration(),
                payload_parameterization=self.payload._payload_mup_configuration())
            self._mup_owned_configuration=copy.deepcopy({k:v for k,v in self._configuration.items() if k!='objective'})
            self._mup_axes=(self.go.axis_ids,self.ec.axis_ids)
            self._mup_vocabulary=self.payload._vocabulary_json
            self._mup_backbone_reference=(self.backbone,)

    def validate_parameterization_configuration(self):
        if self._mup_owned_configuration is None:
            raise ValueError('explicit applied annotation muP owner required')
        if (any(k not in self._configuration or not exact_wire(v,self._configuration[k])
                for k,v in self._mup_owned_configuration.items())
                or self.backbone is not self._mup_backbone_reference[0]
                or not exact_wire(self.backbone.configuration(),self._configuration['native_backbone'])
                or not exact_wire(self.payload._payload_mup_configuration(),self._configuration['payload_parameterization'])
                or (self.go.axis_ids,self.ec.axis_ids)!=self._mup_axes
                or self.payload._vocabulary_json!=self._mup_vocabulary):
            raise ValueError('annotation muP owner/native/payload/axis configuration drift')
        if hasattr(self,'physical') and not exact_wire(self.physical.get_extra_state(),self._configuration.get('physical')):
            raise ValueError('physical annotation owner configuration drift')

    def _readout_features(self,latent):
        if self._mup_owned_configuration is None:return latent
        self.validate_parameterization_configuration();spec=self._configuration['mup']
        return latent*(spec['output_mult']/(self._configuration['width']/spec['base_width']))

    def get_extra_state(self):
        import copy
        if self._mup_owned_configuration is not None:self.validate_parameterization_configuration()
        return copy.deepcopy(self._configuration)

    def set_extra_state(self,state):
        if state!=self._configuration:
            raise ValueError('annotation model configuration checkpoint mismatch')

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,
                              missing_keys,unexpected_keys,error_msgs):
        incoming=state_dict.get(prefix+'_extra_state')
        if self._mup_owned_configuration is not None or (type(incoming) is dict and 'mup' in incoming):
            self.validate_parameterization_configuration()
            if not exact_wire(incoming,self.get_extra_state()):
                raise ValueError('annotation parameterization checkpoint configuration mismatch')
            self.payload._validate_parameterization_checkpoint(state_dict,prefix+'payload.')
            self.backbone._validate_initialization_checkpoint(state_dict,prefix+'backbone.')
            for name in ('payload','go','ec')+ (('physical',) if hasattr(self,'physical') else ()):
                if not exact_wire(state_dict.get(prefix+name+'._extra_state'),getattr(self,name).get_extra_state()):
                    raise ValueError('annotation child axis/payload checkpoint mismatch')
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,
                                     missing_keys,unexpected_keys,error_msgs)

    def sequence_identifiers(self):
        """One known entity, exact orthogonality at the configured native rank."""
        weight=self.backbone.feature.weight
        return torch.eye(1,self.backbone.incidence.in_features//2,
                         device=weight.device,dtype=weight.dtype)

    def forward(self,views):
        if not views:
            raise ValueError('nonempty prediction batch required')
        device=self.go.output.weight.device
        graphs=[]
        for view in views:
            record,task=annotation_graph_task(view)
            graphs.append(self.payload.forward_task(record,task=task,
                          identifiers=self.sequence_identifiers()))
        latent=self.backbone(graphs)[0][:,0]
        latent=self._readout_features(latent)
        return {'GO':self.go(latent),'EC':self.ec(latent)}

    def objective(self,logits,views,*,applicable):
        """Sum per-head observed BCE means; applicability is explicit per row.

        No loss reweighting by unknown label counts. Inactive heads have connected
        zero gradients. Unknown values never enter BCE arithmetic. Applicability
        is task metadata and is never passed to the sequence encoder.
        """
        if set(logits)!={'GO','EC'} or set(applicable)!={'GO','EC'}:
            raise ValueError('GO/EC logits and explicit applicability required')
        losses={}
        for name,project,axis in [('GO',go_targets,self.go.axis_ids),
                                   ('EC',ec_targets,self.ec.axis_ids)]:
            z=logits[name];active=applicable[name]
            if (not isinstance(active,torch.Tensor) or active.dtype!=torch.bool
                    or active.shape!=(len(views),) or active.device!=z.device):
                raise ValueError('Boolean per-row applicability on logits device required')
            y,observed=project(views,axis,device=z.device,dtype=z.dtype)
            losses[name]=observed_go_loss(z,y,observed & active[:,None])
        losses['total']=losses['GO']+losses['EC']
        return losses
