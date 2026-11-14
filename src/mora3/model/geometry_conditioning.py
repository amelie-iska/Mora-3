"""Opt-in full-geometry readout sharing native transforms, not incidence rank.

Local adaptation: new scalar payload and exact-rank incidence projections feed
shared TokenGT transforms. This bridge does not attach pixel/response heads,
join sources, admit training labels or differentiate discrete native decoding.
A caller must explicitly configure sufficient identifier width; overflow fails.
"""
import torch
import copy
from .mixed_payload_projection import exact_wire
from torch import nn
from .payload import PayloadEncoder
from .tokengt import TokenGT
from mora3.tokenization.shortest_path import ByteTokenizer


class GeometryConditioner(nn.Module):
    def __init__(self, backbone, *, id_width,parameterization_policy='legacy',base_width=None,
                 payload_max_tokens=8192):
        super().__init__()
        if type(id_width) is not int or id_width <= 0:
            raise ValueError('positive integer identifier width required')
        if not isinstance(backbone, TokenGT):
            raise ValueError('native TokenGT backbone required')
        if parameterization_policy not in ('legacy','mup-geometry-v1'):
            raise ValueError('known explicit geometry parameterization required')
        if parameterization_policy=='legacy' and base_width is not None:
            raise ValueError('legacy geometry has no base width')
        new=parameterization_policy!='legacy'
        native=backbone.configuration() if new else None
        if new and (native['schema']!='tokengt-native-configuration-v3'
                or native['parameterization']['base_width']!=base_width):
            raise ValueError('explicit applied native geometry backbone/base required')
        if not new and len(backbone._native_configuration)!=10:
            raise ValueError('new native initializer requires explicit geometry parameterization')
        if type(payload_max_tokens) is not int or not 1 <= payload_max_tokens <= 262144:
            raise ValueError('Exact positive bounded payload token capacity required')
        if payload_max_tokens != 8192 and not new:
            raise ValueError('Nondefault payload capacity requires explicit native muP configuration')
        self.parameterization_policy=parameterization_policy
        self._source_backbone_reference=(backbone,)
        feature_width=backbone.feature.in_features
        width=backbone.feature.out_features
        self.payload=PayloadEncoder(ByteTokenizer(controls=('pad','node','edge')),feature_width,
            max_tokens=payload_max_tokens,
            **(dict(parameterization_policy='mup-payload-v1',base_width=base_width) if new else {}))
        # Only incidence is independent. Preserve every selected reference layer,
        # including attention variant; never construct a replacement attention.
        if new:
            self.graph=TokenGT(feature_width,id_width,width=width,heads=native['heads'],layers=native['layers'],
                ffn_width=native['ffn_width'],scaling_policy='mup-v1',initialization_policy='mup-transformer-v1',
                attn_mult=native['attn_mult'],encoder_var=native['encoder_var'],parameterization_policy='mup-tokengt-v1',
                base_width=base_width,base_ffn_width=native['parameterization']['base_ffn_width'])
        else:self.graph=TokenGT(feature_width,id_width,width=width,heads=1,layers=1)
        for name in ('feature','role','graph_token','layers','final_norm'):
            setattr(self.graph,name,getattr(backbone,name))
        self.config=dict(version=1,architecture='native-full-geometry-shared-v1',
                         id_width=id_width,feature_width=feature_width,width=width,
                         shared_layers=len(backbone.layers),
                         identifiers='exact-identity-zero-padded',
                         pooling='scalar-positional-then-graph-token',
                         discrete_geometry_gradient=False)
        self._mup_configuration=None
        if new:
            self.config.update(version=2,parameterization_policy=parameterization_policy,
                native_graph=self.graph.configuration(),payload=self.payload._payload_mup_configuration())
            if payload_max_tokens != 8192:
                self.config.update(version=3,payload_max_tokens=payload_max_tokens,
                    payload_input_policy='complete-explicit-token-cap-v1')
            self._mup_configuration=copy.deepcopy(self.config)

    def get_extra_state(self):
        if self._mup_configuration is not None:
            source=self._source_backbone_reference[0]
            if (not exact_wire(self.config,self._mup_configuration)
                    or self.parameterization_policy!='mup-geometry-v1'
                    or not exact_wire(self.graph.configuration(),self.config['native_graph'])
                    or not exact_wire(self.payload._payload_mup_configuration(),self.config['payload'])
                    or any(getattr(self.graph,n) is not getattr(source,n)
                           for n in ('feature','role','graph_token','layers','final_norm'))):
                raise ValueError('geometry shared operator/payload/configuration drift')
            return copy.deepcopy(self.config)
        return self.config.copy()

    def set_extra_state(self,state):
        if state != self.config:
            raise ValueError('geometry conditioning configuration mismatch')

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        if self._mup_configuration is not None:
            if not exact_wire(state_dict.get(prefix+'_extra_state'),self.get_extra_state()):
                raise ValueError('geometry checkpoint configuration mismatch')
            self.payload._validate_parameterization_checkpoint(state_dict,prefix+'payload.')
            self.graph._validate_initialization_checkpoint(state_dict,prefix+'graph.')
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def forward(self,record,task):
        """Only condition/desired fields; target availability never read here.

        All known nodes/edges remain. Known topology is a required task contract;
        this method cannot be used to recover unknown target topology.
        """
        self.get_extra_state()
        count=len(record.nodes)
        rank=self.config['id_width']
        if count > rank:
            raise ValueError('geometry exceeds configured exact identifier rank')
        weight=self.graph.feature.weight
        ids=torch.eye(count,rank,device=weight.device,dtype=weight.dtype)
        graph=self.payload.forward_task(record,task=task,identifiers=ids)
        return self.graph([graph])[0][:,0]
