"""Bounded typed scalar features for TokenGT; see I3-payload-features/review.md.

Position-weighted pooling is an explicit lossy adaptation, not a paper algorithm.
Only caller-allowed observed fields enter features. Topology/kinds must already
be known conditioning information. No autoregressive decoder is implemented.
"""
from dataclasses import replace
import copy
import hashlib
import json
import math

import torch
from torch import nn
from mora3.data.schema import GraphRecord
from mora3.tokenization.graph_payload import GraphPayloadCodec


class PayloadEncoder(nn.Module):
    def __init__(self,tokenizer,width,*,max_tokens=8192,pooling='positional',
                 parameterization_policy='legacy',base_width=None):
        super().__init__()
        if type(width) is not int or width <= 0 or type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError('Positive integer width and max_tokens required')
        self.codec = GraphPayloadCodec(tokenizer)
        self.max_tokens = max_tokens
        if pooling not in ('positional', 'bag'):
            raise ValueError('Unknown pooling mode')
        from .mixed_payload_projection import MixedPayloadProjection,mixed_payload_configuration
        if parameterization_policy not in ('legacy','mup-payload-v1'):
            raise ValueError('explicit supported payload parameterization required')
        if parameterization_policy=='legacy' and base_width is not None:
            raise ValueError('legacy payload accepts no base width')
        new=parameterization_policy!='legacy'
        if new:
            mixed_payload_configuration(width,base_width)
            from mora3.tokenization.shortest_path import ByteTokenizer
            if type(tokenizer) is not ByteTokenizer:
                raise ValueError('exact ordered byte tokenizer required for new payload policy')
        self.pooling = pooling
        self.offset = tokenizer.size
        # Disjoint metadata byte namespace plus two field boundaries.
        self.embedding = nn.Embedding(self.offset + 258, width)
        self.projection=(MixedPayloadProjection(width,base_width=base_width) if new
                         else nn.Linear(width+1,width))
        self._payload_parameterization=None
        if new:
            controls=tuple(sorted(tokenizer._controls.items(),key=lambda row:row[1]))
            names=tuple('payload:'+b.hex() for b in tokenizer.vocabulary)+tuple(
                'control:'+name for name,_ in controls)+tuple(
                'metadata-byte:'+str(i) for i in range(256))+('field-start','field-value')
            digest=hashlib.sha256(json.dumps(names,separators=(',',':')).encode()).hexdigest()
            self._tokenizer_reference=(tokenizer.vocabulary,controls)
            self._payload_parameterization=dict(schema='payload-parameterization-v1',
                policy=parameterization_policy,token_axis_sha256=digest,token_rows=len(names),
                max_tokens=max_tokens,pooling=pooling,offset=self.offset,
                projection=self.projection.get_extra_state())

    def _payload_mup_configuration(self):
        from .mixed_payload_projection import MixedPayloadProjection,exact_wire
        spec=self._payload_parameterization
        if spec is None:raise ValueError('explicit applied payload parameterization required')
        live=self.codec.tokenizer
        if (live.vocabulary is not self._tokenizer_reference[0]
                or tuple(sorted(live._controls.items(),key=lambda row:row[1]))!=self._tokenizer_reference[1]
                or self.max_tokens!=spec['max_tokens'] or self.pooling!=spec['pooling']
                or self.offset!=spec['offset'] or live.size!=self.offset
                or type(self.projection) is not MixedPayloadProjection
                or not exact_wire(self.projection.get_extra_state(),spec['projection'])
                or tuple(self.embedding.weight.shape)!=(spec['token_rows'],spec['projection']['width'])):
            raise ValueError('payload tokenizer/axis/shape/pooling policy drift')
        return copy.deepcopy(spec)

    def _save_to_state_dict(self,destination,prefix,keep_vars):
        super()._save_to_state_dict(destination,prefix,keep_vars)
        if self._payload_parameterization is not None:
            destination[prefix+'_payload_parameterization']=self._payload_mup_configuration()

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,
                              missing_keys,unexpected_keys,error_msgs):
        new=self._validate_parameterization_checkpoint(state_dict,prefix)
        if new:state_dict.pop(prefix+'_payload_parameterization')
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,
                                     missing_keys,unexpected_keys,error_msgs)

    def _validate_parameterization_checkpoint(self,state_dict,prefix):
        """Pure policy preflight usable by a larger source-owned model root."""
        from .mixed_payload_projection import exact_wire
        new=self._payload_parameterization is not None;key=prefix+'_payload_parameterization'
        if (key in state_dict)!=new:
            raise ValueError('payload parameterization checkpoint policy mismatch')
        if new:
            if not exact_wire(state_dict[key],self._payload_mup_configuration()):
                raise ValueError('payload parameterization checkpoint configuration mismatch')
            self.projection.validate_checkpoint(state_dict,prefix+'projection.')
        elif prefix+'projection._extra_state' in state_dict:
            raise ValueError('legacy payload cannot load mixed projection state')
        return new

    def pool(self, ids):
        """Reference positional equation; no padding, truncation or graph expansion."""
        if self._payload_parameterization is not None:self._payload_mup_configuration()
        if not ids or len(ids) > self.max_tokens:
            raise ValueError('Entity stream exceeds bounds or is empty')
        if any(type(t) is not int or not 0 <= t < self.embedding.num_embeddings for t in ids):
            raise ValueError('Invalid stream token')
        weight = self.embedding.weight
        tokens = torch.tensor(ids, dtype=torch.long, device=weight.device)
        positions = torch.arange(1, len(ids)+1, dtype=weight.dtype, device=weight.device)
        frequencies = 10000 ** (torch.arange(weight.shape[1], dtype=weight.dtype,
                                            device=weight.device) / weight.shape[1])
        values = self.embedding(tokens)
        pooled = ((values * (1 + torch.sin(positions[:,None] / frequencies))).mean(0)
                  if self.pooling == 'positional' else values.mean(0))
        return self.projection(torch.cat((pooled, weight.new_tensor([math.log1p(len(ids))]))))

    def _header(self, value):
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
        return [self.offset + b for b in raw]

    def forward(self, record, *, input_fields, identifiers):
        """Return TokenGT graph tensors; identifiers supplied under caller RNG policy.

        input_fields contains (node|edge, local index, measurement name). Selecting
        predicted observations is permitted and their status is encoded explicitly.
        Missing selected observations fail closed. No implicit select-all mode.
        """
        if not isinstance(record, GraphRecord):
            raise TypeError('Expected validated GraphRecord')
        if any(not e.directed for e in record.edges):
            raise ValueError('Undirected incidence policy not implemented')
        selected = set(input_fields)
        known = {(role, i, m.name): m for role, entities in
                 (('node', record.nodes), ('edge', record.edges))
                 for i, entity in enumerate(entities) for m in entity.measurements}
        if not selected <= known.keys() or any(not known[k].available for k in selected):
            raise ValueError('Unknown or unavailable input field')
        # Exclude before tokenization, so excluded payloads cannot affect length,
        # positional weights, headers or computational work of feature encoding.
        groups = {}
        for role, entities in (('node', record.nodes), ('edge', record.edges)):
            groups[role+'s'] = tuple(replace(e, measurements=tuple(
                m for m in e.measurements if (role,i,m.name) in selected))
                for i,e in enumerate(entities))
        packed = self.codec.encode(replace(record, **groups))
        features = []
        for e in packed['entities']:
            stream = self._header({'kind':e['kind'], 'directed':e['directed']})
            for f in sorted(e['measurements'], key=lambda f: f['name']):
                stream += [self.offset+256]
                stream += self._header({k:f[k] for k in ('name','units','status','value_type')})
                stream += [self.offset+257] + f['token_ids']
            features.append(self.pool(stream))
        device = self.embedding.weight.device
        return dict(features=torch.stack(features),
                    endpoints=torch.tensor([e['endpoints'] for e in packed['entities']], device=device, dtype=torch.long),
                    roles=torch.tensor([int(e['role']=='edge') for e in packed['entities']], device=device, dtype=torch.long),
                    identifiers=identifiers)

    def forward_task(self, record, *, task, identifiers):
        """Encode only explicit conditions/desired constraints from a task.

        Target availability is deliberately not consulted on the input path.
        Missing targets are valid at inference; training must separately ask the
        task for available_targets. Known topology and entity kinds remain inputs.
        """
        from mora3.data.tasks import ConditionalTask
        if not isinstance(task, ConditionalTask):
            raise ValueError('validated ConditionalTask required')
        return self(record, input_fields=task.input_fields(record), identifiers=identifiers)
