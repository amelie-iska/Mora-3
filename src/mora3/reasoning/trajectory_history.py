"""Full-prefix conditioning using an existing registered GeometryConditioner.

Every previous latent state is encoded as a native reasoning-state node with a
forward computational edge. These are generated latent coordinates, NOT observed
molecular or tissue positions. The shared graph transformer and its already
registered payload/incidence parameters are reused; this adapter adds no weights.
The prefix is detached for score-function training; gradients reach all existing
encoder/policy parameters on rescoring. Future states are never supplied.
"""
from hashlib import sha256
import json
import math
import re
import torch
from mora3.data.schema import GraphRecord,Node,Edge,Measurement
from mora3.data.tasks import ConditionalTask
from mora3.data.provenance import Provenance
from mora3.model.geometry_conditioning import GeometryConditioner
from mora3.model.profile_gfn import ProfileGFNPolicy


def _wire(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


class NativePrefixContext:
    def __init__(self,policy,geometry,*,source_context_sha256,coefficient,max_payload_bytes=2_000_000):
        if type(policy)is not ProfileGFNPolicy or type(geometry)is not GeometryConditioner:
            raise ValueError('Existing native shared policy and geometry conditioner required')
        if (type(source_context_sha256)is not str or re.fullmatch('[0-9a-f]{64}',source_context_sha256)is None
                or type(coefficient)not in(int,float)or not math.isfinite(coefficient)or coefficient<=0
                or type(max_payload_bytes)is not int or not 1<=max_payload_bytes<=8_000_000):
            raise ValueError('Explicit source, positive frozen coefficient and complete payload budget required')
        self.policy=policy;self.geometry=geometry;self.source_context_sha256=source_context_sha256
        self.coefficient=float(coefficient);self.max_payload_bytes=max_payload_bytes
        self._identity=(policy,geometry)
        self._frozen=_wire(self._binding())

    def _binding(self):
        p,g=self.policy,self.geometry
        if (p is not self._identity[0] or g is not self._identity[1]
                or g.config['width']!=p.context_width or g.config['id_width']<p.horizon+1
                or any(getattr(g.graph,n)is not getattr(p.profiles.shared.backbone,n)
                    for n in('feature','role','graph_token','layers','final_norm'))):
            raise ValueError('Exact shared transformer identity and complete prefix identifier rank required')
        return dict(schema='Mora3.NativePrefixContext/v1',source_context_sha256=self.source_context_sha256,
            geometry=g.get_extra_state(),policy=p.get_extra_state(),coefficient=self.coefficient,
            max_payload_bytes=self.max_payload_bytes,horizon=p.horizon,dimension=p.dimension,
            prefix_direction='computational-not-biological-causal',
            combines='biological-context-plus-coefficient-times-prefix-token-times-one-plus-biological-context',
            objective_kind='complete-trajectory-pathspace-not-endpoint-reward',
            complete_prefix=True,adds_learned_parameters=False,discrete_payload_coordinate_gradient=False,
            observed_physical_geometry_claim=False)

    def binding(self):
        result=self._binding()
        if _wire(result)!=self._frozen:raise ValueError('Frozen native full-prefix conditioner changed')
        return result

    def parameter_versions(self):
        self.binding()
        return tuple((id(p),p._version)for p in self.geometry.parameters())

    def condition(self,prefix,base_context):
        self.binding();p=self.policy;g=self.geometry
        p._context(base_context)
        if (not isinstance(prefix,torch.Tensor)or prefix.ndim!=3 or prefix.shape[0]!=len(base_context)
                or prefix.shape[2]!=p.dimension or prefix.requires_grad or not prefix.is_floating_point()
                or prefix.device!=base_context.device or prefix.dtype!=base_context.dtype
                or not torch.isfinite(prefix).all()or prefix.shape[1]>=p.horizon+1
                or prefix.shape[1]+1>g.config['id_width']):
            raise ValueError('Complete detached known past prefix within exact native rank required')
        records=[];tasks=[]
        for row in prefix:
            values=torch.cat((row.new_zeros(1,p.dimension),row),0)
            nodes=[];edges=[];roles=[]
            for i,state in enumerate(values):
                fields=[Measurement('reasoning_layer',i,'reasoning-step','observed',True),
                    Measurement('objective_kind','complete-trajectory-pathspace-not-endpoint-reward',None,'observed',True)]
                for j,value in enumerate(state):
                    fields.append(Measurement('latent_coordinate:'+str(j),float(value),'latent-coordinate','predicted',True))
                fields.append(Measurement('evidence_kind','model-generated-prefix',None,'observed',True))
                for m in fields:roles.append((('node',i,m.name),'condition'))
                if i==len(values)-1:
                    fields.append(Measurement('requested_next_state',None,None,'missing',False))
                    roles.append((('node',i,'requested_next_state'),'target'))
                nodes.append(Node('reasoning-prefix-state',tuple(fields)))
                if i:
                    edges.append(Edge(i-1,i,'computational-dependency',True,
                        (Measurement('direction_evidence','computational',None,'observed',True),)))
                    roles.append((('edge',i-1,'direction_evidence'),'condition'))
            record=GraphRecord('generated-prefix',tuple(nodes),tuple(edges),
                (Provenance('model-generated-reasoning-prefix','v1','no-biological-causal-claim',
                            self.source_context_sha256,'internal-derived-not-source-admission'),))
            task=ConditionalTask(tuple(roles),topology_known=True)
            from dataclasses import asdict
            if len(_wire(asdict(record)))+len(task.canonical_bytes())>self.max_payload_bytes:
                raise ValueError('Complete prefix exceeds explicit payload budget; no truncation')
            records.append(record);tasks.append(task)
        history=torch.cat([g(r,t)for r,t in zip(records,tasks)],0)
        if history.shape!=base_context.shape:raise ValueError('Native prefix context width changed')
        # Task-tagged modulation permits a distinct conditional normalizer for
        # whole-path versus endpoint objectives while preserving gradients to c.
        # It is an explicit adaptation requiring numerical muP qualification.
        result=base_context+self.coefficient*history*(1+base_context)
        if not torch.isfinite(result).all():raise ValueError('Nonfinite native prefix-conditioned context')
        return result
