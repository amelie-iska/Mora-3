"""Native Rfam positive-observed family objective and desired-family sequence prior.

Shared positional payload pooling is lossy; no homology-heldout or calibrated
family classification claim. Unknown families are never negative supervision.
Corpus source admission remains the reader's responsibility.
"""
import copy
import math
from .mixed_payload_projection import exact_wire
import json
import re
import torch
from torch import nn
from torch.nn import functional as F
from mora3.data.rfam_family import RfamFamilyView, project_family_task
from mora3.data.nucleotide_prior import ALPHABET
from mora3.data.schema import GraphRecord, Node, Measurement
from mora3.data.provenance import Provenance
from mora3.data.tasks import ConditionalTask
from .nucleotide_decoder import NucleotideDecoder


class RfamFamilyModel(NucleotideDecoder):
    def __init__(self, decoder, family_axis, *, manifest_sha256,parameterization_policy='legacy'):
        if not isinstance(decoder,NucleotideDecoder):raise ValueError('shared nucleotide decoder required')
        axis=tuple(family_axis)
        if (not axis or axis!=tuple(sorted(set(axis)))
                or any(not isinstance(x,str) or re.fullmatch(r'RF[0-9]{5}',x) is None for x in axis)
                or not isinstance(manifest_sha256,str) or re.fullmatch('[0-9a-f]{64}',manifest_sha256) is None):
            raise ValueError('complete ordered family axis and manifest hash required')
        if parameterization_policy not in ('legacy','mup-rfam-v1'):
            raise ValueError('known explicit Rfam parameterization required')
        spec=getattr(decoder,'_recurrent_parameterization',None)
        if (parameterization_policy=='legacy')!=(spec is None):
            raise ValueError('matching explicit nucleotide/Rfam parameterization required')
        if spec is not None:decoder._validate_recurrent_parameterization()
        # Reuse the exact existing decoder and shared encoder parameters once.
        nn.Module.__init__(self)
        self.shared=decoder.shared;self.codec=decoder.codec
        self.parameterization_policy=parameterization_policy
        self._recurrent_parameterization=copy.deepcopy(spec)
        self._recurrent_owned_spec=copy.deepcopy(spec)
        self._recurrent_shared_reference=(self.shared,)
        self._recurrent_codec=getattr(decoder,'_recurrent_codec',None)
        self._family_declaration=(axis,manifest_sha256,parameterization_policy)
        self._family_mup=None
        self.embedding=decoder.embedding;self.rnn=decoder.rnn;self.output=decoder.output
        self.family_axis=axis;self.manifest_sha256=manifest_sha256
        p=next(self.shared.parameters())
        self.family_output=nn.Linear(self.embedding.embedding_dim,len(axis)).to(device=p.device,dtype=p.dtype)
        if spec is not None:
            with torch.no_grad():
                ratio=math.sqrt(spec['width']/spec['base_width'])
                self.family_output.weight.mul_(ratio);self.family_output.bias.mul_(ratio)
            self._family_mup=dict(schema='rfam-family-readout-parameterization-v1',policy=parameterization_policy,
                base_width=spec['base_width'],output_mult=spec['output_mult'],
                readout='native-fresh-times-sqrt-width-over-base-input-divided-by-width-over-base',
                positive_only_prediction='specificity-limited-diagnostic-not-closed-world-biological-loss')
        self._family_owned_mup=copy.deepcopy(self._family_mup)

    def get_extra_state(self):
        if self._family_mup is not None:
            if (self._family_declaration!=(self.family_axis,self.manifest_sha256,self.parameterization_policy)
                    or not exact_wire(self._family_mup,self._family_owned_mup)
                    or type(self.family_output) is not nn.Linear or self.family_output._modules
                    or (self.family_output.in_features,self.family_output.out_features)!=(self.embedding.embedding_dim,len(self.family_axis))
                    or tuple(self.family_output.weight.shape)!=(len(self.family_axis),self.embedding.embedding_dim)
                    or tuple(self.family_output.bias.shape)!=(len(self.family_axis),)):
                raise ValueError('Rfam exact source axis/manifest/readout policy drift')
        result=dict(version=1,family_axis=self.family_axis,
            manifest_sha256=self.manifest_sha256,missing_is_negative=False,
            objective='mean-positive-observed-family-softplus-negative-logit',
            generation='native-symbol-GRU-family-clan-conditioned',
            base=NucleotideDecoder.get_extra_state(self))
        if self._family_mup is not None:result.update(version=2,mup=copy.deepcopy(self._family_mup))
        return copy.deepcopy(result)

    def set_extra_state(self,state):
        if state!=self.get_extra_state():raise ValueError('Rfam source/axis configuration mismatch')

    def _views(self,views,direction):
        views=tuple(views)
        if not views:raise ValueError('nonempty Rfam views required')
        for v in views:
            if not isinstance(v,RfamFamilyView):raise ValueError('Rfam view required')
            p=v.provenance()
            if (p.get('manifest_sha256')!=self.manifest_sha256
                    or tuple(p.get('family_axis',()))!=self.family_axis
                    or p.get('envelope',{}).get('task_name')!='Rfam_source_family_'+direction+'_prior'):
                raise ValueError('Rfam direction/registry binding mismatch')
            _,families,_=project_family_task(dict(schema='mora3.CorpusTask/v1',
                task='Rfam_source_family_'+direction+'_prior',conditions=v.inputs(),targets=v.targets()),direction)
            if not set(families)<=set(self.family_axis):raise ValueError('family outside bound axis')
        return views

    def _latent(self,conditions,kind):
        conditions=tuple(conditions)
        if not conditions:raise ValueError('nonempty conditions required')
        graphs=[]
        for c in conditions:
            if not isinstance(c,dict):raise ValueError('explicit native conditions required')
            if kind=='prediction':
                s=c.get('source_native_sequence')
                if set(c)!={'source_native_sequence'} or not isinstance(s,str) or not s or set(s)-set(ALPHABET):
                    raise ValueError('complete native sequence only')
                value=s
            else:
                # Reuse the reader's strict RF/CL syntax and direction allowlist.
                _,fs,_=project_family_task(dict(schema='mora3.CorpusTask/v1',
                    task='Rfam_source_family_generation_prior',conditions=c,
                    targets={'source_native_sequence':'A'}),'generation')
                if not set(fs)<=set(self.family_axis):raise ValueError('family outside bound axis')
                value=json.dumps(c,sort_keys=True)
            record=GraphRecord('input',(Node('nucleotide',(
                Measurement('condition',value,None,'observed',True),
                Measurement('requested_output',None,None,'missing',False))),),(),
                (Provenance('native-input','v1','input','0'*64,'source-admission-external'),))
            task=ConditionalTask(((('node',0,'condition'),'condition'),
                (('node',0,'requested_output'),'target')),topology_known=True)
            graphs.append(self.shared.payload.forward_task(record,task=task,
                identifiers=self.shared.sequence_identifiers()))
        return self.shared.backbone(graphs)[0][:,0]

    def condition(self,conditions):return self._latent(conditions,'generation')

    def predict(self,views):
        views=self._views(views,'prediction')
        latent=self._latent(tuple(v.inputs() for v in views),'prediction')
        if self._family_mup is not None:
            self._validate_recurrent_parameterization();spec=self._recurrent_parameterization
            latent=latent*(spec['output_mult']/(spec['width']/spec['base_width']))
        return self.family_output(latent)

    def prediction_objective(self,views):
        views=self._views(views,'prediction');z=self.predict(views)
        rows=[];cols=[];index={f:i for i,f in enumerate(self.family_axis)}
        for i,v in enumerate(views):
            for family in v.targets()['source_Rfam_family_ids']:
                rows.append(i);cols.append(index[family])
        # Select positives before arithmetic. No pseudo-negatives or unknown loss.
        return F.softplus(-z[rows,cols]).mean()

    def teacher_batch(self,views):
        views=self._views(views,'generation')
        ids=[self.codec.encode(v.targets()['source_native_sequence']) for v in views]
        p=next(self.parameters());n=max(len(x)-1 for x in ids)
        prefix=torch.full((len(ids),n),self.codec.pad,device=p.device,dtype=torch.long)
        target=prefix.clone()
        for i,x in enumerate(ids):
            prefix[i,:len(x)-1]=prefix.new_tensor(x[:-1]);target[i,:len(x)-1]=target.new_tensor(x[1:])
        return prefix,target

    def generation_objective(self,views):
        views=self._views(views,'generation');prefix,target=self.teacher_batch(views)
        return F.cross_entropy(self(tuple(v.inputs() for v in views),prefix).reshape(-1,self.codec.size),
            target.reshape(-1),ignore_index=self.codec.pad)

    def objective(self,views):
        """Explicit generation path; prediction has its separate named objective."""
        return self.generation_objective(views)
