"""Stage-frozen complete source-likelihood reward for the Euclidean latent GFN.

R(z|x)=N(z;0,I32)*mean_y p_ref(full source SELFIES+EOS|x,z).
Every duplicate source observation retains weight1/n. The reference is freshly
copied only at explicit250-update boundaries. This module does not authenticate
source-group completeness or authorize training; the native source reader and
campaign must bind the complete immutable observations to its source manifest.
"""
import copy
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path
import re

import torch
from torch import nn
from mora3.data.molecule_profile import MoleculeProfileView
from mora3.data.schema import GraphRecord,Node,Measurement
from mora3.data.provenance import Provenance
from mora3.data.tasks import ConditionalTask
from .profile_latent_decoder import ProfileLatentSelfiesDecoder
from .profile_decoder import ProfileSelfiesDecoder
from .mixed_payload_projection import exact_wire

_SHA=re.compile('[0-9a-f]{64}')
_MODULES={'payload','backbone','profile_input','embedding','rnn','output','latent_projection'}


def _sha(raw):return hashlib.sha256(raw).hexdigest()


def _tensor_digest(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        if isinstance(value,torch.Tensor):
            if value.layout!=torch.strided or value.is_meta:raise ValueError('materialized dense frozen tensors required')
            h.update(json.dumps([name,str(value.dtype),list(value.shape)],separators=(',',':')).encode())
            h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class ProfileObservedMolecules:
    condition_bytes:bytes
    smiles:tuple
    source_group_manifest_sha256:str


class FrozenProfileLikelihoodReward(nn.Module):
    def __init__(self,model,*,stage_optimizer_update,source_group_manifest_sha256):
        super().__init__()
        if (type(model)is not ProfileLatentSelfiesDecoder or type(model.decoder)is not ProfileSelfiesDecoder
                or (model.dimension,model.horizon)!=(32,8)or model._policy.get('hidden')!=64
                or type(stage_optimizer_update)is not int or stage_optimizer_update<0 or stage_optimizer_update%250
                or type(source_group_manifest_sha256)is not str or not _SHA.fullmatch(source_group_manifest_sha256)):
            raise ValueError('explicit source primary latent32/horizon8/hidden64 and250-update stage required')
        decoder=model.decoder;profiles=decoder.profiles;decoder._validate_recurrent_parameterization()
        if (model._policy.get('version')!=2 or model._policy.get('dimension')!=32 or model._policy.get('horizon')!=8
                or model._policy.get('context_width')!=decoder.rnn.hidden_size
                or not exact_wire(model._policy.get('profiles'),profiles.get_extra_state())):
            raise ValueError('source policy/profile owner configuration drift')
        self.axes=copy.deepcopy(profiles.axes);self.codec=copy.deepcopy(decoder.codec)
        self.max_tokens=decoder.max_tokens;self.dimension=32;self.horizon=8
        self.stage_optimizer_update=stage_optimizer_update
        self.source_group_manifest_sha256=source_group_manifest_sha256
        # One memo preserves every copied alias, without copying any unrelated
        # annotation/profile readout/GFN policy heads or retaining live modules.
        original=(profiles.shared.payload,profiles.backbone,profiles.profile_input,
            decoder.embedding,decoder.rnn,decoder.output,model.latent_projection)
        copied=copy.deepcopy(original)
        for name,module in zip(('payload','backbone','profile_input','embedding','rnn','output','latent_projection'),copied):
            self.add_module(name,module)
        # RNNBase deepcopy can retain weak cache references to the live parent.
        # Rebuild only its registered-parameter pointer cache; no forward/RNG.
        self.rnn._init_flat_weights()
        original_storage={p.untyped_storage().data_ptr()for module in original for p in module.parameters()if p.numel()}
        if any(p.numel()and p.untyped_storage().data_ptr()in original_storage for p in self.parameters()):
            raise ValueError('independent frozen reference storage required')
        for p in self.parameters():
            if not torch.isfinite(p).all():raise ValueError('finite source reference parameters required')
            p.requires_grad_(False)
        super().train(False)
        self._readout=copy.deepcopy(decoder._recurrent_parameterization)
        code_modules={inspect.getmodule(t)for t in (type(self),type(model),type(decoder),type(profiles),
            type(self.payload),type(self.backbone),type(self.rnn))}
        codes={Path(m.__file__).name:_sha(Path(m.__file__).read_bytes())for m in code_modules}
        self._base_configuration=dict(schema='Mora3.FrozenProfileLikelihoodReward/v1',
            source_group_manifest_sha256=source_group_manifest_sha256,stage_optimizer_update=stage_optimizer_update,
            refresh_interval=250,dimension=32,horizon=8,fixed_hidden=64,
            source_owner=dict(profile_axes=self.axes,codec_symbols=self.codec.symbols,max_tokens=self.max_tokens,
                payload=self.payload.get_extra_state(),backbone=self.backbone.configuration(),
                readout=self._readout,profile_policy=copy.deepcopy(model._policy)),
            objective='normalizedN(0,I32)+logsumexp(full-source-observation-sequence-logprob)-log(n);optional-stop-log8',
            source_admission='external-complete-observation-reader-and-global-campaign;none-issued-here',
            reference='independently-owned-minimal-executable-frozen-within-stage;initial-random-stage0-explicit',
            code_sha256=codes,torch_version=str(torch.__version__))
        self._base_configuration=copy.deepcopy(self._base_configuration)
        self._configuration=dict(**copy.deepcopy(self._base_configuration),
            snapshot_tensor_sha256=_tensor_digest(self.state_dict()),
            snapshot_cpu_rng_sha256=_sha(torch.get_rng_state().numpy().tobytes()))
        self._refresh_versions()

    def _refresh_versions(self):
        self._versions={n:(id(p),p._version,str(p.device),p.dtype,p.untyped_storage().data_ptr(),p.requires_grad)
            for n,p in self.named_parameters()}
        self._buffer_versions={n:(id(b),b._version,str(b.device),b.dtype,b.untyped_storage().data_ptr())
            for n,b in self.named_buffers()}

    def _validate(self):
        current={n:(id(p),p._version,str(p.device),p.dtype,p.untyped_storage().data_ptr(),p.requires_grad)
            for n,p in self.named_parameters()}
        buffers={n:(id(b),b._version,str(b.device),b.dtype,b.untyped_storage().data_ptr())for n,b in self.named_buffers()}
        if (current!=self._versions or buffers!=self._buffer_versions or set(self._modules)!=_MODULES
                or any(m.training for m in self.modules())or any(p.requires_grad for p in self.parameters())):
            raise ValueError('frozen reference tensor/storage/device/mode drift')
        source=self._base_configuration['source_owner']
        if (self.stage_optimizer_update!=self._base_configuration['stage_optimizer_update']
                or self.source_group_manifest_sha256!=self._base_configuration['source_group_manifest_sha256']
                or not exact_wire(self._readout,source['readout'])
                or self.codec.index!={s:i for i,s in enumerate(source['codec_symbols'])}
                or self.codec.size!=len(source['codec_symbols'])or(self.codec.pad,self.codec.bos,self.codec.eos)!=(0,1,2)
                or not exact_wire(self.payload.get_extra_state(),source['payload'])
                or not exact_wire(self.backbone.configuration(),source['backbone'])
                or self.axes!=self._base_configuration['source_owner']['profile_axes']
                or self.codec.symbols!=self._base_configuration['source_owner']['codec_symbols']
                or self.max_tokens!=self._base_configuration['source_owner']['max_tokens']):
            raise ValueError('frozen source/codec policy drift')

    def train(self,mode=True):
        if mode is not False:raise ValueError('frozen reference never enters train mode')
        return super().train(False)

    def get_extra_state(self):
        # During construction, state_dict is used once to hash only tensor bytes.
        if not hasattr(self,'_configuration'):return None
        return copy.deepcopy(self._configuration)

    def set_extra_state(self,state):
        if not exact_wire(state,self._configuration):raise ValueError('frozen reference checkpoint configuration mismatch')

    def _context(self,views):
        graphs=[];weight=self.backbone.feature.weight
        identifiers=torch.eye(1,self.backbone.incidence.in_features//2,device=weight.device,dtype=weight.dtype)
        for v in views:
            x=v.inputs()
            if (type(v)is not MoleculeProfileView or v.direction!='profile_to_molecule' or v.modality not in self.axes
                    or v.axis!=self.axes[v.modality]or set(x)!={'assay_modality','cell_line','source_profile_array'}
                    or x['assay_modality']!=v.modality):
                raise ValueError('exact target-free full native inverse profile condition required')
            a=x['source_profile_array']
            if tuple(a['axis_ids'])!=v.axis:raise ValueError('complete source profile axis required')
            record=GraphRecord('input',(Node('molecule-assay',(
                Measurement('condition',json.dumps({k:x[k]for k in('assay_modality','cell_line')},sort_keys=True),None,'observed',True),
                Measurement('requested_output',None,None,'missing',False))),),(),
                (Provenance('native-input','v1','input','0'*64,'source-admission-external'),))
            task=ConditionalTask(((('node',0,'condition'),'condition'),(('node',0,'requested_output'),'target')),topology_known=True)
            graph=self.payload.forward_task(record,task=task,identifiers=identifiers)
            y=graph['features'].new_tensor(a['values'])
            if y.shape!=(len(v.axis),)or not torch.isfinite(y).all():raise ValueError('complete finite source profile values required')
            graph['features']=graph['features']+self.profile_input[v.modality](y)[None]
            graphs.append(graph)
        return self.backbone(graphs)[0][:,0]

    @torch.no_grad()
    def log_reward(self,views,latents,groups,*,optimizer_update,optional_stop=False):
        self._validate();views=tuple(views);groups=tuple(groups);p=self.latent_projection.weight
        if (type(optimizer_update)is not int or not self.stage_optimizer_update<=optimizer_update<self.stage_optimizer_update+250
                or type(optional_stop)is not bool or not views or len(groups)!=len(views)
                or not isinstance(latents,torch.Tensor)or latents.shape!=(len(views),32)
                or latents.dtype!=p.dtype or latents.device!=p.device or not torch.isfinite(latents).all()):
            raise ValueError('matching finite source latents and explicit current frozen stage required')
        encoded=[]
        for v,g in zip(views,groups):
            if (type(g)is not ProfileObservedMolecules or g.condition_bytes!=v.input_bytes
                    or g.source_group_manifest_sha256!=self.source_group_manifest_sha256
                    or type(g.smiles)is not tuple or not g.smiles or any(type(s)is not str or not s for s in g.smiles)):
                raise ValueError('complete same-condition observed source molecule group required')
            encoded.append(tuple(self.codec.encode(s,max_tokens=self.max_tokens)for s in g.smiles))
        context=self._context(views)+self.latent_projection(latents);scores=[]
        spec=self._readout;factor=spec['output_mult']/(spec['width']/spec['base_width'])
        for row,sequences in enumerate(encoded):
            values=[]
            for ids in sequences:
                tokens=torch.tensor(ids,device=p.device,dtype=torch.long)
                h,_=self.rnn(self.embedding(tokens[:-1][None]),context[row:row+1][None].contiguous())
                logits=self.output(h*factor)
                if not torch.isfinite(logits).all():raise ValueError('finite full source sequence logits required')
                logp=logits.double().log_softmax(-1)[0].gather(-1,tokens[1:,None]).sum()
                values.append(logp)
            scores.append(torch.logsumexp(torch.stack(values),0)-math.log(len(values)))
        prior=-.5*(latents.double().square().sum(-1)+32*math.log(2*math.pi))
        result=prior+torch.stack(scores)-(math.log(8)if optional_stop else 0.)
        if not torch.isfinite(result).all():raise ValueError('finite complete source-guided log reward required')
        return result

    def checkpoint(self):
        self._validate();state=copy.deepcopy(self.state_dict())
        if _tensor_digest(state)!=self._configuration['snapshot_tensor_sha256']:
            raise ValueError('frozen reference tensor digest mismatch')
        return dict(schema='Mora3.FrozenProfileReferenceCheckpoint/v1',configuration=self.get_extra_state(),state=state)

    @classmethod
    def from_checkpoint(cls,template,checkpoint,*,source_group_manifest_sha256,stage_optimizer_update):
        if type(checkpoint)is not dict or set(checkpoint)!={'schema','configuration','state'}or checkpoint['schema']!='Mora3.FrozenProfileReferenceCheckpoint/v1':
            raise ValueError('complete frozen reference checkpoint required')
        candidate=cls(template,source_group_manifest_sha256=source_group_manifest_sha256,stage_optimizer_update=stage_optimizer_update)
        cfg=checkpoint['configuration'];required=set(candidate._configuration)
        if (type(cfg)is not dict or set(cfg)!=required
                or any(not exact_wire(cfg[k],v)for k,v in candidate._base_configuration.items())
                or any(type(cfg[k])is not str or not _SHA.fullmatch(cfg[k])for k in('snapshot_tensor_sha256','snapshot_cpu_rng_sha256'))):
            raise ValueError('frozen checkpoint exact source/stage/code policy mismatch')
        state=checkpoint['state'];expected=candidate.state_dict()
        if type(state)is not dict and not isinstance(state,dict):raise ValueError('complete frozen state required')
        if set(state)!=set(expected):raise ValueError('complete frozen state key set required')
        for n,v in expected.items():
            actual=state[n]
            if isinstance(v,torch.Tensor):
                if (not isinstance(actual,torch.Tensor)or actual.shape!=v.shape or actual.dtype!=v.dtype
                        or actual.layout!=v.layout or actual.is_meta):raise ValueError('frozen state exact tensor schema required')
            elif not exact_wire(actual,cfg if n=='_extra_state'else v):raise ValueError('frozen state source sidecar mismatch')
        if _tensor_digest(state)!=cfg['snapshot_tensor_sha256']:raise ValueError('frozen state digest mismatch')
        candidate._configuration=copy.deepcopy(cfg);candidate.load_state_dict(state,strict=True)
        candidate._refresh_versions();candidate._validate();return candidate
