"""Opt-in latent residual bridge; latent density is never molecule likelihood.

Independent adaptation: h0 = full_profile_context + W terminal_latent.
No therapeutic reward or calibration is supplied. Existing decoder is unchanged.
"""
from .configuration_snapshot import configuration_snapshot
import copy
import torch
from torch import nn
from torch.nn import functional as F
from rdkit import Chem
from .profile_decoder import ProfileSelfiesDecoder
from .profile_gfn import ProfileGFNPolicy


class ProfileLatentSelfiesDecoder(nn.Module):
    def __init__(self, decoder, policy, *, share_decoder=False):
        super().__init__()
        if (not isinstance(decoder, ProfileSelfiesDecoder)
                or not isinstance(policy, ProfileGFNPolicy)
                or decoder.profiles is not policy.profiles):
            raise ValueError('decoder and policy must share the identical native profile encoder')
        if type(share_decoder) is not bool:
            raise ValueError('share_decoder must be boolean')
        # A cooperative parent owns the decoder, its device and training modes.
        # A plain reference preserves deepcopy alias identity without registering
        # duplicate checkpoint paths or parameters. Standalone default unchanged.
        if share_decoder:
            object.__setattr__(self, 'decoder', decoder)
        else:
            self.decoder=decoder
        # Bind kernel/source configuration without registering a second encoder
        # or pretending reconstruction gradients train detached GFN samples.
        self._policy=copy.deepcopy(policy.get_extra_state())
        self.dimension=policy.dimension;self.horizon=policy.horizon
        self.latent_projection=nn.Linear(self.dimension,policy.context_width,bias=False)
        p=next(decoder.parameters());self.to(device=p.device,dtype=p.dtype)

    def get_extra_state(self):
        p=self.latent_projection.weight
        return configuration_snapshot(dict(version=1,architecture='profile-plus-linear-terminal-latent-GRU-v1',
            policy=self._policy,decoder=self.decoder.get_extra_state(),precision=str(p.dtype),
            semantics='latent-conditioned-source-reconstruction-not-molecule-density',
            generation='greedy-strict-nonempty-sanitized-molecule-v1'))

    def set_extra_state(self,state):
        if state!=self.get_extra_state():raise ValueError('latent decoder configuration mismatch')

    def _context(self,views,latents):
        views=self.decoder._views(views);p=self.latent_projection.weight
        if (not isinstance(latents,torch.Tensor) or latents.shape!=(len(views),self.dimension)
                or latents.device!=p.device or latents.dtype!=p.dtype or not torch.isfinite(latents).all()):
            raise ValueError('one finite matching terminal latent per condition required')
        context=self.decoder.profiles.condition(views)+self.latent_projection(latents)
        if not torch.isfinite(context).all():raise ValueError('finite residual context required')
        return context

    def forward(self,views,prefix,latents):
        views=self.decoder._views(views);p=self.latent_projection.weight
        if (not isinstance(prefix,torch.Tensor) or prefix.dtype!=torch.long or prefix.device!=p.device
                or prefix.ndim!=2 or prefix.shape[0]!=len(views)
                or not 1<=prefix.shape[1]<self.decoder.max_tokens
                or (prefix<0).any() or (prefix>=self.decoder.codec.size).any()):
            raise ValueError('bounded matching valid prefix required')
        context=self._context(views,latents)
        hidden,_=self.decoder.rnn(self.decoder.embedding(prefix),context.unsqueeze(0).contiguous())
        return self.decoder._readout(hidden)

    def objective(self,views,latents):
        views=self.decoder._views(views);prefix,target=self.decoder.teacher_batch(views)
        return F.cross_entropy(self(views,prefix,latents).flatten(0,1),target.flatten(),
                               ignore_index=self.decoder.codec.pad)

    def terminal_latents(self,paths):
        """Gather one detached terminal per original row, preserving ragged identity.

        Caller must retain exactly the same source-bound condition order as rollout;
        path tensors alone do not authenticate biological/source row identities.
        """
        x=paths['states'];lengths=paths['lengths'];mask=paths['mask'];p=self.latent_projection.weight
        if (x.ndim!=3 or x.shape[1:]!=(self.horizon,self.dimension) or not len(x)
                or x.requires_grad or x.device!=p.device or x.dtype!=p.dtype or not torch.isfinite(x).all()
                or type(paths['optional_stop']) is not bool):
            raise ValueError('finite detached matching complete paths required')
        if (lengths.shape!=(len(x),) or lengths.dtype!=torch.long or lengths.device!=x.device
                or (lengths<1).any() or (lengths>self.horizon).any()
                or (not paths['optional_stop'] and (lengths!=self.horizon).any())):
            raise ValueError('valid terminal lengths required')
        expected=torch.arange(self.horizon,device=x.device)[None,:]<lengths[:,None]
        if (mask.dtype!=torch.bool or mask.device!=x.device or mask.shape!=expected.shape
                or not torch.equal(mask,expected) or (x[~mask]!=0).any()):
            raise ValueError('valid prefix mask and zero padding required')
        return x[torch.arange(len(x),device=x.device),lengths-1].detach()

    def validate_sequences(self,sequences):
        """Retain raw tokens; never return empty/invalid/incomplete molecule as valid."""
        tokens=tuple(tuple(s) for s in sequences);smiles=[];status=[];done=[]
        for row in tokens:
            terminated=self.decoder.codec.eos in row;done.append(terminated)
            text=None;state='unterminated'
            if terminated:
                state='invalid_molecule'
                try:
                    decoded=self.decoder.codec.decode(row)
                    mol=Chem.MolFromSmiles(decoded) if decoded else None
                    if mol is not None and mol.GetNumAtoms()>0:text=decoded;state='valid'
                except (ValueError,IndexError):pass
            smiles.append(text);status.append(state)
        return dict(token_ids=tokens,terminated=tuple(done),smiles=tuple(smiles),
                    valid=tuple(s=='valid' for s in status),status=tuple(status))

    @torch.no_grad()
    def generate(self,views,latents):
        views=self.decoder._views(views);context=self._context(views,latents);d=self.decoder
        h=context.unsqueeze(0).contiguous()
        token=torch.full((len(views),1),d.codec.bos,device=context.device,dtype=torch.long)
        sequences=[[d.codec.bos] for _ in views];done=[False]*len(views)
        for _ in range(d.max_tokens-1):
            z,h=d.rnn(d.embedding(token),h);logits=d._readout(z[:,0])
            logits[:,d.codec.pad]=-torch.inf;logits[:,d.codec.bos]=-torch.inf
            token=logits.argmax(-1)[:,None]
            for i,t in enumerate(token[:,0].tolist()):
                if not done[i]:sequences[i].append(t);done[i]=t==d.codec.eos
            if all(done):break
        return self.validate_sequences(sequences)
