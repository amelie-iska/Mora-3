"""Fixed admitted profile prediction batch; no inverse decoder objective.

Bindings cover actual immutable view bytes, not only caller-supplied digest labels.
Admission remains the native reader's responsibility; this is no new rights grant.
"""
from dataclasses import dataclass
import hashlib
import math
import torch
from mora3.data.molecule_profile import MoleculeProfileView
from .molecule_profile import SharedMoleculeProfileModel
from .annotation_trainer import AnnotationTrainer, _digest


@dataclass(frozen=True)
class MoleculeProfileBatch:
    views: tuple

    def __post_init__(self):
        object.__setattr__(self,'views',tuple(self.views))
        if not self.views or any(not isinstance(v,MoleculeProfileView) for v in self.views):
            raise ValueError('nonempty native views required')
        for v in self.views:
            if v.direction!='molecule_to_profile':
                raise ValueError('prediction objective only; inverse decoder unavailable')
            p=v.provenance();a=p.get('admission',{})
            if (p.get('split')!='train' or a.get('source_rights')!='CPG0003CC0'
                    or a.get('full_feature_axis') is not True
                    or a.get('no_same_cell_join') is not True
                    or not p.get('effective_groups')):
                raise ValueError('source train admission required')

    def binding(self):
        return _digest(dict(version=1,rows=[dict(direction=v.direction,
            modality=v.modality,axis=v.axis,reader_binding=v.binding_sha256,
            input=hashlib.sha256(v.input_bytes).hexdigest(),
            target=hashlib.sha256(v.target_bytes).hexdigest(),
            provenance=hashlib.sha256(v.provenance_bytes).hexdigest()) for v in self.views]))


class MoleculeProfileTrainer(AnnotationTrainer):
    """Full-axis native Adam using transactional full-state restore machinery."""
    def __init__(self,model,batch,*,lr=.001):
        if not isinstance(model,SharedMoleculeProfileModel) or not isinstance(batch,MoleculeProfileBatch):
            raise ValueError('native molecule/profile model and batch required')
        if not math.isfinite(lr) or lr<=0:
            raise ValueError('finite positive learning rate required')
        if getattr(model,'parameterization_policy','legacy')!='legacy':
            raise ValueError('new profile owner requires complete MuAdamW family; legacy Adam trainer unavailable')
        model._validate(batch.views)
        self.model=model;self.batch=batch;self.steps=0
        self.optimizer=torch.optim.Adam(model.parameters(),lr=lr)
        self.binding=_digest(dict(kind='native-molecule-profile-trainer-v1',batch=batch.binding(),
            configuration=model.get_extra_state(),optimizer='Adam',lr=lr,
            betas=[.9,.999],eps=1e-8,weight_decay=0,
            device=str(next(model.parameters()).device),dtype=str(next(model.parameters()).dtype)))

    def step(self):
        self.model.train();self.optimizer.zero_grad(set_to_none=True)
        loss=self.model.objective(self.model(self.batch.views),self.batch.views)
        if not torch.isfinite(loss):raise ValueError('nonfinite profile loss')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.model.parameters()):
            raise ValueError('nonfinite profile gradient')
        self.optimizer.step();self.steps+=1
        return {'total':float(loss.detach())}


from .residue_cooperative import ResidueCooperativeModel, ResidueCooperativeTrainer


class MoleculeProfileCooperativeModel(ResidueCooperativeModel):
    """Separate source pools sharing the native backbone and annotation payload.

    No same-cell or cross-assay pairing is inferred. Inverse conditioning remains
    available through profile.condition(); no molecular decoder is introduced.
    """
    def __init__(self, native, annotation, feature_axis, axes, *, flow_version=1,
                 sigma_min=.01,parameterization_policy='legacy',base_width=None,
                 readout_mult=None,attn_mult=None,encoder_var=None):
        super().__init__(native, annotation, feature_axis,
                         flow_version=flow_version, sigma_min=sigma_min,parameterization_policy=parameterization_policy,
                         base_width=base_width,readout_mult=readout_mult,
                         attn_mult=attn_mult,encoder_var=encoder_var)
        self.profile = SharedMoleculeProfileModel(annotation,axes,**(
            dict(parameterization_policy='mup-molecule-profile-v1') if self.parameterization_policy!='legacy' else {}))
        self.configuration = dict(self.configuration,
            architecture='unpaired-native-annotation-image-response-flow-residue-profile-v1',
            profile=self.profile.get_extra_state())


class MoleculeProfileCooperativeTrainer(ResidueCooperativeTrainer):
    """Seventh cyclic objective, full-axis MSE and independent profile cursor.

    Equal objective updates are an experimental mixture, not quality selection.
    Ordered complete profile bytes and batch size bind transactional restoration.
    Old trainers/checkpoints retain their original contracts without conversion.
    """
    def __init__(self, model, batch, pools, *, residue_batch, profile_batch,
                 profile_batch_size, batch_size, flow_batch_size,
                 sigma_min=.01, lr=.001):
        if not isinstance(model, MoleculeProfileCooperativeModel) or not isinstance(
                profile_batch, MoleculeProfileBatch):
            raise ValueError('native cooperative molecule/profile model and batch required')
        if type(profile_batch_size) is not int or profile_batch_size <= 0:
            raise ValueError('positive integer profile batch size required')
        model.profile._validate(profile_batch.views)
        super().__init__(model, batch, pools, residue_batch=residue_batch,
            batch_size=batch_size, flow_batch_size=flow_batch_size,
            sigma_min=sigma_min, lr=lr)
        self.profile_batch = profile_batch
        self.profile_batch_size = profile_batch_size
        self.kinds = self.kinds + ('profile',)
        self.binding = _digest(dict(version=1, residue_binding=self.binding,
            profile_binding=profile_batch.binding(), profile_batch_size=profile_batch_size,
            schedule=self.kinds, policy='equal-active-objective-cyclic-profile-v1'))

    def additional_objective(self, kind):
        if kind != 'profile':
            return super().additional_objective(kind)
        pool = self.profile_batch.views
        size = min(self.profile_batch_size, len(pool))
        start = (self.steps // len(self.kinds)) * size
        views = tuple(pool[(start+i) % len(pool)] for i in range(size))
        profile = self.model.profile
        return profile.objective(profile(views), views), len(views)
