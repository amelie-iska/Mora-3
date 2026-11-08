"""Source-bound eighth cooperative objective; independent observational pools.

Equal cyclic update counts are experimental. No image joins, causal claims, or
chemical-heldout generation quality follow from this training interface.
"""
from dataclasses import dataclass, replace
from .molecule_profile_training import (MoleculeProfileBatch,
    MoleculeProfileCooperativeModel, MoleculeProfileCooperativeTrainer)
from .profile_decoder import ProfileSelfiesDecoder
from .annotation_trainer import _digest


@dataclass(frozen=True)
class DecoderBatch(MoleculeProfileBatch):
    def __post_init__(self):
        object.__setattr__(self, 'views', tuple(self.views))
        if not self.views or any(getattr(v, 'direction', None) != 'profile_to_molecule' for v in self.views):
            raise ValueError('nonempty native inverse views required')
        # Identical source admission gates in either direction. Stored and hashed
        # records retain their true inverse direction and complete original bytes.
        MoleculeProfileBatch(tuple(replace(v, direction='molecule_to_profile') for v in self.views))


class DecoderCooperativeModel(MoleculeProfileCooperativeModel):
    def __init__(self, native, annotation, feature_axis, axes, *, codec,
                 max_tokens, flow_version=1, sigma_min=.01,
                 parameterization_policy='legacy',base_width=None,readout_mult=None,
                 attn_mult=None,encoder_var=None):
        super().__init__(native, annotation, feature_axis, axes,
                         flow_version=flow_version, sigma_min=sigma_min,parameterization_policy=parameterization_policy,
                         base_width=base_width,readout_mult=readout_mult,
                         attn_mult=attn_mult,encoder_var=encoder_var)
        self.decoder = ProfileSelfiesDecoder(self.profile,codec,max_tokens=max_tokens,**(
            dict(parameterization_policy='mup-sequence-v1') if self.parameterization_policy!='legacy' else {}))
        self.configuration = dict(self.configuration,
            architecture='unpaired-native-eight-objective-profile-SELFIES-v1',
            decoder=self.decoder.get_extra_state())


class DecoderCooperativeTrainer(MoleculeProfileCooperativeTrainer):
    def __init__(self, model, batch, pools, *, decoder_batch, decoder_batch_size,
                 residue_batch, profile_batch, profile_batch_size, batch_size,
                 flow_batch_size, sigma_min=.01, lr=.001):
        if not isinstance(model, DecoderCooperativeModel) or not isinstance(decoder_batch, DecoderBatch):
            raise ValueError('native decoder cooperative model and batch required')
        if type(decoder_batch_size) is not int or decoder_batch_size <= 0:
            raise ValueError('positive integer decoder batch size required')
        model.decoder.teacher_batch(decoder_batch.views)
        super().__init__(model, batch, pools, residue_batch=residue_batch,
            profile_batch=profile_batch, profile_batch_size=profile_batch_size,
            batch_size=batch_size, flow_batch_size=flow_batch_size,
            sigma_min=sigma_min, lr=lr)
        self.decoder_batch = decoder_batch
        self.decoder_batch_size = decoder_batch_size
        self.kinds = self.kinds + ('molecule_decoder',)
        self.binding = _digest(dict(version=1, profile_binding=self.binding,
            decoder_binding=decoder_batch.binding(), decoder_batch_size=decoder_batch_size,
            schedule=self.kinds, policy='equal-active-objective-cyclic-decoder-v1'))

    def additional_objective(self, kind):
        if kind != 'molecule_decoder':
            return super().additional_objective(kind)
        pool = self.decoder_batch.views
        size = min(self.decoder_batch_size, len(pool))
        start = (self.steps // len(self.kinds)) * size
        views = tuple(pool[(start+i) % len(pool)] for i in range(size))
        return self.model.decoder.objective(views), len(views)
