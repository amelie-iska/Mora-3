"""Ninth native objective with independent source pools and one shared Adam owner.

Equal cyclic updates are experimental; assay priors do not imply cross-source
pairing, calibrated toxicity or supported phenotype-conditioned decoding.
"""
from .profile_decoder_cooperative import DecoderCooperativeModel, DecoderCooperativeTrainer
from .tox21 import SharedTox21Model
from .tox21_training import Tox21Batch
from .annotation_trainer import _digest


class Tox21CooperativeModel(DecoderCooperativeModel):
    def __init__(self, native, annotation, feature_axis, axes, *, codec,
                 max_tokens, flow_version=1, sigma_min=.01,
                 parameterization_policy='legacy',base_width=None,readout_mult=None,
                 attn_mult=None,encoder_var=None):
        super().__init__(native, annotation, feature_axis, axes, codec=codec,
            max_tokens=max_tokens, flow_version=flow_version, sigma_min=sigma_min,parameterization_policy=parameterization_policy,
                         base_width=base_width,readout_mult=readout_mult,
                         attn_mult=attn_mult,encoder_var=encoder_var)
        self.tox21 = SharedTox21Model(annotation,**(dict(parameterization_policy='mup-tox21-v1') if self.parameterization_policy!='legacy' else {}))
        self.configuration = dict(self.configuration,
            architecture='unpaired-native-nine-objective-Tox21-v1',
            tox21=self.tox21.get_extra_state())


class Tox21CooperativeTrainer(DecoderCooperativeTrainer):
    def __init__(self, model, batch, pools, *, tox21_batch, tox21_batch_size,
                 decoder_batch, decoder_batch_size, residue_batch, profile_batch,
                 profile_batch_size, batch_size, flow_batch_size, sigma_min=.01, lr=.001):
        if not isinstance(model, Tox21CooperativeModel) or not isinstance(tox21_batch, Tox21Batch):
            raise ValueError('native cooperative Tox21 model and batch required')
        if type(tox21_batch_size) is not int or tox21_batch_size <= 0:
            raise ValueError('positive integer Tox21 batch size required')
        model.tox21._validate(tox21_batch.views)
        super().__init__(model, batch, pools, decoder_batch=decoder_batch,
            decoder_batch_size=decoder_batch_size, residue_batch=residue_batch,
            profile_batch=profile_batch, profile_batch_size=profile_batch_size,
            batch_size=batch_size, flow_batch_size=flow_batch_size, sigma_min=sigma_min, lr=lr)
        self.tox21_batch = tox21_batch
        self.tox21_batch_size = tox21_batch_size
        self.kinds = self.kinds + ('tox21',)
        self.binding = _digest(dict(version=1, decoder_binding=self.binding,
            tox21_binding=tox21_batch.binding(), tox21_batch_size=tox21_batch_size,
            schedule=self.kinds, policy='equal-active-objective-cyclic-Tox21-v1'))

    def additional_objective(self, kind):
        if kind != 'tox21':
            return super().additional_objective(kind)
        pool = self.tox21_batch.views
        size = min(self.tox21_batch_size, len(pool))
        start = (self.steps // len(self.kinds)) * size
        views = tuple(pool[(start+i) % len(pool)] for i in range(size))
        return self.model.tox21.objective(self.model.tox21(views), views), len(views)
