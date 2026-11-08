"""Residue incidence in the native cooperative model; no inferred source joins.

This model interface preserves the standalone residue equations. Its separately
versioned configuration is not compatible with prior unified trainer bindings.
Scheduling and admission of residue batches remain trainer responsibilities.
"""
from .flow_unified import FlowUnifiedModel, FlowUnifiedTrainer
from .residue_objective import SharedResidueModel
from .residue_trainer import ResidueTrainingBatch
from .annotation_trainer import _digest


class ResidueCooperativeModel(FlowUnifiedModel):
    def __init__(self, native, annotation, feature_axis, *, flow_version=1,
                 sigma_min=.01,parameterization_policy='legacy',base_width=None,
                 readout_mult=None,attn_mult=None,encoder_var=None):
        super().__init__(native, annotation, flow_version=flow_version,
                         sigma_min=sigma_min,parameterization_policy=parameterization_policy,
                         base_width=base_width,readout_mult=readout_mult,
                         attn_mult=attn_mult,encoder_var=encoder_var)
        self.residue = SharedResidueModel(annotation, feature_axis,
            parameterization_policy='mup-residue-v1' if parameterization_policy=='mup-cooperative-v1' else 'legacy')
        self.configuration = dict(self.configuration,
            architecture='unpaired-native-annotation-image-response-flow-residue-v1',
            residue=self.residue.get_extra_state())

    def residue_objective(self, views, *, applicable):
        views = tuple(views)
        return self.residue.objective(self.residue(views), views,
                                      applicable=applicable)['total']


class ResidueCooperativeTrainer(FlowUnifiedTrainer):
    """Fixed full residue pool after each native/flow cycle; no inferred joins.

    The independent residue pool retains all residues and its applicability mask.
    Source/target/axis/mask hashes and schedule bind full-state continuation.
    Equal objective updates are an experimental mixture, not quality promotion.
    """
    def __init__(self, model, batch, pools, *, residue_batch, batch_size,
                 flow_batch_size, sigma_min=.01, lr=.001):
        if not isinstance(model, ResidueCooperativeModel) or not isinstance(
                residue_batch, ResidueTrainingBatch):
            raise ValueError('native residue cooperative model and batch required')
        if residue_batch.feature_axis != model.residue.head.feature_axis:
            raise ValueError('residue feature axis mismatch')
        super().__init__(model, batch, pools, batch_size=batch_size,
                         flow_batch_size=flow_batch_size, sigma_min=sigma_min, lr=lr)
        self.residue_batch = residue_batch
        self.kinds = self.kinds + ('residue',)
        self.binding = _digest(dict(version=1, native_flow_binding=self.binding,
            residue_binding=residue_batch.binding(), schedule=self.kinds,
            policy='equal-active-objective-full-residue-pool-v1'))

    def additional_objective(self, kind):
        if kind != 'residue':
            return super().additional_objective(kind)
        batch = self.residue_batch
        loss = self.model.residue_objective(batch.views,
            applicable=batch.applicability(next(self.model.parameters()).device))
        return loss, len(batch.views)
