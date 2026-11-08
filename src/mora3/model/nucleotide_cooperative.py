"""Tenth source-bound objective; independent pools, shared parameters and Adam.

Cyclic weighting is experimental, not a claim of paired nucleotide/cell data.
Complete native sequences and admission bytes bind recovery without truncation.
"""
from dataclasses import dataclass
import hashlib
import torch
from mora3.data.nucleotide_prior import NucleotidePriorView, nucleotide_prior_view
from .annotation_trainer import _digest
from .nucleotide_decoder import NucleotideDecoder, _is_rnacentral_view, _validate_rnacentral_view
from .tox21_cooperative import Tox21CooperativeModel, Tox21CooperativeTrainer


@dataclass(frozen=True)
class NucleotideBatch:
    views: tuple

    def __post_init__(self):
        object.__setattr__(self, 'views', tuple(self.views))
        if not self.views or any(not isinstance(v, NucleotidePriorView) for v in self.views):
            raise ValueError('nonempty admitted nucleotide views required')
        seen_rnacentral=set()
        for v in self.views:
            p=v.provenance()
            if _is_rnacentral_view(v):
                _validate_rnacentral_view(v)
                if p['task_id'] in seen_rnacentral:
                    raise ValueError('duplicate RNAcentral pool allocation')
                seen_rnacentral.add(p['task_id'])
                continue
            try:
                row=dict(p['envelope'])
                from mora3.data.nucleotide_prior import canonical
                row['native_payload']=canonical(p['native_payload'])
                # Original serialization may differ; validate the original payload
                # digest through canonical semantic equality below, then rebind for
                # the reader's byte integrity check. No source is rewritten.
                row['native_payload_sha256']=hashlib.sha256(row['native_payload']).hexdigest()
                checked=nucleotide_prior_view(row,effective_groups=p['effective_groups'],
                    corpus_binding=p['corpus_binding'])
                normalized=dict(p, envelope=dict(p['envelope'], native_payload_sha256=row['native_payload_sha256']))
                if (checked.sequence!=v.sequence or checked.input_bytes()!=v.input_bytes()
                        or checked.provenance()!=normalized):
                    raise ValueError('native admission mismatch')
            except (KeyError, TypeError) as exc:
                raise ValueError('complete source admission required') from exc

    def binding(self):
        return _digest(dict(version=1, rows=[hashlib.sha256(v.canonical_bytes()).hexdigest()
            for v in self.views], masks='all-source-observed;ACGTU-base-resolved-separately'))


class NucleotideCooperativeModel(Tox21CooperativeModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nucleotide=NucleotideDecoder(self.annotation,**(
            dict(parameterization_policy='mup-sequence-v1') if self.parameterization_policy!='legacy' else {}))
        self.configuration=dict(self.configuration,
            architecture='unpaired-native-ten-objective-nucleotide-v1',
            nucleotide=self.nucleotide.get_extra_state())


class NucleotideCooperativeTrainer(Tox21CooperativeTrainer):
    def __init__(self, model, batch, pools, *, nucleotide_batch, nucleotide_batch_size, **kwargs):
        if not isinstance(model,NucleotideCooperativeModel) or not isinstance(nucleotide_batch,NucleotideBatch):
            raise ValueError('native nucleotide cooperative model and batch required')
        if type(nucleotide_batch_size) is not int or nucleotide_batch_size<=0:
            raise ValueError('positive integer nucleotide batch size required')
        super().__init__(model,batch,pools,**kwargs)
        self.nucleotide_batch=nucleotide_batch
        self.nucleotide_batch_size=nucleotide_batch_size
        self.kinds=self.kinds+('nucleotide',)
        self.binding=_digest(dict(version=1,tox21_binding=self.binding,
            nucleotide_binding=nucleotide_batch.binding(),nucleotide_batch_size=nucleotide_batch_size,
            precision=dict(cudnn_tf32=torch.backends.cudnn.allow_tf32,
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                deterministic=torch.are_deterministic_algorithms_enabled()),
            schedule=self.kinds,policy='equal-active-objective-cyclic-nucleotide-v1'))

    def additional_objective(self,kind):
        if kind!='nucleotide':return super().additional_objective(kind)
        pool=self.nucleotide_batch.views
        size=min(self.nucleotide_batch_size,len(pool))
        start=(self.steps//len(self.kinds))*size
        views=tuple(pool[(start+i)%len(pool)] for i in range(size))
        return self.model.nucleotide.objective(views),len(views)


class RfamCooperativeModel(NucleotideCooperativeModel):
    """Full-axis annotation and native Rfam heads over the same shared encoder.

    This is a model interface, not a scheduler or an admission grant. Rfam
    prediction and generation remain independent source-prior tasks, with
    positive-only family supervision. Existing nucleotide decoder parameters
    are reused by identity; only the family output layer adds parameters.
    Prior ten-objective checkpoints keep their original architecture contract.
    """
    def __init__(self, *args, annotation_registry, family_axis,
                 rfam_manifest_sha256, **kwargs):
        from mora3.data.annotation_axes import AnnotationAxes
        from .rfam_family import RfamFamilyModel
        super().__init__(*args, **kwargs)
        if (not isinstance(annotation_registry, AnnotationAxes)
                or self.annotation.go.axis_ids != annotation_registry.axis('GO')
                or self.annotation.ec.axis_ids != annotation_registry.axis('EC')
                or self.annotation.get_extra_state().get('annotation_registry')
                    != annotation_registry.binding()):
            raise ValueError('complete source-bound annotation registry required')
        self.rfam = RfamFamilyModel(self.nucleotide, family_axis,
                                   manifest_sha256=rfam_manifest_sha256,**(
            dict(parameterization_policy='mup-rfam-v1') if self.parameterization_policy!='legacy' else {}))
        self.configuration = dict(self.configuration,
            architecture='unpaired-native-full-annotation-rfam-interface-v1',
            rfam=self.rfam.get_extra_state(),
            rfam_decoder_ownership='reuse-native-nucleotide-parameters-once',
            rfam_tasks=('positive-observed-family-prediction',
                        'family-clan-conditioned-native-generation'))


class RfamCooperativeTrainer(NucleotideCooperativeTrainer):
    """Twelve independent source objectives with once-owned shared Adam state.

    Separate Rfam batch sizes determine separate cyclic source cursors. Source
    admission is supplied by the reader; a byte binding is not authentication.
    This new schedule does not convert historical ten-objective checkpoints.
    """
    def __init__(self, model, batch, pools, *, rfam_batch,
                 rfam_prediction_batch_size, rfam_generation_batch_size, **kwargs):
        from .rfam_family_training import RfamFamilyBatch
        if not isinstance(model, RfamCooperativeModel) or not isinstance(rfam_batch, RfamFamilyBatch):
            raise ValueError('native Rfam cooperative model and bound batch required')
        for size in (rfam_prediction_batch_size, rfam_generation_batch_size):
            if type(size) is not int or size <= 0:
                raise ValueError('positive integer Rfam batch sizes required')
        for direction in ('prediction', 'generation'):
            model.rfam._views(getattr(rfam_batch, direction), direction)
        super().__init__(model, batch, pools, **kwargs)
        self.rfam_batch = rfam_batch
        self.rfam_prediction_batch_size = rfam_prediction_batch_size
        self.rfam_generation_batch_size = rfam_generation_batch_size
        self.kinds += ('rfam_prediction', 'rfam_generation')
        self._rfam_policy = self._execution_policy()
        self.binding = _digest(dict(version=1, nucleotide_binding=self.binding,
            execution_policy=self._rfam_policy,
            rfam_binding=rfam_batch.binding(),
            prediction_batch_size=rfam_prediction_batch_size,
            generation_batch_size=rfam_generation_batch_size,
            schedule=self.kinds, policy='equal-active-objective-cyclic-Rfam-v1'))

    def additional_objective(self, kind):
        if kind not in ('rfam_prediction', 'rfam_generation'):
            return super().additional_objective(kind)
        direction = kind.removeprefix('rfam_')
        pool = getattr(self.rfam_batch, direction)
        size = min(getattr(self, 'rfam_'+direction+'_batch_size'), len(pool))
        start = (self.steps // len(self.kinds)) * size
        views = tuple(pool[(start+i) % len(pool)] for i in range(size))
        return getattr(self.model.rfam, direction+'_objective')(views), len(views)

    def _execution_policy(self):
        p = next(self.model.parameters())
        return dict(device=str(p.device), dtype=str(p.dtype),
            threads=torch.get_num_threads(), deterministic=torch.are_deterministic_algorithms_enabled(),
            cudnn_tf32=torch.backends.cudnn.allow_tf32,
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32)

    def _check_policy(self):
        if self._execution_policy() != self._rfam_policy:
            raise ValueError('Rfam cooperative runtime policy mismatch')

    def step(self):
        self._check_policy()
        return super().step()

    def load_state_dict(self, state):
        self._check_policy()
        from .rfam_family_training import validate_adam_state
        validate_adam_state(self.model, state)
        return super().load_state_dict(state)
