"""Exact original Tox21 routes for the guarded primary training campaign."""

from dataclasses import dataclass, field
from decimal import Decimal, localcontext
import re

import torch

from mora3.data.admet import canonical, digest
from mora3.data.finite_union_native_rows import SelectedNativeSourceBatch
from mora3.data.tox21 import FORWARD, INVERSE, SOURCE
from mora3.data.tox21_current import Tox21CurrentSource
from mora3.model.annotation_vocabulary import AnnotationVocabulary
from mora3.model.full_owner import SourceBoundOwner
from mora3.model.pubchem_decoder import PubChemByteCodec
from mora3.model.tox21_current_model import NativeTox21SourceModel
from .mup_campaign import ValidationBatchReduction
from .primary_native_routes import (
    FiniteNativeTrainRoute, NativeRouteBinding, NativeRouteWindow, PreparedNativeTargets, finite_view_policy_sha256,
)


_SCHEMA = 'mora3.CorpusTask/v1'
_SHA256 = re.compile(r'[a-f0-9]{64}\Z')


@dataclass(frozen=True)
class Tox21CampaignAdapter:
    """Keep original forward and inverse roles and the exact learned target codec."""

    source: Tox21CurrentSource
    current_view: object
    vocabulary: AnnotationVocabulary
    source_snapshot_sha256: str
    direction: str
    chunk_size: int = 1024
    _codec: PubChemByteCodec = field(init=False, repr=False, compare=False)
    _identity: bytes = field(init=False, repr=False)

    def __post_init__(self):
        if (
            type(self.source) is not Tox21CurrentSource or type(self.vocabulary) is not AnnotationVocabulary
            or self.direction not in (FORWARD, INVERSE)
            or type(self.source_snapshot_sha256) is not str or not _SHA256.fullmatch(self.source_snapshot_sha256)
            or type(self.chunk_size) is not int or self.chunk_size < 1
        ):
            raise ValueError('Exact Tox21 source, snapshot, direction, vocabulary and chunk size required')
        codec = PubChemByteCodec(self.vocabulary.tokenizer())
        object.__setattr__(self,'_codec',codec)
        object.__setattr__(self,'_identity',canonical(self._document()))

    @property
    def target_unit(self):
        return ('observed source assay-call positions' if self.direction==FORWARD
                else 'native shortest-path payload tokens and EOS')

    @property
    def loss_unit(self):
        return 'observed native call BCE in nats' if self.direction==FORWARD else 'full native byte-token CE in nats'

    def _document(self):
        original = self.source._view
        view_policy = finite_view_policy_sha256(self.current_view)
        if (finite_view_policy_sha256(original)!=self.source.axes.native_view_policy_sha256
                or original.root!=self.current_view.root or canonical(original.manifest)!=canonical(self.current_view.manifest)):
            raise ValueError('Same original source union and exact native Tox21 snapshot required')
        return dict(
            schema='Mora3.Tox21PrimaryCampaignAdapter/v1',source_snapshot_sha256=self.source_snapshot_sha256,
            current_native_view_policy_sha256=view_policy,original_source_axes=self.source.axes.binding(),
            source_family=SOURCE,native_task=self.direction,native_schema=_SCHEMA,
            source_paths=self.source.source_paths,loss_unit=self.loss_unit,target_unit=self.target_unit,
            payload_vocabulary_sha256=digest(self.vocabulary.specification_json.encode('utf-8')),
            byte_codec=self._codec.configuration(),chunk_size=self.chunk_size,
            inverse_train_recompute=True,inverse_validation_recompute=False,
        )

    def executable_binding(self):
        document = self._document()
        if canonical(document)!=self._identity:
            raise ValueError('Tox21 campaign source snapshot or codec binding changed')
        return document

    def train_binding(self, *, batch_size):
        policy = self.executable_binding()['current_native_view_policy_sha256']
        path_set = set(self.source.source_paths)
        paths, revisions = [], set()
        for descriptor in self.current_view.descriptors:
            path = descriptor['source_shard']['path']
            if path not in path_set:
                continue
            retained = {row['source_revision'] for row in descriptor['source_task_schema_counts']
                        if (row['source_family'],row['task_name'],row['native_schema'],row['split'],
                            row['eligible_view_disposition'])==(SOURCE,self.direction,_SCHEMA,'train',0)}
            if retained:
                paths.append(path)
                revisions.update(retained)
        route = FiniteNativeTrainRoute(
            self.current_view,task_key=self.direction,source_snapshot_sha256=self.source_snapshot_sha256,
            native_view_policy_sha256=policy,source_family=SOURCE,task_name=self.direction,native_schema=_SCHEMA,
            source_revisions=tuple(sorted(revisions)),source_paths=tuple(sorted(paths)),batch_size=batch_size,
        )
        return NativeRouteBinding(route,self.prepare,self.loss_unit,self.target_unit)

    def _observed_targets(self, batch, *, split):
        self.executable_binding()
        if type(batch) is not SelectedNativeSourceBatch or batch.original_split!=split:
            raise ValueError(f'Exact original {split} Tox21 native batch required')
        self.current_view.require_native_rows(batch,split=split)
        native = batch.native_rows
        columns = tuple(native[name].to_pylist() for name in ('source_family','task_name','native_schema'))
        if any(key!=(SOURCE,self.direction,_SCHEMA) for key in zip(*columns,strict=True)):
            raise ValueError('One exact original Tox21 task direction and schema required')
        views = self.source.views(batch,split=split,direction=self.direction)
        if len(views)!=native.num_rows:
            raise ValueError('Every exact original Tox21 row must remain present')
        if self.direction==FORWARD:
            return sum(sum(view.observed_mask) for view in views)
        return sum(len(self._codec.encode(view.targets()['canonical_isomeric_smiles'].encode('utf-8')))-1
                   for view in views)

    def validation_observed_targets(self, batch):
        """Enroll the original validation denominator without reading a model."""
        return self._observed_targets(batch,split='validation')

    def _loss(self, branch, batch, *, split):
        if (
            type(branch) is not NativeTox21SourceModel
            or canonical(branch.axes.binding())!=canonical(self.source.axes.binding())
            or canonical(branch.decoder.codec.configuration())!=canonical(self._codec.configuration())
            or branch.decoder.shared.payload.get_extra_state()['vocabulary']!=self.vocabulary.specification_json
        ):
            raise ValueError('Exact registered native Tox21 owner and unchanged vocabulary required')
        if self.direction==FORWARD:
            return branch.prediction_objective(self.source,batch,split=split)
        return branch.generation_objective(self.source,batch,split=split,chunk_size=self.chunk_size,
                                           recompute=split=='train')

    def _owner_branch(self, model):
        if (type(model) is not SourceBoundOwner
                or model.get_extra_state()['source_snapshot_sha256']!=self.source_snapshot_sha256):
            raise ValueError('Exact registered primary source owner and campaign snapshot required')
        return model.tox21_source

    def prepare(self, window):
        if type(window) is not NativeRouteWindow:
            raise ValueError('Exact original Tox21 train route window required')
        count = self._observed_targets(window.batch,split='train')

        def evaluate(model):
            branch = self._owner_branch(model)
            model.require_training_admission()
            return self._loss(branch,window.batch,split='train')

        return PreparedNativeTargets(count,None,evaluate)

    def _validation_reduction(self, branch, batch):
        if torch.is_grad_enabled():
            raise ValueError('Tox21 matched validation requires no-grad evaluation')
        count = self.validation_observed_targets(batch)
        loss = self._loss(branch,batch,split='validation')
        if not torch.isfinite(loss) or loss.ndim:
            raise ValueError('Tox21 native validation requires a finite scalar objective')
        with localcontext() as ctx:
            ctx.prec = 34
            total = Decimal(str(float(loss)))*count
        return ValidationBatchReduction(total,count,None)

    def evaluate_validation(self, model, batch):
        return self._validation_reduction(self._owner_branch(model),batch)
