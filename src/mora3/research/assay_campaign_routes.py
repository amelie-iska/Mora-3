"""Original NCI and FreeSolv campaign routes with native reduction denominators."""

from dataclasses import dataclass, field, replace
from decimal import Decimal, localcontext
import re

import torch

from mora3.data.admet import canonical, digest
from mora3.data.finite_union_native_rows import SelectedNativeSourceBatch
from mora3.data.nci_current_assays import CONCLUSION, NciCurrentAssaySource
from mora3.data.nci_aids import FAMILIES
from mora3.data.source_scalar_native_rows import SCOPES, project_native_scalar_batch
from mora3.model.annotation_vocabulary import AnnotationVocabulary
from mora3.model.full_owner import SourceBoundOwner
from mora3.model.nci_assay_model import NativeNciAssayModel
from mora3.model.pubchem_decoder import PubChemByteCodec
from mora3.model.source_scalar_objectives import SharedNativeScalarModel, SourceScalarAxis
from .primary_native_routes import (
    FiniteNativeTrainRoute, NativeRouteBinding, NativeRouteWindow, PreparedNativeTargets, finite_view_policy_sha256,
)


_SCHEMA = 'mora3.CorpusTask/v1'
_SHA256 = re.compile(r'[a-f0-9]{64}\Z')
_FREESOLV = 'FreeSolv'


def _view_scope(original,current,snapshot):
    current_policy = finite_view_policy_sha256(current)
    original_policy = finite_view_policy_sha256(original)
    if (type(snapshot) is not str or not _SHA256.fullmatch(snapshot)
            or original.root!=current.root or canonical(original.manifest)!=canonical(current.manifest)):
        raise ValueError('Exact campaign snapshot and same original native source union required')
    return original_policy,current_policy


def _scalar_owner_view(adapter, axis):
    if axis == adapter.axis:
        return adapter.original_view
    current_axis = replace(
        adapter.axis, filter_payload_sha256=digest(canonical(adapter.current_view.effective_filter))
    )
    if axis != current_axis:
        raise ValueError('Exact original or current scalar source axis required')
    return adapter.current_view


def _original_train_route(adapter, *, batch_size, allowed_paths=None):
    policy = adapter.executable_binding()['current_view_policy_sha256']
    paths,revisions = [],set()
    for descriptor in adapter.current_view.descriptors:
        path = descriptor['source_shard']['path']
        if allowed_paths is not None and path not in allowed_paths:
            continue
        matches = {row['source_revision'] for row in descriptor['source_task_schema_counts'] if (
            row['source_family'],row['task_name'],row['native_schema'],row['split'],row['eligible_view_disposition']
        )==(adapter.family,adapter.task_key,_SCHEMA,'train',0)}
        if matches:
            paths.append(path)
            revisions.update(matches)
    route = FiniteNativeTrainRoute(adapter.current_view,task_key=adapter.task_key,
        source_snapshot_sha256=adapter.source_snapshot_sha256,native_view_policy_sha256=policy,
        source_family=adapter.family,task_name=adapter.task_key,native_schema=_SCHEMA,
        source_revisions=tuple(sorted(revisions)),source_paths=tuple(sorted(paths)),batch_size=batch_size)
    return NativeRouteBinding(route,adapter.prepare,adapter.loss_unit,adapter.target_unit)


def _require_original_task_batch(adapter,batch,split):
    adapter.executable_binding()
    if type(batch) is not SelectedNativeSourceBatch or batch.original_split!=split or split not in ('train','validation'):
        raise ValueError('Exact original train or validation role required for an assay route')
    adapter.current_view.require_native_rows(batch,split=split)
    rows = batch.native_rows
    columns = tuple(rows[name].to_pylist() for name in ('source_family','task_name','native_schema'))
    if any(key!=(adapter.family,adapter.task_key,_SCHEMA) for key in zip(*columns,strict=True)):
        raise ValueError('One complete exact original assay direction and schema required')


def _nci_partition_layout(tasks):
    grouped = {}
    for i,task in enumerate(tasks):
        if task.direction!='prediction':
            raise ValueError('Exact NCI prediction task direction required')
        grouped.setdefault(task.partition,[]).append(i)
    return tuple((partition,tuple(indices),len(indices)*(1 if partition.family==CONCLUSION else len(partition.coordinate_axis)))
                 for partition,indices in grouped.items())


def _nci_target_count(tasks,direction,codec):
    if not tasks or direction not in ('prediction','generation') or any(task.direction!=direction for task in tasks):
        raise ValueError('Exact nonempty native NCI objective direction required')
    if direction=='generation':
        return sum(len(codec.encode(task.target))-1 for task in tasks)
    return sum(count for _,_,count in _nci_partition_layout(tasks))


def _partition_weighted_mean(layout,results):
    if not layout or tuple((key,indices) for key,indices,_ in layout)!=tuple(
            (key,indices) for key,indices,_ in results):
        raise ValueError('Every native assay partition and original coordinate must contribute exactly once')
    numerator = sum(loss.double()*count for (_,_,count),(_,_,loss) in zip(layout,results,strict=True))
    return numerator/sum(count for _,_,count in layout)


def _validation_reduction(loss,observed_targets):
    from .mup_campaign import ValidationBatchReduction

    if loss.ndim or not torch.isfinite(loss):
        raise ValueError('Finite scalar native assay validation objective required')
    with localcontext() as ctx:
        ctx.prec = 34
        numerator = Decimal(str(float(loss)))*observed_targets
    return ValidationBatchReduction(numerator,observed_targets,None)


def _registered_owner(model,adapter,attribute,branch_type):
    if (type(model) is not SourceBoundOwner
            or model.get_extra_state()['source_snapshot_sha256']!=adapter.source_snapshot_sha256
            or type(getattr(model,attribute)) is not branch_type):
        raise ValueError('Exact registered native assay owner and source snapshot required')
    return getattr(model,attribute)


@dataclass(frozen=True)
class NciCampaignAdapter:
    source: NciCurrentAssaySource
    current_view: object
    vocabulary: AnnotationVocabulary
    source_snapshot_sha256: str
    family: str
    direction: str
    chunk_size: int = 1024
    _codec: PubChemByteCodec = field(init=False,repr=False,compare=False)
    _identity: bytes = field(init=False,repr=False)

    def __post_init__(self):
        if (type(self.source) is not NciCurrentAssaySource or type(self.vocabulary) is not AnnotationVocabulary
                or self.family not in FAMILIES or self.direction not in ('prediction','generation')
                or type(self.chunk_size) is not int or self.chunk_size<1):
            raise ValueError('Exact original NCI source, direction, vocabulary and chunk policy required')
        object.__setattr__(self,'_codec',PubChemByteCodec(self.vocabulary.tokenizer()))
        object.__setattr__(self,'_identity',canonical(self._document()))

    @property
    def task_key(self):
        return FAMILIES[self.family]+self.direction+'_prior'

    @property
    def target_unit(self):
        if self.direction=='generation':
            return 'native shortest-path connectivity tokens and EOS'
        return 'original source conclusions' if self.family==CONCLUSION else 'observed native log10 endpoint coordinates'

    @property
    def loss_unit(self):
        if self.direction=='generation':
            return 'full native connectivity-token CE in nats'
        return 'literal source conclusion CE in nats' if self.family==CONCLUSION else 'native log10 endpoint Gaussian NLL in nats'

    def _document(self):
        original,current = _view_scope(self.source._view,self.current_view,self.source_snapshot_sha256)
        if original!=self.source.axes.native_view_policy_sha256:
            raise ValueError('Original NCI source axis/census filter binding differs')
        return dict(schema='Mora3.NciPrimaryCampaignAdapter/v1',source_snapshot_sha256=self.source_snapshot_sha256,
            current_view_policy_sha256=current,native_axes=self.source.axes.binding(),source_paths=self.source.source_paths,
            family=self.family,task=self.task_key,native_schema=_SCHEMA,direction=self.direction,
            payload_vocabulary_sha256=digest(self.vocabulary.specification_json.encode('utf-8')),
            byte_codec=self._codec.configuration(),chunk_size=self.chunk_size,
            loss_unit=self.loss_unit,target_unit=self.target_unit,prediction_reduction='original native coordinates per partition')

    def executable_binding(self):
        result = self._document()
        if canonical(result)!=self._identity:
            raise ValueError('Exact NCI campaign executable binding changed')
        return result

    def train_binding(self, *, batch_size):
        return _original_train_route(self,batch_size=batch_size,allowed_paths=set(self.source.source_paths))

    def _targets(self,batch,split):
        _require_original_task_batch(self,batch,split)
        tasks = self.source.tasks(batch,split=split,direction=self.direction)
        if len(tasks)!=batch.native_rows.num_rows or any(task.partition.family!=self.family for task in tasks):
            raise ValueError('Every exact NCI original row and native partition required')
        return tasks,_nci_target_count(tasks,self.direction,self._codec)

    def validation_observed_targets(self,batch):
        return self._targets(batch,'validation')[1]

    def _loss(self,model,batch,split,tasks):
        branch = _registered_owner(model,self,'nci',NativeNciAssayModel)
        if (canonical(branch.axes.binding())!=canonical(self.source.axes.binding())
                or canonical(branch.decoder.codec.configuration())!=canonical(self._codec.configuration())
                or branch.decoder.shared.payload.get_extra_state()['vocabulary']!=self.vocabulary.specification_json):
            raise ValueError('Exact native NCI owner, axis and fitted target codec required')
        if self.direction=='generation':
            return branch.generation_objective(self.source,batch,split=split,chunk_size=self.chunk_size,recompute=split=='train')
        outputs = branch.prediction_objectives(self.source,batch,split=split)
        return _partition_weighted_mean(_nci_partition_layout(tasks),
            tuple((output.partition,output.indices,output.loss) for output in outputs))

    def prepare(self,window):
        if type(window) is not NativeRouteWindow:
            raise ValueError('Exact original NCI source train window required')
        tasks,count = self._targets(window.batch,'train')

        def evaluate(model):
            _registered_owner(model,self,'nci',NativeNciAssayModel)
            model.require_training_admission()
            return self._loss(model,window.batch,'train',tasks)

        return PreparedNativeTargets(count,None,evaluate)

    def evaluate_validation(self,model,batch):
        if torch.is_grad_enabled():
            raise ValueError('Original NCI validation is evaluation only')
        tasks,count = self._targets(batch,'validation')
        return _validation_reduction(self._loss(model,batch,'validation',tasks),count)


@dataclass(frozen=True)
class FreeSolvCampaignAdapter:
    original_view: object
    current_view: object
    axis: SourceScalarAxis
    source_snapshot_sha256: str
    _identity: bytes = field(init=False,repr=False)

    family = _FREESOLV
    task_key = SCOPES[_FREESOLV]
    loss_unit = 'experimental hydration free-energy point Gaussian NLL in nats'
    target_unit = 'reported experimental hydration free-energy points in kcal/mol'

    def __post_init__(self):
        if type(self.axis) is not SourceScalarAxis:
            raise ValueError('Exact independently verified native scalar axis required')
        self.axis.__post_init__()
        object.__setattr__(self,'_identity',canonical(self._document()))

    def _document(self):
        original,current = _view_scope(self.original_view,self.current_view,self.source_snapshot_sha256)
        if digest(canonical(self.original_view.effective_filter))!=self.axis.filter_payload_sha256:
            raise ValueError('Original FreeSolv source scalar projection filter differs')
        partitions = tuple(key for key,_ in self.axis.partitions if key[0]==self.family and key[1]==self.task_key)
        if not partitions:
            raise ValueError('Complete verified FreeSolv native scalar partition required')
        return dict(schema='Mora3.FreeSolvPrimaryCampaignAdapter/v1',source_snapshot_sha256=self.source_snapshot_sha256,
            original_view_policy_sha256=original,current_view_policy_sha256=current,
            scalar_proof_sha256=self.axis.proof_sha256,source_partitions=partitions,family=self.family,
            task=self.task_key,native_schema=_SCHEMA,loss_unit=self.loss_unit,target_unit=self.target_unit)

    def executable_binding(self):
        result = self._document()
        if canonical(result)!=self._identity:
            raise ValueError('Exact FreeSolv campaign executable binding changed')
        return result

    def train_binding(self, *, batch_size):
        return _original_train_route(self,batch_size=batch_size)

    def _targets(self,batch,split):
        _require_original_task_batch(self,batch,split)
        scalar = project_native_scalar_batch(self.original_view,batch,split=split)
        if not scalar.scalar_rows or any(row.target.kind!='reported_point' for row in scalar.scalar_rows):
            raise ValueError('Every selected FreeSolv row must contain an authentic reported experimental point')
        return scalar

    def validation_observed_targets(self,batch):
        return len(self._targets(batch,'validation').scalar_rows)

    def _loss(self,model,scalar,split):
        branch = _registered_owner(model,self,'scalar',SharedNativeScalarModel)
        view = _scalar_owner_view(self, branch.axis)
        grouped = {}
        for i,row in enumerate(scalar.scalar_rows):
            grouped.setdefault(row.partition_key,[]).append(i)
        outputs = branch.source_batch_objectives(view,scalar,split=split)
        return _partition_weighted_mean(tuple((key,tuple(indices),len(indices)) for key,indices in grouped.items()),
            tuple((output.partition_key,output.original_batch_indices,output.loss) for output in outputs))

    def prepare(self,window):
        if type(window) is not NativeRouteWindow:
            raise ValueError('Exact original FreeSolv source train window required')
        scalar = self._targets(window.batch,'train')

        def evaluate(model):
            _registered_owner(model,self,'scalar',SharedNativeScalarModel)
            model.require_training_admission()
            return self._loss(model,scalar,'train')

        return PreparedNativeTargets(len(scalar.scalar_rows),None,evaluate)

    def evaluate_validation(self,model,batch):
        if torch.is_grad_enabled():
            raise ValueError('Original FreeSolv validation is evaluation only')
        scalar = self._targets(batch,'validation')
        return _validation_reduction(self._loss(model,scalar,'validation'),len(scalar.scalar_rows))
