"""Train-only original native scalar routes with observed partition weighting."""
from dataclasses import dataclass

from mora3.data.admet import canonical, digest
from mora3.data.source_scalar_native_rows import project_native_scalar_batch
from mora3.model.source_scalar_objectives import SharedNativeScalarModel, SourceScalarAxis
from .assay_campaign_routes import (
    _original_train_route, _partition_weighted_mean, _registered_owner, _require_original_task_batch,
    _scalar_owner_view, _view_scope,
)
from .primary_native_routes import NativeRouteWindow, PreparedNativeTargets

_FAMILIES={
    'ChEMBL37-AZ-experimental-logD7_4':('source_reported_experimental_logD7_4','original reported experimental logD7.4 coordinates'),
    'UniProt-kinetics':('curated_kinetic_annotation_value','original kinetic coordinates partitioned by literal measurement and unit'),
}


@dataclass(frozen=True)
class TrainOnlyScalarCampaignAdapter:
    original_view: object
    current_view: object
    axis: SourceScalarAxis
    source_snapshot_sha256: str
    family: str
    loss_unit = 'native point or censored coordinate Gaussian NLL in nats'

    def __post_init__(self):
        if type(self.axis) is not SourceScalarAxis or self.family not in _FAMILIES:
            raise ValueError('Exact independently verified original train-only scalar family required')
        self.axis.__post_init__()
        object.__setattr__(self,'_identity',canonical(self._document()))

    @property
    def task_key(self):
        return _FAMILIES[self.family][0]

    @property
    def target_unit(self):
        return _FAMILIES[self.family][1]

    def _document(self):
        original,current = _view_scope(self.original_view,self.current_view,self.source_snapshot_sha256)
        partitions = tuple(key for key,_ in self.axis.partitions if key[:2]==(self.family,self.task_key))
        if not partitions or digest(canonical(self.original_view.effective_filter))!=self.axis.filter_payload_sha256:
            raise ValueError('Exact original scalar source filter and native partitions required')
        return dict(schema='Mora3.TrainOnlyScalarCampaignAdapter/v1',family=self.family,task=self.task_key,
            original_view_policy_sha256=original,current_view_policy_sha256=current,
            source_snapshot_sha256=self.source_snapshot_sha256,scalar_proof_sha256=self.axis.proof_sha256,
            source_partitions=partitions,loss_unit=self.loss_unit,target_unit=self.target_unit)

    def executable_binding(self):
        doc=self._document()
        if canonical(doc)!=self._identity:
            raise ValueError('Exact original scalar executable binding changed')
        return doc

    def train_binding(self,*,batch_size):
        return _original_train_route(self,batch_size=batch_size)

    def prepare(self,window):
        if type(window) is not NativeRouteWindow:
            raise ValueError('Exact original native scalar train window required')
        _require_original_task_batch(self,window.batch,'train')
        scalar=project_native_scalar_batch(self.original_view,window.batch,split='train')
        grouped={}
        for i,row in enumerate(scalar.scalar_rows):
            grouped.setdefault(row.partition_key,[]).append(i)
        layout=tuple((key,tuple(indices),sum(scalar.scalar_rows[i].target.numeric_coordinate_available for i in indices))
                     for key,indices in grouped.items())
        count=sum(item[2] for item in layout)
        if not count:
            return PreparedNativeTargets(0,'original native numeric coordinate unavailable in this window',None)

        def evaluate(model):
            branch = _registered_owner(model, self, 'scalar', SharedNativeScalarModel)
            model.require_training_admission()
            view = _scalar_owner_view(self, branch.axis)
            outputs = branch.source_batch_objectives(view, scalar, split='train')
            return _partition_weighted_mean(layout,tuple(
                (r.partition_key,r.original_batch_indices,r.loss)for r in outputs))

        return PreparedNativeTargets(count,None,evaluate)
