"""Original-role ExpansionRx and guard-clear ReactZyme training routes."""
from dataclasses import dataclass, replace

import pyarrow as pa

from mora3.data.admet import canonical
from mora3.data.expansionrx_current import ExpansionRxCurrentSource
from mora3.data.reactzyme_current import ReactZymeCurrentSource
from mora3.model.expansionrx_native_model import ExpansionRxNativeModel
from mora3.model.full_owner import SourceBoundOwner
from mora3.model.pubchem_decoder import PubChemByteCodec
from mora3.model.reactzyme_product import NativeReactZymeProductModel

from .assay_campaign_routes import _original_train_route, _require_original_task_batch, _view_scope
from .primary_native_routes import NativeRouteWindow, PreparedNativeTargets

_EXPANSION='ExpansionRx-source-native-ADMET-readouts'
_CHOICES={
    'entity_to_measurement':(_EXPANSION,'source_native_ADMET_entity_to_measurement',ExpansionRxCurrentSource),
    'measurement_to_entity':(_EXPANSION,'source_native_ADMET_measurement_to_entity',ExpansionRxCurrentSource),
    'enzyme_product':('ReactZyme','enzyme_annotated_reaction_product',ReactZymeCurrentSource),
}


@dataclass(frozen=True)
class NativeAdmetReactionCampaignAdapter:
    source: object
    current_view: object
    source_snapshot_sha256: str
    direction: str
    codec: PubChemByteCodec | None = None
    chunk_size: int = 1024

    def __post_init__(self):
        if (self.direction not in _CHOICES or type(self.source) is not _CHOICES[self.direction][2]
                or type(self.chunk_size) is not int or self.chunk_size<1
                or (self.direction=='entity_to_measurement' and self.codec is not None)
                or (self.direction!='entity_to_measurement' and type(self.codec) is not PubChemByteCodec)):
            raise ValueError('Exact native direction, source and decoder policy required')
        old=self.source._view
        current=self.current_view
        paths=set(self.source.source_paths)
        if (any(old._sealed_by_path.get(path)!=current._sealed_by_path.get(path) for path in paths)
                or {r for r in old._ordinal_exclusions if r[0] in paths}
                   != {r for r in current._ordinal_exclusions if r[0] in paths}):
            raise ValueError('Current finite view changes an original source coordinate')
        object.__setattr__(self,'_identity',canonical(self._document()))

    @property
    def family(self):
        return _CHOICES[self.direction][0]

    @property
    def task_key(self):
        return _CHOICES[self.direction][1]

    @property
    def loss_unit(self):
        if self.direction=='entity_to_measurement':
            return 'reported point or censored Gaussian NLL in native assay partitions'
        return 'complete native byte-token and EOS negative log likelihood in nats'

    @property
    def target_unit(self):
        if self.direction=='entity_to_measurement':
            return 'reported point or censoring coordinate with original partition unit'
        return 'observed native byte tokens and EOS'

    def _document(self):
        original,current=_view_scope(self.source._view,self.current_view,self.source_snapshot_sha256)
        return dict(schema='Mora3.NativeAdmetReactionCampaignAdapter/v1',family=self.family,task=self.task_key,
            source_snapshot_sha256=self.source_snapshot_sha256,original_view_policy_sha256=original,
            current_view_policy_sha256=current,source_axes=self.source.axes.binding(),direction=self.direction,
            chunk_size=self.chunk_size,codec=None if self.codec is None else self.codec.configuration(),
            target_unit=self.target_unit,loss_unit=self.loss_unit)

    def executable_binding(self):
        value=self._document()
        if canonical(value)!=self._identity:
            raise ValueError('Exact native source, decoder or executable route changed')
        return value

    def train_binding(self,*,batch_size):
        return _original_train_route(self,batch_size=batch_size,allowed_paths=set(self.source.source_paths))

    def _guard_clear_batch(self,batch):
        path=batch.source_shard['path']
        kept=[]
        payloads=batch.native_rows['native_payload_sha256'].to_pylist()
        for i,ordinal in enumerate(batch.source_row_ordinals):
            restricted=self.source._unresolved.get((path,ordinal))
            if restricted is None:
                kept.append(i)
            elif restricted!=payloads[i]:
                raise ValueError('Unresolved original ReactZyme guard coordinate changed')
        if not kept:
            return None
        if len(kept)==len(batch.source_row_ordinals):
            return batch
        return replace(batch,source_row_ordinals=tuple(batch.source_row_ordinals[i] for i in kept),
            native_rows=batch.native_rows.take(pa.array(kept,type=pa.int64())))

    def prepare(self,window):
        if type(window) is not NativeRouteWindow:
            raise ValueError('Exact original native train window required')
        batch=window.batch
        _require_original_task_batch(self,batch,'train')
        if self.direction=='enzyme_product':
            batch=self._guard_clear_batch(batch)
            if batch is None:
                return PreparedNativeTargets(0,'all original source lineages in this window are audit-only',None)
            tasks=self.source.tasks(batch,split='train')
            raw=tuple(task.product_bytes for task in tasks)
        else:
            tasks=self.source.tasks(batch,split='train',direction=self.direction)
            raw=tuple(task.native_connectivity_bytes() for task in tasks) if self.codec is not None else ()
        if len(tasks)!=len(batch.source_row_ordinals):
            raise ValueError('Exact selected original direction and task count required')
        count=len(tasks) if self.codec is None else sum(len(self.codec.tokenizer.encode(x))+1 for x in raw)

        def evaluate(owner):
            if type(owner) is not SourceBoundOwner:
                raise ValueError('Exact registered source-bound owner required')
            owner.require_training_admission()
            branch=owner.reactzyme if self.direction=='enzyme_product' else owner.expansionrx
            kind=NativeReactZymeProductModel if self.direction=='enzyme_product' else ExpansionRxNativeModel
            if (type(branch) is not kind or canonical(branch.axes.binding())!=canonical(self.source.axes.binding())
                    or self.codec is not None and canonical(branch.decoder.codec.configuration())!=canonical(self.codec.configuration())):
                raise ValueError('Exact registered source-native objective and shared byte codec required')
            if self.direction=='enzyme_product':
                return branch.objective(self.source,batch,split='train',chunk_size=self.chunk_size,recompute=True)
            if self.direction=='measurement_to_entity':
                return branch.generation_objective(self.source,batch,split='train',chunk_size=self.chunk_size,recompute=True)
            results=branch.prediction_objectives(self.source,batch,split='train')
            if sorted(i for result in results for i in result.indices)!=list(range(count)):
                raise ValueError('Every observed original point must contribute to exactly one partition')
            return sum(result.loss*len(result.indices) for result in results)/count

        return PreparedNativeTargets(count,None,evaluate)
