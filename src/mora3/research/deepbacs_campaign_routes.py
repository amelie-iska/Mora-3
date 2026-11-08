"""Train-only native route binding for the original DeepBacs reverse raster task.

Forward grayscale segmentation is intentionally unavailable here: the registered
full owner exposes an RGB cooperative instance head, while the exact DeepBacs
source input is grayscale and its standalone MorphologyInstance is not owner-bound.
"""
from dataclasses import dataclass
import hashlib
import re

import torch

from mora3.data.admet import canonical
from mora3.data.deepbacs_native import DeepBacsNativeSource
from mora3.data.finite_union_native_rows import FiniteUnionNativeView, SelectedNativeSourceBatch
from mora3.model.full_owner import SourceBoundOwner
from mora3.model.native_grayscale_flow import NativeGrayscaleFlow
from mora3.research.primary_native_routes import (
    FiniteNativeTrainRoute, NativeRouteBinding, NativeRouteWindow,
    PreparedNativeTargets, finite_view_policy_sha256,
)

FAMILY = 'DeepBacs-Staph'
TASK = 'mask_to_image'
SCHEMA = 'mora3.RasterMorphologyTask/v1'
REVISION = 'Zenodo5550933.v1'


@dataclass(frozen=True)
class DeepBacsNativeRouteAdapter:
    view: FiniteUnionNativeView
    source: DeepBacsNativeSource
    source_snapshot_sha256: str
    batch_size: int = 1

    def __post_init__(self):
        if (type(self.view) is not FiniteUnionNativeView or type(self.source) is not DeepBacsNativeSource
                or type(self.batch_size) is not int or self.batch_size != 1
                or self.view.effective_filter is None
                or type(self.source_snapshot_sha256) is not str
                or not re.fullmatch(r'[0-9a-f]{64}',self.source_snapshot_sha256)):
            raise ValueError('Exact pinned DeepBacs source/view and one-record flow objective required')
        original=self.source._view
        paths={path for path,_ in self.source._rows}
        if (original.root!=self.view.root or canonical(original.manifest)!=canonical(self.view.manifest)
                or any(original._sealed_by_path.get(p)!=self.view._sealed_by_path.get(p)for p in paths)
                or {r for r in original._ordinal_exclusions if r[0] in paths}
                   != {r for r in self.view._ordinal_exclusions if r[0] in paths}):
            raise ValueError('Current finite source view changed an original DeepBacs coordinate')
        object.__setattr__(self,'_identity',canonical(self._document()))

    def _document(self):
        return dict(schema='Mora3.DeepBacsReverseCampaignAdapter/v2',
            snapshot=self.source_snapshot_sha256,view=finite_view_policy_sha256(self.view),
            envelope=self.source.envelope.binding(),family=FAMILY,task=TASK,native_schema=SCHEMA,
            revision=REVISION,role='train',batch_size=self.batch_size,
            time_policy='uniform interior binary24 grid; private trial/window RNG',
            noise_policy='standard Gaussian over the complete original detector shape; private trial/window RNG')

    def executable_binding(self):
        doc=self._document()
        if canonical(doc)!=self._identity:
            raise ValueError('Native DeepBacs route or RNG policy changed')
        return doc

    def train_binding(self):
        self.executable_binding()
        paths=[]
        for descriptor in self.view.descriptors:
            ref=descriptor['source_shard']
            if any((row['source_family'],row['task_name'],row['native_schema'],row['source_revision'],row['split'],row['eligible_view_disposition'])
                   ==(FAMILY,TASK,SCHEMA,REVISION,'train',0)
                   for row in descriptor['source_task_schema_counts']):
                paths.append(ref['path'])
        if not paths:
            raise ValueError('No admitted original DeepBacs reverse-direction train shards')
        route=FiniteNativeTrainRoute(self.view,task_key='DeepBacs-Staph:mask_to_image',
            source_snapshot_sha256=self.source_snapshot_sha256,native_view_policy_sha256=finite_view_policy_sha256(self.view),
            source_family=FAMILY,task_name=TASK,native_schema=SCHEMA,source_revisions=(REVISION,),
            source_paths=tuple(paths),batch_size=self.batch_size)
        return NativeRouteBinding(route,self.prepare,'normalized detector-pixel flow-matching MSE',
                                  'complete original detector pixels')

    def prepare(self, window):
        self.executable_binding()
        if type(window) is not NativeRouteWindow:
            raise ValueError('Exact original DeepBacs train route window required')
        batch=window.batch
        if type(batch) is not SelectedNativeSourceBatch or batch.original_split!='train':
            raise ValueError('Original DeepBacs train source rows required')
        self.view.require_native_rows(batch,split='train')
        columns=tuple(batch.native_rows[name].to_pylist() for name in ('source_family','task_name','native_schema','source_revision'))
        if any(key!=(FAMILY,TASK,SCHEMA,REVISION) for key in zip(*columns,strict=True)):
            raise ValueError('Exact original DeepBacs reverse task rows required')
        packets=self.source.packets(batch,split='train',direction=TASK)
        if len(packets)!=batch.native_rows.num_rows:
            raise ValueError('Every original selected reverse row must have one exact image/mask packet')
        count=sum(packet.image.height*packet.image.width for packet in packets)
        if count<1:
            return PreparedNativeTargets(0,'original reverse raster has no detector pixels',None)
        def evaluate(owner):
            if type(owner) is not SourceBoundOwner:
                raise ValueError('Exact registered source-bound owner required')
            owner.require_training_admission()
            branch=owner.gray_reverse
            if (type(branch) is not NativeGrayscaleFlow or branch.source_envelope!=self.source.envelope
                    or any(packet.image.image_dtype!=branch.detector_dtype for packet in packets)):
                raise ValueError('Registered native reverse-flow branch and exact detector/source envelope required')
            packet,=packets
            reference=branch.output.weight
            if reference.dtype not in (torch.float32,torch.float64):
                raise ValueError('Exact FP32 or FP64 native detector flow precision required')
            key=canonical([torch.initial_seed(),window.evidence_sha256,packet.source_coordinate[2]])
            seed=int.from_bytes(hashlib.sha256(key).digest()[:8],'big')%(2**63-1)
            generator=torch.Generator(device=reference.device).manual_seed(seed)
            shape=(packet.image.height,packet.image.width)
            time=torch.randint(1,2**24,(),device=reference.device,generator=generator).to(reference.dtype)/2**24
            noise=torch.randn(shape,device=reference.device,dtype=reference.dtype,generator=generator)
            return branch.objective(self.source,batch,split='train',noise=noise,time=time)
        return PreparedNativeTargets(count,None,evaluate)


def bind_deepbacs_mask_to_image(view,source,*,source_snapshot_sha256):
    return DeepBacsNativeRouteAdapter(view,source,source_snapshot_sha256).train_binding()
