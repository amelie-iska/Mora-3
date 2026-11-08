"""Original DeepBacs grayscale image-to-instance-mask campaign binding."""

from dataclasses import dataclass
import re

import torch

from mora3.data.admet import canonical
from mora3.data.deepbacs_native import DeepBacsNativeSource, DeepBacsRasterPacket
from mora3.data.finite_union_native_rows import FiniteUnionNativeView
from mora3.model.full_owner import SourceBoundOwner
from mora3.model.instance_affinity import affinity_objective
from mora3.model.morphology import MorphologyInstance, foreground_objective

from .primary_native_routes import (
    FiniteNativeTrainRoute, NativeRouteBinding, NativeRouteWindow, PreparedNativeTargets,
    finite_view_policy_sha256,
)


FAMILY = 'DeepBacs-Staph'
TASK = 'image_to_mask'
SCHEMA = 'mora3.RasterMorphologyTask/v1'
REVISION = 'Zenodo5550933.v1'


def grayscale_instance_loss(model, packet):
    if (type(packet) is not DeepBacsRasterPacket or packet.direction != TASK
            or packet.original_split != 'train'):
        raise ValueError('Source-verified original DeepBacs forward train packet required')
    # The exact source reader owns training-role admission for the input-only raster.
    logits = model(packet.image)
    labels = packet.labels(device=logits.device)
    ids = torch.unique(labels)
    ids = ids[ids != 0]
    instances = labels[None] == ids[:, None, None]
    return foreground_objective(logits[0], instances) + affinity_objective(logits[1:], instances)


@dataclass(frozen=True)
class DeepBacsForwardCampaignAdapter:
    view: FiniteUnionNativeView
    source: DeepBacsNativeSource
    source_snapshot_sha256: str

    def __post_init__(self):
        if (type(self.view) is not FiniteUnionNativeView or type(self.source) is not DeepBacsNativeSource
                or type(self.source_snapshot_sha256) is not str
                or not re.fullmatch('[0-9a-f]{64}', self.source_snapshot_sha256)):
            raise ValueError('Exact current DeepBacs source, view and campaign snapshot required')
        original = self.source._view
        paths = frozenset(path for path, _ in self.source._rows)
        if (original.root != self.view.root or canonical(original.manifest) != canonical(self.view.manifest)
                or any(original._sealed_by_path.get(p) != self.view._sealed_by_path.get(p) for p in paths)
                or {p for p in original._ordinal_exclusions if p[0] in paths}
                != {p for p in self.view._ordinal_exclusions if p[0] in paths}):
            raise ValueError('Current DeepBacs source coordinates differ from the original raster census')
        object.__setattr__(self, '_policy', canonical(self._binding()))

    def _binding(self):
        return dict(
            schema='Mora3.DeepBacsForwardCampaignAdapter/v1',
            source_snapshot_sha256=self.source_snapshot_sha256,
            view_policy_sha256=finite_view_policy_sha256(self.view),
            census_view_policy_sha256=finite_view_policy_sha256(self.source._view),
            raster_envelope=self.source.envelope.binding(), source_family=FAMILY,
            source_task=TASK, native_schema=SCHEMA, source_revision=REVISION, split='train', batch_size=1,
            objective='complete-pixel foreground BCE plus valid local right/down affinity BCE',
        )

    def executable_binding(self):
        doc = self._binding()
        if canonical(doc) != self._policy:
            raise ValueError('Original DeepBacs source or raster envelope changed')
        return doc

    def train_binding(self):
        doc = self.executable_binding()
        paths = tuple(sorted(item['source_shard']['path'] for item in self.view.descriptors if any(
            (row['source_family'], row['task_name'], row['native_schema'], row['source_revision'], row['split'],
             row['eligible_view_disposition']) == (FAMILY, TASK, SCHEMA, REVISION, 'train', 0)
            for row in item['source_task_schema_counts']
        )))
        route = FiniteNativeTrainRoute(
            self.view, task_key=f'{FAMILY}:{TASK}', source_snapshot_sha256=self.source_snapshot_sha256,
            native_view_policy_sha256=doc['view_policy_sha256'], source_family=FAMILY, task_name=TASK,
            native_schema=SCHEMA, source_revisions=(REVISION,), source_paths=paths, batch_size=1,
        )
        return NativeRouteBinding(
            route, self.prepare, 'sum of mean foreground and mean local-affinity Bernoulli NLL in nats',
            'complete original pixels plus valid right/down affinities',
        )

    def prepare(self, window):
        self.executable_binding()
        if type(window) is not NativeRouteWindow or window.original_source_rows != 1:
            raise ValueError('Exact single original DeepBacs train window required')
        self.view.require_native_rows(window.batch, split='train')
        packet, = self.source.packets(window.batch, split='train', direction=TASK)
        height, width = packet.image.height, packet.image.width
        # Original masks have one source label per pixel, so only border edges lack targets.
        count = height * width + height * (width - 1) + (height - 1) * width

        def evaluate(owner):
            if type(owner) is not SourceBoundOwner:
                raise ValueError('Exact registered source-bound owner required')
            owner.require_training_admission()
            branch = owner.gray_forward
            if type(branch) is not MorphologyInstance or branch.source_envelope != self.source.envelope:
                raise ValueError('Exact registered grayscale instance branch and source envelope required')
            return grayscale_instance_loss(branch, packet)

        return PreparedNativeTargets(count, None, evaluate)
