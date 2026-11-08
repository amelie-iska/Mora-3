"""Exact original-role Rfam prediction and generation route bindings."""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import re

import pyarrow.parquet as pq

from mora3.data.admet import canonical
from mora3.data.finite_union_native_rows import FiniteUnionNativeView, read_exact
from mora3.data.rfam_family import _project_original_row
from mora3.model.full_owner import SourceBoundOwner
from mora3.model.rfam_family import RfamFamilyModel
from mora3.model.source_assembly_inputs import PINS

from .primary_native_routes import (
    FiniteNativeTrainRoute, NativeRouteBinding, NativeRouteWindow, PreparedNativeTargets,
    finite_view_policy_sha256,
)


FAMILY = 'Rfam15.1-source-family-prior'
SCHEMA = 'mora3.CorpusTask/v1'
VIEW = '11b7bb1121567ec3098c044c67a9254b954242f37e9f43a4002655c87c29c046'
MANIFEST_PATH, MANIFEST_SHA = PINS['rfam_manifest']


@dataclass(frozen=True)
class RfamCampaignAdapter:
    view: FiniteUnionNativeView
    direction: str
    source_snapshot_sha256: str
    batch_size: int = 1

    def __post_init__(self):
        if (type(self.direction) is not str or self.direction not in ('prediction', 'generation')
                or type(self.source_snapshot_sha256) is not str
                or not re.fullmatch('[0-9a-f]{64}', self.source_snapshot_sha256)
                or type(self.batch_size) is not int or self.batch_size < 1
                or finite_view_policy_sha256(self.view) != VIEW):
            raise ValueError('Exact current Rfam source view, direction and bounded batch required')
        raw = (self.view.root / MANIFEST_PATH).read_bytes()
        if sha256(raw).hexdigest() != MANIFEST_SHA:
            raise ValueError('Exact native Rfam manifest digest required')
        manifest = json.loads(raw)
        shards = {item['path']: item for item in manifest['shards']}
        relevant = {
            item['source_shard']['path']: item for item in self.view.descriptors
            if any(row['source_family'] == FAMILY for row in item['source_task_schema_counts'])
        }
        if (set(relevant) != set(shards)
                or any(path in shards for path, _ in self.view._ordinal_exclusions)
                or any(
                    any(item['source_shard'][key] != shards[path][key] for key in ('sha256', 'bytes', 'rows'))
                    or any(item['original_group_sidecar'][key] != manifest['effective_group_lookup'][path][key]
                           for key in ('sha256', 'bytes', 'rows'))
                    for path, item in relevant.items()
                )):
            raise ValueError('Current Rfam source or authoritative group identity differs from original manifest')
        object.__setattr__(self, '_manifest', manifest)
        object.__setattr__(self, '_manifest_wire', canonical(manifest))
        object.__setattr__(self, '_axis', tuple(manifest['family_ids']))
        object.__setattr__(self, '_shards', shards)
        object.__setattr__(self, '_paths', tuple(sorted(relevant)))

    @property
    def task_key(self):
        return f'Rfam_source_family_{self.direction}_prior'

    def executable_binding(self):
        if (finite_view_policy_sha256(self.view) != VIEW or canonical(self._manifest) != self._manifest_wire
                or tuple(self._manifest['family_ids']) != self._axis):
            raise ValueError('Exact Rfam source view, manifest and family axis required')
        return dict(schema='Mora3.RfamCampaignAdapter/v1', task_key=self.task_key, family=FAMILY,
                    role='train', native_schema=SCHEMA, current_view_policy_sha256=VIEW,
                    source_snapshot_sha256=self.source_snapshot_sha256,
                    manifest_sha256=MANIFEST_SHA, family_axis_sha256=sha256(canonical(self._axis)).hexdigest(),
                    batch_size=self.batch_size, direction=self.direction)

    def train_binding(self):
        self.executable_binding()
        route = FiniteNativeTrainRoute(
            self.view, task_key=self.task_key, source_snapshot_sha256=self.source_snapshot_sha256,
            native_view_policy_sha256=VIEW, source_family=FAMILY, task_name=self.task_key,
            native_schema=SCHEMA, source_revisions=(self._manifest['source_release'],),
            source_paths=self._paths, batch_size=self.batch_size,
        )
        target = 'observed positive source family assignments' if self.direction == 'prediction' else 'native nucleotide symbols and EOS'
        return NativeRouteBinding(route, self.prepare, 'native conditional negative log likelihood in nats', target)

    def prepare(self, window):
        self.executable_binding()
        if type(window) is not NativeRouteWindow or not 1 <= window.original_source_rows <= self.batch_size:
            raise ValueError('Exact bounded original Rfam train window required')
        batch = window.batch
        self.view.require_native_rows(batch, split='train')
        path = batch.source_shard['path']
        shard = self._shards[path]
        side = self._manifest['effective_group_lookup'][path]
        groups = pq.read_table(BytesIO(read_exact(side, root=self.view.root))).to_pylist()
        if len(groups) != shard['rows']:
            raise ValueError('Complete original Rfam group sidecar required')
        rows = batch.native_rows.to_pylist()
        if any(row['task_name'] != self.task_key for row in rows):
            raise ValueError('Original Rfam direction differs from route')
        packets = tuple(
            _project_original_row(row, groups[ordinal], m=self._manifest, axis=self._axis,
                                  shard=shard, side=side, expected_manifest_sha256=MANIFEST_SHA,
                                  direction=self.direction)
            for row, ordinal in zip(rows, batch.source_row_ordinals, strict=True)
        )
        prediction = self.direction == 'prediction'
        count = (sum(len(packet.targets()['source_Rfam_family_ids']) for packet in packets) if prediction
                 else sum(len(packet.targets()['source_native_sequence']) + 1 for packet in packets))

        def evaluate(owner):
            if type(owner) is not SourceBoundOwner:
                raise ValueError('Exact registered source-bound owner required')
            owner.require_training_admission()
            branch = owner.core.parent.parent.rfam
            if (type(branch) is not RfamFamilyModel or branch.family_axis != self._axis
                    or branch.manifest_sha256 != MANIFEST_SHA):
                raise ValueError('Exact registered native Rfam branch and complete family axis required')
            if prediction:
                return branch.prediction_objective(packets)
            return branch.generation_objective(packets)

        return PreparedNativeTargets(count, None, evaluate)
