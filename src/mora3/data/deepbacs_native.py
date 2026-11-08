"""Exact current DeepBacs raster views, separating original pixels and label maps."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from mora3.model.raster_source_envelope import RasterSourceEnvelope
from .corpus_morphology import EVIDENCE, POLICY
from .finite_union_native_rows import FiniteUnionNativeView, read_exact
from .full_source_raster_input import FullSourceRasterInput
from .provenance import Provenance
from .raster_morphology import RasterMorphologyTask


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _filter_bytes(view):
    return json.dumps(getattr(view, 'effective_filter', None), sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode()


@dataclass(frozen=True)
class NativeGrayMaskCondition:
    height: int
    width: int
    label_dtype: str
    label_bytes: bytes
    source_envelope: RasterSourceEnvelope
    provenance: Provenance

    def __post_init__(self):
        if (
            type(self.source_envelope) is not RasterSourceEnvelope or type(self.provenance) is not Provenance
            or type(self.height) is not int or not 1 <= self.height <= self.source_envelope.max_height
            or type(self.width) is not int or not 1 <= self.width <= self.source_envelope.max_width
            or type(self.label_dtype) is not str or type(self.label_bytes) is not bytes
        ):
            raise ValueError('Complete native grayscale mask condition and source envelope required')
        dtype = np.dtype(self.label_dtype)
        if dtype.kind != 'u' or dtype.itemsize not in (1, 2) or len(self.label_bytes) != self.height * self.width * dtype.itemsize:
            raise ValueError('Complete native unsigned integer mask condition required')

    def partition(self):
        # Assign first spatial occurrence ordinals. Repeated and disconnected
        # memberships stay equal. Source-specific arbitrary IDs are not features.
        labels = np.frombuffer(self.label_bytes, dtype=self.label_dtype)
        values, first, inverse = np.unique(labels, return_index=True, return_inverse=True)
        foreground = np.flatnonzero(values != 0)
        order = foreground[np.argsort(first[foreground])]
        mapping = np.zeros(len(values), dtype=np.int64)
        mapping[order] = np.arange(1, len(order) + 1, dtype=np.int64)
        return mapping[inverse].reshape(self.height, self.width)


@dataclass(frozen=True)
class DeepBacsRasterPacket:
    image: FullSourceRasterInput
    label_dtype: str
    label_bytes: bytes
    direction: str
    source_coordinate: tuple[str, int, str]
    original_split: str
    mask_provenance: Provenance

    def mask_condition(self):
        if self.direction != 'mask_to_image':
            raise ValueError('Observed mask conditions require the exact source reverse direction')
        return NativeGrayMaskCondition(
            self.image.height, self.image.width, self.label_dtype, self.label_bytes,
            self.image.source_envelope, self.mask_provenance,
        )

    def labels(self, *, device):
        labels = np.frombuffer(self.label_bytes, dtype=self.label_dtype)
        native = labels.reshape(self.image.height, self.image.width).astype(np.int64, copy=True)
        return torch.from_numpy(native).to(device)


class DeepBacsNativeSource:
    """Pinned current source rows. The historical source test is development-exposed."""

    def __init__(self, view, manifest_path, expected_sha256):
        if type(view) is not FiniteUnionNativeView:
            raise ValueError('Exact finite native DeepBacs view required')
        root = view.root
        path = (root / Path(manifest_path)).resolve()
        path.relative_to(root)
        raw = path.read_bytes()
        if _sha(raw) != expected_sha256:
            raise ValueError('Exact full native grayscale envelope checksum required')
        proof = json.loads(raw)
        if (
            proof['schema'] != 'Mora3.CompleteCurrentNativeImageEnvelope/v2'
            or proof['exact_grayscale_source_frame'] is not True
            or proof['original_test_development_exposed'] is not True
            or proof['target_masks_not_image_to_mask_conditions'] is not True
        ):
            raise ValueError('Complete source grayscale roles and target exclusion proof required')
        filter_ref = proof['source_effective_view']
        filter_path = (root / filter_ref['path']).resolve()
        filter_path.relative_to(root)
        filter_raw = read_exact(dict(filter_ref, bytes=filter_path.stat().st_size), root=root)
        canonical_filter = json.dumps(json.loads(filter_raw), sort_keys=True,
                                      separators=(',', ':'), allow_nan=False).encode()
        if _filter_bytes(view) != canonical_filter:
            raise ValueError('Grayscale source envelope and current effective view differ')
        self._filter_sha256 = _sha(canonical_filter)
        self._view = view
        self._rows = {
            (row['source_shard']['path'], row['absolute_original_row_ordinal']): (
                row['source_payload_sha256'], row['direction'], row['original_split'],
                tuple(row['shape']), row['source_image_sha256'], row['source_mask_sha256'],
            )
            for row in proof['rows']
        }
        if not self._rows or len(self._rows) != len(proof['rows']) or len(self._rows) != proof['complete_retained_rows']:
            raise ValueError('Complete distinct original grayscale coordinates required')
        self.envelope = RasterSourceEnvelope(expected_sha256, proof['max_height'], proof['max_width'], None)

    def packets(self, batch, *, split, direction):
        if split not in ('train', 'test') or direction not in ('image_to_mask', 'mask_to_image'):
            raise ValueError('Exact DeepBacs source train/test and target direction required')
        if _sha(_filter_bytes(self._view)) != self._filter_sha256:
            raise ValueError('Bound DeepBacs effective source filter changed')
        self._view.require_native_rows(batch, split=split)
        path = batch.source_shard['path']
        packets = []
        for ordinal, row in zip(batch.source_row_ordinals, batch.native_rows.to_pylist()):
            # Original source shards may contain both directions.
            if row['task_name'] != direction:
                continue
            raw = row['native_payload']
            digest = _sha(raw)
            if (
                (row['source_family'], row['source_revision'], row['native_schema'], row['split'],
                 row['evidence_class'], row['split_policy'])
                != ('DeepBacs-Staph', 'Zenodo5550933.v1', 'mora3.RasterMorphologyTask/v1', split, EVIDENCE, POLICY)
                or row['admitted_for_task'] is not True or digest != row['native_payload_sha256']
            ):
                raise ValueError('Exact native DeepBacs source role, rights envelope and payload required')
            task = RasterMorphologyTask.from_bytes(raw)
            expected = self._rows.get((path, ordinal))
            if (
                expected != (digest, direction, split, (task.height, task.width), task.image_sha256, task.mask_sha256)
                or (task.direction, task.split, task.source_id, task.archive_sha256)
                != (direction, split, row['record_id'], row['source_sha256'])
                or task.source != 'https://zenodo.org/records/5550933' or task.revision != 'v1'
                or task.license != 'CC-BY-4.0' or row['full_axis_values'] != task.height * task.width
            ):
                raise ValueError('Complete original DeepBacs raster identity differs')
            image_bytes = bytes.fromhex(task.image_hex)
            image = FullSourceRasterInput(
                task.height, task.width, task.image_dtype, image_bytes, _sha(image_bytes), self.envelope,
                Provenance(task.source, task.revision, task.source_id, task.image_sha256, task.license),
            )
            packets.append(DeepBacsRasterPacket(
                image, task.label_dtype, bytes.fromhex(task.labels_hex), direction, (path, ordinal, digest), split,
                Provenance(task.source, task.revision, task.source_id + '#mask', task.mask_sha256, task.license),
            ))
        if not packets:
            raise ValueError('Native batch has no selected original DeepBacs direction')
        return tuple(packets)
