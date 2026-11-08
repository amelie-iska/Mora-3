"""Native grayscale ingress and exact integer-map affinity loss."""
from dataclasses import asdict, replace
import hashlib
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from mora3.data.finite_union_native_rows import FiniteUnionNativeView, full9
from mora3.data.raster_morphology import RasterMorphologyTask
from mora3.model.instance_affinity import affinity_objective
from mora3.model.morphology import MorphologyInstance, foreground_objective
import test_finite_union_native_rows as native_fixture


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fixture_source(tmp_path, *, direction='image_to_mask', detector_dtype='|u1'):
    from mora3.data.deepbacs_native import DeepBacsNativeSource
    fixture = native_fixture.NativeViewTests()
    fixture.root = tmp_path
    image = np.arange(20, dtype=detector_dtype).tobytes()
    labels = bytes([0, 3, 3, 0, 0, 5, 3, 3, 3, 0, 5, 5, 0, 7, 7, 5, 0, 0, 7, 7])
    prototype = RasterMorphologyTask(
        4, 5, detector_dtype, '|u1', image.hex(), labels.hex(), 'https://zenodo.org/records/5550933',
        'v1', 'fixture.tif', 'a' * 64, sha(image), sha(labels), 'CC-BY-4.0', 'fixture-parent',
        'train', 'Staphylococcus aureus', 'fixture', 'fluorescence',
        'manual ellipses drawn on Nile Red fluorescence; whole-cell instance labels', None,
    )
    prototype = replace(prototype, direction=direction)
    originals = (prototype, replace(prototype, split='test'))
    rows = []
    for task in originals:
        raw = task.canonical_bytes()
        rows.append(dict(
            task_id='fixture-' + task.split, record_id=task.source_id,
            source_family='DeepBacs-Staph', source_revision='Zenodo5550933.v1',
            native_schema='mora3.RasterMorphologyTask/v1', task_name=direction,
            source_sha256=task.archive_sha256, split=task.split, full_axis_values=20,
            evidence_class='source_observed_image_manual_ellipse_instance_annotation',
            split_policy='official-source-train-test;test-development-exposed-not-blind',
            admitted_for_task=True, native_payload=raw, native_payload_sha256=sha(raw),
        ))
    parquet_path = tmp_path / 'original.parquet'
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)
    descriptor = fixture.descriptor([0, 2])
    native_bytes = parquet_path.read_bytes()
    descriptor['source_shard'].update(bytes=len(native_bytes), sha256=sha(native_bytes))
    descriptor['source_full9'] = list(full9(parquet_path))
    descriptor['original_group_sidecar'] = fixture.put('groups.parquet', b'original source guard fixture')
    path, expected_sha = fixture.manifest([descriptor])
    view = FiniteUnionNativeView(path, expected_sha, root=tmp_path)
    view.effective_filter = dict(schema='fixture-native-filter', retained=2)
    effective = fixture.put('effective.json', json.dumps(view.effective_filter).encode())
    proof = dict(
        schema='Mora3.CompleteCurrentNativeImageEnvelope/v2', exact_grayscale_source_frame=True,
        original_test_development_exposed=True, target_masks_not_image_to_mask_conditions=True,
        complete_retained_rows=2, max_height=4, max_width=5,
        source_effective_view={k: v for k, v in effective.items() if k != 'bytes'},
        rows=[dict(source_shard=descriptor['source_shard'], absolute_original_row_ordinal=i,
                   source_payload_sha256=row['native_payload_sha256'], direction=direction,
                   original_split=row['split'], shape=[4, 5], source_image_sha256=prototype.image_sha256,
                   source_mask_sha256=prototype.mask_sha256)
              for i, row in enumerate(rows)],
    )
    ref = fixture.put('proof.json', json.dumps(proof).encode())
    source = DeepBacsNativeSource(view, tmp_path / ref['path'], ref['sha256'])
    batches = tuple(next(view.iter_native_task_batches(split=split)) for split in ('train', 'test'))
    return source, batches, originals


def test_native_boundary_keeps_integer_label_target_out_of_pixels(tmp_path):
    source, batches, originals = fixture_source(tmp_path)
    packet, = source.packets(batches[0], split='train', direction='image_to_mask')
    assert not hasattr(packet.image, 'labels')
    assert torch.equal(packet.image.input_image(), originals[0].input_image())
    labels = packet.labels(device='cpu')
    assert set(labels.flatten().tolist()) == {0, 3, 5, 7}
    with pytest.raises(ValueError):
        source.packets(batches[0], split='test', direction='image_to_mask')
    with pytest.raises(ValueError):
        source.packets(batches[0], split='train', direction='mask_to_image')


def test_integer_map_objective_agrees_with_all_reference_masks_and_gradients(tmp_path):
    from mora3.model.native_grayscale_objective import grayscale_integer_label_loss
    source, batches, originals = fixture_source(tmp_path)
    packet, = source.packets(batches[0], split='train', direction='image_to_mask')
    target = packet.labels(device='cpu')
    logits = torch.linspace(-4, 4, 60, dtype=torch.float64).reshape(3, 4, 5).requires_grad_()
    baseline = originals[0].instances()
    actual = grayscale_integer_label_loss(logits, target)
    expected = foreground_objective(logits[0], baseline) + affinity_objective(logits[1:], baseline)
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
    ga, = torch.autograd.grad(actual, logits)
    gb, = torch.autograd.grad(expected, logits)
    torch.testing.assert_close(ga, gb, rtol=1e-14, atol=1e-14)
    assert torch.equal(actual, grayscale_integer_label_loss(logits, target * 11))


def test_same_model_uses_train_and_development_test_without_target_context(tmp_path):
    from mora3.model.native_grayscale_objective import native_grayscale_segmentation_objective
    source, batches, _ = fixture_source(tmp_path)
    model = MorphologyInstance(8, parameterization_policy='mup-image-v1', base_width=8,
                               source_envelope=source.envelope).double()
    loss = native_grayscale_segmentation_objective(model, source, batches[0], split='train')
    loss.backward()
    assert model.head.weight.grad is not None and torch.count_nonzero(model.head.weight.grad)
    with pytest.raises(ValueError, match='evaluation'):
        native_grayscale_segmentation_objective(model, source, batches[1], split='test')
    with torch.no_grad():
        observed = native_grayscale_segmentation_objective(model, source, batches[1], split='test')
        torch.testing.assert_close(loss.detach(), observed, rtol=0, atol=0)
