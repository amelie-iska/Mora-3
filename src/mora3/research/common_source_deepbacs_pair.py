"""Original DeepBacs raster-direction source conservation under two N29 bindings.

Only original source train rows, never development-exposed test rows, are read.
Neither source-task pairing nor original labels establish globally independent
biological splits. The original native optimizer/owner is never constructed.
"""
from hashlib import sha256
from pathlib import Path
import re

import numpy as np

from mora3.data.admet import canonical
from mora3.data.raster_morphology import RasterMorphologyTask
from mora3.research.deepbacs_campaign_routes import DeepBacsNativeRouteAdapter
from mora3.research.deepbacs_forward_campaign_route import DeepBacsForwardCampaignAdapter
from mora3.research.remaining_eight_registry import load_assets
from mora3.research.primary_campaign_source_rebind import load_inactive_common_source_rebind
from mora3.research.primary_source_factory_preflight import Ref, document
from mora3.research.primary_native_routes import finite_view_policy_sha256
from mora3.research.mup_campaign import executable_adapter_sha256
from mora3.research.common_source_native_replay import (
    _source_fingerprint, _first, _checked_rebound_position,
    _checked_two_window_transition,
)

_FAMILY = 'DeepBacs-Staph'
_SCHEMA = 'mora3.RasterMorphologyTask/v1'
_REVISION = 'Zenodo5550933.v1'
_DIRECTIONS = ('image_to_mask', 'mask_to_image')
_VIEW_POLICY = 'c3812ecfcfea8a01f81769bf5ef6fb3880e04a0ac054a6349e9272db14765393'
_HEX = re.compile(r'[0-9a-f]{64}')


def _sha(raw):
    return sha256(raw).hexdigest()


def require_deepbacs_source_pair(forward, reverse):
    """Require identical raster and source grouping, but DISTINCT task records.

    The roles are genuinely opposite. The same acquisition is not two
    independent measurements, and same-source equality is not a split audit.
    """
    keys = {
        'direction', 'source_id', 'original_split', 'parent_group',
        'source_shard_sha256', 'image_sha256', 'mask_sha256',
        'decoded_image_sha256', 'decoded_mask_sha256',
        'image_shape', 'image_dtype', 'label_dtype', 'source_license',
        'source_revision', 'source_image_bytes', 'source_mask_bytes',
        'source_ordinal', 'observed_positions', 'condition_keys', 'target_keys',
    }
    if (type(forward) is not dict or type(reverse) is not dict
            or set(forward) != keys or set(reverse) != keys):
        raise ValueError('Two complete exact directional raster source records required')
    roles = {
        'image_to_mask': (('image',), ('labels',)),
        'mask_to_image': (('labels',), ('image',)),
    }
    for direction, value in zip(_DIRECTIONS, (forward, reverse), strict=True):
        shape = value['image_shape']
        if (value['direction'] != direction
                or (value['condition_keys'], value['target_keys']) != roles[direction]
                or value['original_split'] != 'train'
                or value['source_revision'] != _REVISION
                or value['source_license'] != 'CC-BY-4.0'
                or any(type(value[k]) is not str or not value[k] for k in ('source_id', 'parent_group'))
                or any(type(value[k]) is not str or _HEX.fullmatch(value[k]) is None
                       for k in ('source_shard_sha256', 'image_sha256', 'mask_sha256',
                                 'decoded_image_sha256', 'decoded_mask_sha256'))
                or type(shape) is not list or len(shape) != 2
                or any(type(x) is not int or not 1 <= x <= 512 for x in shape)
                or type(value['source_ordinal']) is not int or value['source_ordinal'] < 0):
            raise ValueError('Original DeepBacs role, source provenance or shape differs')
        h, w = shape
        for dtype_key, bytes_key in (
            ('image_dtype', 'source_image_bytes'), ('label_dtype', 'source_mask_bytes'),
        ):
            try:
                dtype = np.dtype(value[dtype_key])
            except (TypeError, ValueError) as exc:
                raise ValueError('Source-native unsigned raster dtype required') from exc
            if (dtype.kind != 'u' or dtype.itemsize not in (1, 2)
                    or type(value[bytes_key]) is not int
                    or value[bytes_key] != h * w * dtype.itemsize):
                raise ValueError('Original complete aligned raster byte counts required')
        positions = h * w if direction == 'mask_to_image' else h * w + h * (w-1) + (h-1) * w
        if type(value['observed_positions']) is not int or value['observed_positions'] != positions:
            raise ValueError('Exact original directional raster supervision denominator required')
    matching = keys - {'direction', 'source_ordinal', 'observed_positions', 'condition_keys', 'target_keys'}
    if (any(forward[k] != reverse[k] for k in matching)
            or forward['source_ordinal'] == reverse['source_ordinal']):
        raise ValueError('Image-to-mask and mask-to-image must be paired source records, not one task')
    return dict(source_id=forward['source_id'], parent_group=forward['parent_group'],
                image_sha256=forward['image_sha256'], mask_sha256=forward['mask_sha256'],
                decoded_image_sha256=forward['decoded_image_sha256'],
                decoded_mask_sha256=forward['decoded_mask_sha256'],
                image_shape=forward['image_shape'], image_dtype=forward['image_dtype'],
                label_dtype=forward['label_dtype'], source_license=forward['source_license'],
                source_revision=forward['source_revision'], source_image_bytes=forward['source_image_bytes'],
                source_mask_bytes=forward['source_mask_bytes'], forward_ordinal=forward['source_ordinal'],
                reverse_ordinal=reverse['source_ordinal'],
                forward_target_positions=forward['observed_positions'],
                reverse_target_positions=reverse['observed_positions'],
                condition_target_roles={
                    'image_to_mask': {'conditions': ['image'], 'targets': ['labels']},
                    'mask_to_image': {'conditions': ['labels'], 'targets': ['image']},
                })


def _direction_signature(adapter, window):
    """Read exact original Arrow/native bytes and prepare, but do NOT evaluate."""
    if type(adapter) is DeepBacsForwardCampaignAdapter:
        direction = 'image_to_mask'
    elif type(adapter) is DeepBacsNativeRouteAdapter:
        direction = 'mask_to_image'
    else:
        raise ValueError('Only exact original DeepBacs native adapters are supported')
    adapter.executable_binding()
    if window.original_source_rows != 1 or window.batch.original_split != 'train':
        raise ValueError('One genuine original DeepBacs training record required')
    row, = window.batch.native_rows.to_pylist()
    if (row['source_family'] != _FAMILY or row['task_name'] != direction
            or row['native_schema'] != _SCHEMA or row['source_revision'] != _REVISION
            or row['split'] != 'train' or not row['admitted_for_task']):
        raise ValueError('Exact original DeepBacs direction and source roles required')
    _, native_bytes, native_digest, ordinal = _source_fingerprint(window)
    task = RasterMorphologyTask.from_bytes(row['native_payload'])
    packet, = adapter.source.packets(window.batch, split='train', direction=direction)
    if (packet.direction != direction or task.direction != direction
            or (task.height, task.width) != (packet.image.height, packet.image.width)
            or task.image_dtype != packet.image.image_dtype or task.label_dtype != packet.label_dtype
            or packet.image.pixels != bytes.fromhex(task.image_hex)
            or packet.label_bytes != bytes.fromhex(task.labels_hex)
            or packet.image.pixel_sha256 != _sha(packet.image.pixels)
            or task.image_sha256 != packet.image.provenance.sha256
            or task.mask_sha256 != packet.mask_provenance.sha256
            or task.parent_group == '' or task.source_id != row['record_id']
            or packet.image.provenance.source_id != task.source_id
            or packet.mask_provenance.source_id != task.source_id + '#mask'):
        raise ValueError('Original complete image/mask bytes or source provenance changed')
    # Train-role supervised denominators are verified without invoking model code.
    prepared = adapter.prepare(window)
    source = dict(direction=direction, source_id=task.source_id,
                  original_split=task.split, parent_group=task.parent_group,
                  source_shard_sha256=window.batch.source_shard['sha256'],
                  image_sha256=task.image_sha256, mask_sha256=task.mask_sha256,
                  decoded_image_sha256=packet.image.pixel_sha256,
                  decoded_mask_sha256=_sha(packet.label_bytes),
                  image_shape=[task.height, task.width], image_dtype=task.image_dtype,
                  label_dtype=task.label_dtype, source_license=task.license,
                  source_revision=row['source_revision'],
                  source_image_bytes=len(packet.image.pixels),
                  source_mask_bytes=len(packet.label_bytes),
                  source_ordinal=ordinal, observed_positions=prepared.observed_positions,
                  condition_keys=('image',) if direction=='image_to_mask' else ('labels',),
                  target_keys=('labels',) if direction=='image_to_mask' else ('image',))
    if prepared.unavailable_reason is not None or not callable(prepared.evaluate):
        raise ValueError('Original supervised raster target cannot be prepared without model execution')
    return source, dict(native_source_fingerprint=_source_fingerprint(window)[0],
                        native_payload_sha256=native_digest, native_payload_bytes=native_bytes,
                        task_id_sha256=_sha(row['task_id'].encode()),
                        source_archive_sha256=task.archive_sha256,
                        exact_input_and_target_bytes=True, training_objective_evaluated=False)


def audit_original_deepbacs_two_direction_pairs(root, *,
                                               directions=_DIRECTIONS, max_windows=2):
    """Read two original acquisitions, two task directions, two frozen snapshots."""
    if type(directions) is not tuple or directions != _DIRECTIONS or type(max_windows) is not int or max_windows != 2:
        raise ValueError('Only the reviewed two-direction/two-window DeepBacs scope is permitted')
    root = Path(root).resolve()
    if not root.is_dir() or Path.cwd().resolve() != root:
        raise ValueError('Use the original pinned source checkout as the working directory')
    context = load_inactive_common_source_rebind(root)
    manifest = document(root, context.evidence)
    candidate = Ref(**manifest['inactive_candidate_snapshot'])
    common = context.common_snapshot
    if candidate.sha256 == common.sha256:
        raise ValueError('Candidate and common29 source identities must be distinct')
    view, _, _, source, *_ = load_assets(root)
    policy = finite_view_policy_sha256(view)
    if policy != _VIEW_POLICY:
        raise ValueError('Original protected DeepBacs source-view policy changed')

    scoped = {}
    for direction in _DIRECTIONS:
        key = f'{_FAMILY}:{direction}'
        planned, = tuple(row for row in manifest['route_bindings'] if row['task_key'] == key)
        if (planned['common_view_policy_sha256'] != policy
                or planned['common']['max_windows'] != 2
                or planned['candidate']['max_windows'] != 2):
            raise ValueError('Original bounded DeepBacs route plan changed')
        for label, snapshot in (('candidate', candidate), ('common', common)):
            cls = DeepBacsForwardCampaignAdapter if direction == 'image_to_mask' else DeepBacsNativeRouteAdapter
            adapter = cls(view, source, snapshot.sha256)
            binding = adapter.train_binding()
            actual = dict(task_key=key, initial_cursor_sha256=_sha(binding.route.initial_cursor),
                          adapter_sha256=executable_adapter_sha256(
                              binding.prepare, root=Path(__file__).resolve().parents[3]), max_windows=2)
            if (actual != planned[label] or binding.route.native_view_policy_sha256 != policy
                    or binding.route.source_snapshot_sha256 != snapshot.sha256):
                raise ValueError('Frozen original DeepBacs adapter/cursor differs')
            scoped[(direction, label)] = (adapter, binding)

    evidence_rows = {}
    for direction in _DIRECTIONS:
        for label in ('candidate', 'common'):
            adapter, binding = scoped[(direction, label)]
            route = binding.route
            iterator = route.iter_from(route.initial_cursor)
            position = route._cursor_position(route.initial_cursor)
            rows = []
            try:
                for _ in range(2):
                    window = next(iterator)
                    after = route._cursor_position(window.cursor_after)
                    _checked_two_window_transition(position, after)
                    rows.append((window, *_direction_signature(adapter, window)))
                    position = after
            except StopIteration as exc:
                raise ValueError('Original DeepBacs route exhausted before two windows') from exc
            finally:
                close = getattr(iterator, 'close', None)
                if close is not None: close()
            restarted = _first(route.iter_from(rows[0][0].cursor_after))
            original, signature, detail = rows[1]
            if (restarted.cursor_after != original.cursor_after
                    or restarted.evidence_sha256 != original.evidence_sha256
                    or not restarted.batch.native_rows.equals(original.batch.native_rows, check_metadata=True)
                    or _direction_signature(adapter, restarted) != (signature, detail)):
                raise ValueError('DeepBacs second original source record changed after cursor restart')
            evidence_rows[(direction,label)] = tuple(rows)

    pairs = []
    totals = dict.fromkeys(_DIRECTIONS, 0)
    for idx in range(2):
        paired = {}
        for direction in _DIRECTIONS:
            a, signature, details = evidence_rows[(direction,'candidate')][idx]
            b, same_signature, same_details = evidence_rows[(direction,'common')][idx]
            if (signature != same_signature or details != same_details
                    or a.batch.source_shard != b.batch.source_shard
                    or a.batch.source_row_ordinals != b.batch.source_row_ordinals
                    or not a.batch.native_rows.equals(b.batch.native_rows, check_metadata=True)
                    or _source_fingerprint(a) != _source_fingerprint(b)):
                raise ValueError('Candidate/common source image, mask, native types or bytes disagree')
            candidate_position = scoped[(direction,'candidate')][1].route._cursor_position(a.cursor_after)
            common_position = scoped[(direction,'common')][1].route._cursor_position(b.cursor_after)
            _checked_rebound_position(candidate_position, common_position)
            if a.cursor_after == b.cursor_after:
                raise ValueError('Distinct source snapshot identities require distinct native cursors')
            paired[direction] = signature
            paired[direction+'_details'] = dict(details,
                candidate_cursor_sha256=_sha(a.cursor_after),
                common_cursor_sha256=_sha(b.cursor_after),
                next_position=candidate_position)
            totals[direction] += signature['observed_positions']
        pair = require_deepbacs_source_pair(paired['image_to_mask'], paired['mask_to_image'])
        for direction in _DIRECTIONS:
            details = paired[direction+'_details']
            pair[direction+'_source_fingerprint'] = details['native_source_fingerprint']
            pair['candidate_'+direction+'_cursor_sha256'] = details['candidate_cursor_sha256']
            pair['common_'+direction+'_cursor_sha256'] = details['common_cursor_sha256']
            pair[direction+'_next_position'] = details['next_position']
            pair[direction+'_native_payload_sha256'] = details['native_payload_sha256']
        pair['candidate_common_cursor_positions_equal'] = True
        pairs.append(pair)
    if pairs[0]['image_sha256'] == pairs[1]['image_sha256'] or pairs[0]['source_id']==pairs[1]['source_id']:
        raise ValueError('Two distinct original DeepBacs acquisitions required')
    return dict(schema='Mora3.DeepBacsOriginalDirectionalPairs/v1',
                source_family=_FAMILY, candidate_source_snapshot_sha256=candidate.sha256,
                common_source_snapshot_sha256=common.sha256, original_source_view_policy_sha256=policy,
                original_raster_count=2, original_route_rows=4, paired_rows=tuple(pairs),
                direction_target_positions=totals,
                original_source_target_positions=sum(totals.values()),
                candidate_common_original_rows_equal=True,
                independent_second_window_restarts=True,
                source_test_development_exposed=True, global_group_split_accepted=False,
                matched_owner_validation_complete=False, training_admission=False,
                actual_campaign_runs=0, optimizer_updates=0,
                conclusion='Source directions paired on the same original rasters; development-exposed test remains ineligible as an independent heldout benchmark')
