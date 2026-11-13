"""Complete original DeepBacs train-route source census, not a campaign launch.

The native campaign's frozen two-window per-trial budget is NOT changed. This
read-only *source qualification* exhausts its original seven training images in
both task directions under both historical source identities; it reads only
source/provenance columns, not task targets, from development-exposed test data.
"""
from hashlib import sha256
from pathlib import Path
import json
import re

import pyarrow.parquet as pq

from mora3.data.finite_union_native_rows import full9, read_exact
from mora3.research.common_source_deepbacs_pair import (
    _DIRECTIONS, _FAMILY, _VIEW_POLICY, _direction_signature,
    require_deepbacs_source_pair,
)
from mora3.research.common_source_native_replay import (
    _source_fingerprint, _checked_two_window_transition, _checked_rebound_position,
)
from mora3.research.deepbacs_campaign_routes import DeepBacsNativeRouteAdapter
from mora3.research.deepbacs_forward_campaign_route import DeepBacsForwardCampaignAdapter
from mora3.research.remaining_eight_registry import load_assets
from mora3.research.primary_campaign_source_rebind import load_inactive_common_source_rebind
from mora3.research.primary_source_factory_preflight import Ref, document
from mora3.research.primary_native_routes import finite_view_policy_sha256
from mora3.research.mup_campaign import executable_adapter_sha256


def _sha(data):
    return sha256(data).hexdigest()


def _provider_series(record_id):
    """The shared *filename stem*, NOT proof of a shared biological experiment."""
    if type(record_id) is not str or not record_id.endswith('.tif'):
        raise ValueError('Original TIFF record identifier required')
    stem = Path(record_id).stem
    return re.sub(r'_[0-9]{3}$', '', stem).rstrip('_')


def require_role_group_census(groups):
    """Conserve source IDs within each role; never treat test as independent."""
    if type(groups) is not dict or set(groups) != {'train', 'test'}:
        raise ValueError('Both exact original source roles required')
    parents = {}
    for role in ('train', 'test'):
        values = groups[role]
        if type(values) is not list:
            raise ValueError('Role groups must be original ordered pairs')
        seen = set()
        parents[role] = set()
        for entry in values:
            if (type(entry) is not tuple or len(entry) != 2
                    or any(type(x) is not str or not x for x in entry)):
                raise ValueError('Original source/group strings required')
            source, parent = entry
            if source in seen:
                raise ValueError('duplicate original source identity within role')
            seen.add(source)
            parents[role].add(parent)
    return {
        'train_distinct_source_ids': len(groups['train']),
        'test_distinct_source_ids': len(groups['test']),
        'train_distinct_literal_parent_groups': len(parents['train']),
        'test_distinct_literal_parent_groups': len(parents['test']),
        'literal_parent_group_overlap': sorted(parents['train'] & parents['test']),
        'independent_heldout_accepted': False,
    }


def _original_role_metadata(root, view, role):
    """Read original role metadata and SHA-pinned group sidecar, not test targets."""
    desc = [
        d for d in view.descriptors
        if any(row['source_family'] == _FAMILY and row['split'] == role
               for row in d['source_task_schema_counts'])
    ]
    if len(desc) != 1:
        raise ValueError('Expected exactly one original DeepBacs shard per role')
    d = desc[0]
    shard = d['source_shard']
    path = root / shard['path']
    before = full9(path)
    # Verify immutable whole-file digest as opaque bytes; no test target decoding.
    source_bytes = read_exact(shard, root=root)
    if full9(path) != before:
        raise ValueError('Source identity changed during opaque hash verification')
    del source_bytes
    if d['source_full9'] != list(before):
        raise ValueError('Original source inode/size/time evidence differs')
    read_exact(d['original_group_sidecar'], root=root)
    sidecar = pq.read_table(root / d['original_group_sidecar']['path'])
    cols = ('task_id', 'record_id', 'task_name', 'split', 'source_family', 'source_revision')
    table = pq.read_table(path, columns=list(cols))
    if sidecar.num_rows != table.num_rows or table.num_rows != shard['rows']:
        raise ValueError('Original role/sidecar row conservation differs')
    original = table.to_pylist()
    groups = sidecar.to_pylist()
    if any(
        row['split'] != role or row['source_family'] != _FAMILY
        or row['source_revision'] != 'Zenodo5550933.v1'
        or row['task_name'] != _DIRECTIONS[i % 2]
        or row['task_id'] != groups[i]['task_id']
        or groups[i]['source_file'] != shard['path']
        or groups[i]['evidence_scope'] != 'identity/leakagegroupsonly;notpredictorfeaturesorsamecell/effectproof'
        for i, row in enumerate(original)
    ):
        raise ValueError('Original DeepBacs source/sidecar roles or task IDs differ')
    if len(original) % 2:
        raise ValueError('Original DeepBacs directions do not form complete pairs')
    acquisitions = []
    for i in range(0, len(original), 2):
        left, right = original[i:i+2]
        left_groups = json.loads(groups[i]['effective_groups_json'])
        right_groups = json.loads(groups[i+1]['effective_groups_json'])
        left_ids = json.loads(groups[i]['source_group_ids_json'])
        right_ids = json.loads(groups[i+1]['source_group_ids_json'])
        if (left['record_id'] != right['record_id']
                or left_groups != right_groups or left_ids != right_ids
                or any(type(s) is not str or not s for s in left_groups)
                or not any(x.startswith('image_parent:') for x in left_groups)):
            raise ValueError('Original paired microscopy source/group identity differs')
        parent = next(x for x in left_groups if x.startswith('image_parent:'))
        if parent != 'image_parent:deepbacs:5550933:' + Path(left['record_id']).stem:
            raise ValueError('Original source parent has unexpected filename identity')
        acquisitions.append({
            'record_id': left['record_id'],
            'parent_group': parent.removeprefix('image_parent:'),
            'provider_filename_series': _provider_series(left['record_id']),
            'original_source_ordinals': [i, i+1],
            'source_task_id_sha256': [_sha(x['task_id'].encode()) for x in (left,right)],
            'effective_groups_sha256': _sha(json.dumps(left_groups, sort_keys=True).encode()),
        })
    if len({r['record_id'] for r in acquisitions}) != len(acquisitions):
        raise ValueError('Original source acquired twice under same role')
    return {
        'original_role': role,
        'source_shard': {'path': shard['path'], 'sha256': shard['sha256'],
                         'bytes': shard['bytes'], 'rows': shard['rows']},
        'group_sidecar_sha256': d['original_group_sidecar']['sha256'],
        'task_rows': len(original),
        'acquisitions': acquisitions,
    }


def _exhaust_train_route(adapter, expected):
    route = adapter.train_binding().route
    iterator = route.iter_from(route.initial_cursor)
    previous = route._cursor_position(route.initial_cursor)
    rows = []
    try:
        for _ in range(expected):
            window = next(iterator)
            after = route._cursor_position(window.cursor_after)
            _checked_two_window_transition(previous, after)
            signature, detail = _direction_signature(adapter, window)
            rows.append((window, signature, detail, after))
            previous = after
        try:
            next(iterator)
        except StopIteration:
            pass
        else:
            raise ValueError('Original train route contains unaccounted source rows')
    except StopIteration as exc:
        raise ValueError('Original train route exhausted prematurely') from exc
    finally:
        close = getattr(iterator, 'close', None)
        if close is not None:
            close()
    return rows


def audit_original_deepbacs_complete_train(root):
    """Verify all seven train acquisitions and provenance-only test role census."""
    root = Path(root).resolve()
    if not root.is_dir() or Path.cwd().resolve() != root:
        raise ValueError('Exact original source checkout required')
    view, _, _, source, *_ = load_assets(root)
    policy = finite_view_policy_sha256(view)
    if policy != _VIEW_POLICY:
        raise ValueError('Reviewed original DeepBacs source view changed')
    context = load_inactive_common_source_rebind(root)
    manifest = document(root, context.evidence)
    candidate = Ref(**manifest['inactive_candidate_snapshot'])
    common = context.common_snapshot
    if candidate.sha256 == common.sha256:
        raise ValueError('Both independently reviewed source identities required')

    group_roles = {role: _original_role_metadata(root, view, role)
                   for role in ('train', 'test')}
    train, test = group_roles['train'], group_roles['test']
    if (train['task_rows'] != 14 or test['task_rows'] != 10
            or len(train['acquisitions']) != 7 or len(test['acquisitions']) != 5):
        raise ValueError('Original DeepBacs complete source census changed')
    # The original provider split is train/test only; never mint validation rows.
    if any(any(r['source_family'] == _FAMILY and r['split'] == 'validation'
               for r in d['source_task_schema_counts']) for d in view.descriptors):
        raise ValueError('Unreviewed DeepBacs validation role appeared')
    grouping = require_role_group_census({
        role: [(x['record_id'], x['parent_group'])
               for x in group_roles[role]['acquisitions']]
        for role in ('train', 'test')
    })

    source_rows = {}
    for direction in _DIRECTIONS:
        key = f'{_FAMILY}:{direction}'
        matching = [r for r in manifest['route_bindings'] if r['task_key'] == key]
        if len(matching) != 1:
            raise ValueError('Frozen original direction binding absent')
        plan = matching[0]
        if (plan['common_view_policy_sha256'] != policy
                or any(plan[name]['max_windows'] != 2 for name in ('candidate','common'))):
            raise ValueError('Frozen two-window per-trial plan changed')
        cls = DeepBacsForwardCampaignAdapter if direction == 'image_to_mask' else DeepBacsNativeRouteAdapter
        for label, snapshot in (('candidate', candidate), ('common', common)):
            adapter = cls(view, source, snapshot.sha256)
            binding = adapter.train_binding()
            declared = {'task_key': key,
                        'initial_cursor_sha256': _sha(binding.route.initial_cursor),
                        'adapter_sha256': executable_adapter_sha256(binding.prepare, root=Path(__file__).resolve().parents[3]),
                        'max_windows': 2}
            if (declared != {'task_key': key, **plan[label]}
                    or binding.route.source_snapshot_sha256 != snapshot.sha256
                    or binding.route.native_view_policy_sha256 != policy):
                raise ValueError('Original executable train route differs from frozen plan')
            source_rows[(direction, label)] = _exhaust_train_route(adapter, 7)

    pairs = []
    totals = dict.fromkeys(_DIRECTIONS, 0)
    for idx in range(7):
        paired = {}
        for direction in _DIRECTIONS:
            a = source_rows[(direction, 'candidate')][idx]
            b = source_rows[(direction, 'common')][idx]
            original, signature, detail, a_position = a
            rebound, counterpart, counterpart_detail, b_position = b
            if (signature != counterpart or detail != counterpart_detail
                    or original.batch.source_shard != rebound.batch.source_shard
                    or original.batch.source_row_ordinals != rebound.batch.source_row_ordinals
                    or not original.batch.native_rows.equals(rebound.batch.native_rows, check_metadata=True)
                    or _source_fingerprint(original) != _source_fingerprint(rebound)
                    or original.cursor_after == rebound.cursor_after):
                raise ValueError('Original candidate/common full-train raster row differs')
            _checked_rebound_position(a_position, b_position)
            paired[direction] = signature
            paired[direction+'_proof'] = {
                'native_payload_sha256': detail['native_payload_sha256'],
                'native_payload_bytes': detail['native_payload_bytes'],
                'candidate_cursor_sha256': _sha(original.cursor_after),
                'common_cursor_sha256': _sha(rebound.cursor_after),
            }
            totals[direction] += signature['observed_positions']
        row = require_deepbacs_source_pair(paired['image_to_mask'], paired['mask_to_image'])
        original_group = train['acquisitions'][idx]
        if (row['source_id'] != original_group['record_id']
                or row['parent_group'] != original_group['parent_group']
                or [row['forward_ordinal'], row['reverse_ordinal']]
                   != original_group['original_source_ordinals']):
            raise ValueError('Native full-train pair does not match original sidecar acquisition')
        row['native_payload_sha256_by_direction'] = {
            direction: paired[direction+'_proof']['native_payload_sha256']
            for direction in _DIRECTIONS
        }
        row['candidate_cursor_sha256_by_direction'] = {
            direction: paired[direction+'_proof']['candidate_cursor_sha256']
            for direction in _DIRECTIONS
        }
        row['common_cursor_sha256_by_direction'] = {
            direction: paired[direction+'_proof']['common_cursor_sha256']
            for direction in _DIRECTIONS
        }
        pairs.append(row)
    train_series = {x['provider_filename_series'] for x in train['acquisitions']}
    test_series = {x['provider_filename_series'] for x in test['acquisitions']}
    return {
        'schema': 'Mora3.DeepBacsCompleteOriginalTrainCensus/v1',
        'source_family': _FAMILY, 'source_revision': 'Zenodo5550933.v1',
        'source_view_policy_sha256': policy,
        'candidate_source_snapshot_sha256': candidate.sha256,
        'common_source_snapshot_sha256': common.sha256,
        'original_train_acquisitions': 7, 'original_train_task_rows': 14,
        'original_test_acquisitions_metadata_only': 5,
        'original_test_task_rows_metadata_only': 10,
        'original_validation_task_rows': 0,
        'original_source_roles': group_roles,
        'source_group_census': grouping,
        'train_test_provider_series_shared': bool(train_series & test_series),
        'shared_provider_filename_series': sorted(train_series & test_series),
        'original_train_pairs': pairs,
        'direction_target_positions': totals,
        'original_source_target_positions': sum(totals.values()),
        'complete_both_native_train_routes': True,
        'candidate_common_all_original_train_rows_equal': True,
        'independent_heldout_accepted': False,
        'training_admission': False, 'actual_campaign_runs': 0,
        'optimizer_updates': 0, 'target_runs': 0,
        'test_target_payload_decoded': False,
        'source_test_development_exposed': True,
        'original_train_windows_scanned_for_source_audit': 7,
        'frozen_per_trial_route_budget_unchanged': 2,
        'conclusion': 'Seven original train rasters exhausted in both directions; five development-exposed test rasters inspected only via source/group metadata. The common filename series is not proof of biological independence or leakage, and no held-out benchmark is admitted.',
    }
