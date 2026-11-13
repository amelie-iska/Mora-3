"""Bounded, resumable, complete-original-source GPerturb paired-cell verification.

This is source qualification, not training. Only source-original train-only
observations are opened; no learned objective, GPU or optimizer is invoked.
Each immutable chunk pins native response and inverse cursors for the next call.
"""
from base64 import b64decode, b64encode
from collections import Counter
from hashlib import sha256
from itertools import zip_longest
from pathlib import Path
import json

from mora3.data.admet import canonical
from mora3.research.common_source_gperturb_pair import (
    _POLICY, _direction_signature, require_paired_source_semantics,
)
from mora3.research.gperturb_full_common_response import audited_response_record
from mora3.research.gperturb_full_common_intervention import audited_intervention_record
from mora3.research.gperturb_campaign_routes import GPerturbNativeCampaignAdapter
from mora3.research.primary_campaign_source_rebind import load_inactive_common_source_rebind
from mora3.research.primary_native_routes import finite_view_policy_sha256
from mora3.research.remaining_eight_registry import load_assets

_SCHEMA = 'Mora3.GPerturbPairedNativeOriginalCellsChunk/v1'
_SNAPSHOT = '79d328f542d1158581875ef325e56f213b407992f95db6ee0c7352a3459669d4'
_TOTAL = 20643
_MAX_CHUNK = 512
_EXPECTED_PREVIOUS = {
    'response': ('outputs/common-source-gperturb-full-response-20261111/actual-original-common-response.json',
                 '92a840288fd0bb8b888202eabd5d234fca55aba3adc228e296dcdaaec3d28b09'),
    'intervention': ('outputs/common-source-gperturb-full-intervention-20261111/actual-original-common-intervention.json',
                     'fa57f3e1e550a62fd5f6d5e1aba44e719621b4277badb76a0aad463ca3ad8f9e'),
    'witness': ('outputs/common-source-gperturb-paired-20261111/actual-original-two-direction-cell-pairs.json',
                '2af78b814b4f0d847ced638bc13b64fa6db5513d6d0fda1d76af95d7b5d9f94b'),
}


def checked_native_pair(index, response, inverse, response_ordinal, inverse_ordinal):
    """Validate literal source cell, both opposite masks and original row order."""
    if (type(index) is not int or index < 0
            or type(response_ordinal) is not int or response_ordinal < 0
            or type(inverse_ordinal) is not int or inverse_ordinal <= response_ordinal):
        raise ValueError('Original source index and distinct direction-ordered ordinals required')
    r = audited_response_record(response)
    i = audited_intervention_record(inverse)
    if (r['source_index'] != index or i['source_index'] != index
            or response.get('original_source_revision') != inverse.get('original_source_revision')
            or not isinstance(response.get('original_source_revision'), str)
            or not response['original_source_revision']
            or r['native_payload_sha256'] == i['native_payload_sha256']):
        raise ValueError('Source index, revision or distinct reverse task payload changed')
    source = require_paired_source_semantics(response, inverse)
    if (source['source_index'] != index or
            source['expression_sha256'] != r['expression_sha256'] or
            source['axis_sha256'] != i['axis_sha256']):
        raise ValueError('Exact response/inverse original expression and axes disagree')
    return dict(
        source_index=index, cell_id_hash=source['cell_id_hash'],
        original_source_revision=response['original_source_revision'],
        original_group=source['group'], source_shard_sha256=r['source_shard_sha256'],
        expression_sha256=source['expression_sha256'], axis_sha256=source['axis_sha256'],
        intervention=source['intervention'],
        response_native_payload_sha256=r['native_payload_sha256'],
        intervention_native_payload_sha256=i['native_payload_sha256'],
        response_ordinal=response_ordinal, intervention_ordinal=inverse_ordinal,
        native_target_positions={'response':5000,'intervention':2},
        original_observational_only=True,
    )


def _pinned_json(worktree, record):
    path, digest = record
    data = (worktree/path).read_bytes()
    if sha256(data).hexdigest() != digest:
        raise ValueError('Historical original source evidence changed: '+path)
    return json.loads(data)


def verified_previous_input(raw, expected_sha):
    """The caller must supply the previously preserved raw-byte evidence pin."""
    if (type(raw) is not bytes or type(expected_sha) is not str
            or len(expected_sha) != 64 or
            sha256(raw).hexdigest() != expected_sha):
        raise ValueError('Previously pinned original evidence bytes differ')
    previous = json.loads(raw)
    if type(previous) is not dict:
        raise ValueError('Native source continuation must be a document')
    return previous


def _resume_cursor(encoded, expected_sha):
    if type(encoded) is not str or type(expected_sha) is not str:
        raise ValueError('Native cursor checkpoint and digest required')
    raw = b64decode(encoded, validate=True)
    if sha256(raw).hexdigest() != expected_sha:
        raise ValueError('Native terminal cursor recovery digest changed')
    return raw


def audit_next_original_pairs(root, worktree, *, limit=_MAX_CHUNK,
                              previous=None, previous_sha256=None):
    """Verify <=512 *new* exact native same-cell pairs; save cursor for next chunk.

    All previous results remain immutable, and no earlier cell is replayed.
    The caller separately writes an exclusive JSON file and hashes its contents.
    """
    root, worktree = Path(root).resolve(), Path(worktree).resolve()
    if (root == worktree or Path.cwd().resolve() != root or not worktree.is_dir()
            or type(limit) is not int or not 1 <= limit <= _MAX_CHUNK):
        raise ValueError('Exact original checkout and bounded new source cells required')
    old = {key:_pinned_json(worktree,value) for key,value in _EXPECTED_PREVIOUS.items()}
    if any(old[d].get('source_cells') != _TOTAL
           or old[d].get('source_snapshot_sha256') != _SNAPSHOT
           or old[d].get('source_view_policy_sha256') != _POLICY
           or old[d].get('terminal_cursor_exhausted') is not True
           or old[d].get('training_admission') is not False
           for d in ('response','intervention')):
        raise ValueError('Two real full original directional authority records required')
    if old['response']['intervention_categories_original'] != old['intervention']['intervention_categories_original']:
        raise ValueError('Historical source population categories differ')
    if (old['witness'].get('source_cells') != 2
            or old['witness'].get('common_source_snapshot_sha256') != _SNAPSHOT
            or old['witness'].get('training_admission') is not False):
        raise ValueError('Exact original dual-direction two-cell witness required')
    context = load_inactive_common_source_rebind(root)
    view = load_assets(root)[0]
    if context.common_snapshot.sha256 != _SNAPSHOT or finite_view_policy_sha256(view) != _POLICY:
        raise ValueError('Exact qualified common29 source view or snapshot changed')
    routes = {}
    for direction in ('response','intervention'):
        adapter = GPerturbNativeCampaignAdapter(view, _SNAPSHOT, direction, 1)
        route = adapter.train_binding().route
        if route.task_key != 'GPerturb-sciPlex2:'+direction or route._batch_size != 1:
            raise ValueError('Exact one-row original train source route required')
        routes[direction] = (adapter,route)
    if previous is None:
        start = 0
        cursors = {d:routes[d][1].initial_cursor for d in routes}
        previous_hash = None
        previous_path = None
        previous_categories = {}
    else:
        if (type(previous) is not dict or previous.get('schema') != _SCHEMA
                or previous.get('source_snapshot_sha256') != _SNAPSHOT
                or previous.get('source_policy_sha256') != _POLICY
                or previous.get('total_original_cells') != _TOTAL
                or previous.get('terminal_exhausted') is True):
            raise ValueError('Only exact ongoing original-source chunk may be continued')
        start = previous['end_exclusive']
        if (type(start) is not int or start <= 0 or start >= _TOTAL
                or previous.get('start_inclusive',-1) >= start
                or previous.get('pairs_verified') != start-previous['start_inclusive']):
            raise ValueError('Monotonic source-prefix count or previous segment invalid')
        cursors = {d:_resume_cursor(previous['native_terminal_cursors_base64'][d],
                                     previous['native_terminal_cursor_sha256'][d]) for d in routes}
        for direction in routes:
            expected_pos = previous['native_terminal_positions'][direction]
            if list(routes[direction][1]._cursor_position(cursors[direction])) != expected_pos:
                raise ValueError('Native cursor/resume position does not match verified prior prefix')
        if (type(previous_sha256) is not str or len(previous_sha256) != 64
                or any(c not in '0123456789abcdef' for c in previous_sha256)):
            raise ValueError('External immutable original predecessor byte pin required')
        previous_hash = previous_sha256
        previous_path = previous.get('output_relative_path')
        previous_categories = previous['cumulative_intervention_categories']
        if sum(previous_categories.values()) != start:
            raise ValueError('Prior source intervention-category population count changed')

    count_to_read = min(limit,_TOTAL-start)
    categories = Counter()
    commitment = sha256()
    first = []
    for d, (adapter,route) in routes.items():
        if start == 0 and cursors[d] != route.initial_cursor:
            raise ValueError('First source cursor not native initial cursor')
    iters = {d:iter(routes[d][1].iter_from(cursors[d])) for d in routes}
    try:
        for offset in range(count_to_read):
            index = start+offset
            windows = {}
            signatures = {}
            for direction in ('response','intervention'):
                window = next(iters[direction], None)
                if window is None or window.original_source_rows != 1:
                    raise ValueError('Original paired source exhausted early or changed batch size')
                adapter, route = routes[direction]
                before,after = route._cursor_position(cursors[direction]),route._cursor_position(window.cursor_after)
                if after <= before:
                    raise ValueError('Native source cursor did not advance')
                signatures[direction] = _direction_signature(adapter,window)
                windows[direction] = window
                cursors[direction] = window.cursor_after
            r, i = windows['response'], windows['intervention']
            if r.batch.source_shard != i.batch.source_shard:
                raise ValueError('Reverse task records do not share the original source shard')
            proof = checked_native_pair(index,signatures['response'],signatures['intervention'],
                                        r.batch.source_row_ordinals[0],i.batch.source_row_ordinals[0])
            if offset < 2 and start == 0:
                witness = old['witness']['paired_rows'][offset]
                if (proof['cell_id_hash'] != witness['cell_id_hash']
                        or proof['expression_sha256'] != witness['response_expression_sha256']
                        or proof['intervention_native_payload_sha256'] != witness['intervention_native_payload_sha256']
                        or proof['response_native_payload_sha256'] != witness['response_native_payload_sha256']
                        or sha256(r.cursor_after).hexdigest()!=witness['common_response_cursor_sha256']
                        or sha256(i.cursor_after).hexdigest()!=witness['common_intervention_cursor_sha256']):
                    raise ValueError('Exact historical original two-cell native witness changed')
            if offset in (0, count_to_read-1):
                first.append(dict(index=index,cell_id_hash=proof['cell_id_hash'],
                                  response_payload_sha256=proof['response_native_payload_sha256'],
                                  intervention_payload_sha256=proof['intervention_native_payload_sha256']))
            categories[proof['intervention']['identity']] += 1
            commitment.update(canonical(proof)+b'\n')
        terminal_exhausted = start+count_to_read == _TOTAL
        if terminal_exhausted:
            for direction in ('response','intervention'):
                if next(iters[direction],None) is not None:
                    raise ValueError('Original directional source contains extra unqualified cells')
    finally:
        for iterator in iters.values():
            close = getattr(iterator,'close',None)
            if close is not None:
                close()
    cumulative = Counter(previous_categories)
    cumulative.update(categories)
    if sum(cumulative.values()) != start+count_to_read:
        raise ValueError('Cumulative source cell population mismatch')
    if terminal_exhausted and dict(sorted(cumulative.items())) != old['response']['intervention_categories_original']:
        raise ValueError('Full cell-pair categories differ from the two completed source executions')
    return dict(
        schema=_SCHEMA, source_snapshot_sha256=_SNAPSHOT,source_policy_sha256=_POLICY,
        total_original_cells=_TOTAL, start_inclusive=start,end_exclusive=start+count_to_read,
        pairs_verified=count_to_read,pair_native_target_positions={'response':5000,'intervention':2},
        native_terminal_cursors_base64={d:b64encode(cursors[d]).decode() for d in routes},
        native_terminal_cursor_sha256={d:sha256(cursors[d]).hexdigest() for d in routes},
        native_terminal_positions={d:list(routes[d][1]._cursor_position(cursors[d])) for d in routes},
        paired_original_source_segment_sha256=commitment.hexdigest(),
        segment_intervention_categories=dict(sorted(categories.items())),
        cumulative_intervention_categories=dict(sorted(cumulative.items())),
        previous_original_chunk_sha256=previous_hash,previous_original_chunk_path=previous_path,
        exact_native_endpoints=first,terminal_exhausted=terminal_exhausted,
        source_records_not_independent_experiments=True,global_heldout_split_certified=False,
        physical_concentration_calibrated=False,causal_effect_proven=False,
        learned_objectives_invoked=0,optimizer_updates=0,counted_campaign_runs=0,
        training_admission=False,
    )


if __name__ == '__main__':
    import sys
    if len(sys.argv) not in (4,6):
        raise SystemExit('Usage: python -m mora3.research.gperturb_full_common_pair_cursor ORIGINAL_ROOT WORKTREE OUTPUT.json [PREVIOUS.json PREVIOUS_RAW_SHA256]')
    root, worktree, output = sys.argv[1:4]
    previous = None
    previous_sha256 = None
    if len(sys.argv) == 6:
        prior_path = Path(sys.argv[4]).resolve()
        if not prior_path.is_relative_to(Path(worktree).resolve()):
            raise ValueError('Only local immutable continuation evidence under implementation worktree')
        previous_sha256 = sys.argv[5]
        previous = verified_previous_input(prior_path.read_bytes(), previous_sha256)
        if previous.get('output_relative_path') != str(prior_path.relative_to(Path(worktree).resolve())):
            raise ValueError('Prior native source continuation path changed')
    out = Path(output).resolve()
    if not out.is_relative_to(Path(worktree).resolve()) or out.exists():
        raise ValueError('Exclusive new source continuation output required')
    report = audit_next_original_pairs(root, worktree, previous=previous, previous_sha256=previous_sha256)
    report['output_relative_path'] = str(out.relative_to(Path(worktree).resolve()))
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:
        json.dump(report,f,sort_keys=True,indent=2)
        f.write('\n')
    print(json.dumps({k:report[k] for k in
                      ('start_inclusive','end_exclusive','pairs_verified','terminal_exhausted')},sort_keys=True))
