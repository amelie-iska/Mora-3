"""Read-only all-task native cursor verification before any admitted source update.

These are test-only tiny source fixtures: no optimizer, target or campaign slots.
"""
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from mora3.data.admet import canonical
from mora3.research.primary_native_routes import (
    NativeRouteBinding, RepeatedFiniteNativeTrainRoute,
    require_native_route_cursor_map,
)
from mora3.research.source_text_campaign import ReviewedTextCampaignRoute, TextBudget
from test_primary_native_routes import fixture, route


def _binding(r):
    return NativeRouteBinding(r, lambda _window: None, 'loss', 'target')


def test_all_task_preflight_refuses_wrong_nonselected_cursor_without_reading_targets(tmp_path):
    view = fixture(tmp_path)
    first = route(view, task_key='first')
    second = route(view, task_key='second')
    bindings = {'first': _binding(first), 'second': _binding(second)}
    cursors = {'first': first.initial_cursor, 'second': second.initial_cursor}
    require_native_route_cursor_map(bindings, cursors)
    # A forged checkpoint may put the first task's cursor under the second
    # task. The second task is not the current optimizer task, but the entire
    # cursor map must be checked before evaluating the first task.
    cursors['second'] = first.initial_cursor
    with pytest.raises(ValueError, match='cursor|route|snapshot'):
        require_native_route_cursor_map(bindings, cursors)
    cursors['second'] = second.initial_cursor + b' '
    with pytest.raises(ValueError, match='cursor|route|snapshot'):
        require_native_route_cursor_map(bindings, cursors)


def test_all_task_preflight_rejects_repeat_pass_and_changed_view(tmp_path):
    view = fixture(tmp_path)
    base = route(view)
    repeated = RepeatedFiniteNativeTrainRoute(base, 2)
    bindings = {'fit-objective': _binding(repeated)}
    cursors = {'fit-objective': repeated.initial_cursor}
    require_native_route_cursor_map(bindings, cursors)
    cursors['fit-objective'] = repeated._cursor(2, base.initial_cursor)
    with pytest.raises(ValueError, match='cursor|pass'):
        require_native_route_cursor_map(bindings, cursors)
    cursors['fit-objective'] = repeated.initial_cursor
    view.effective_filter = {'changed': True}
    with pytest.raises(ValueError, match='view|snapshot'):
        require_native_route_cursor_map(bindings, cursors)


def _reviewed_text_route():
    # This minimal test-only route has no source reader and no model objective.
    text = object.__new__(ReviewedTextCampaignRoute)
    text.view = SimpleNamespace(_require_unchanged=lambda: None)
    text.codec = SimpleNamespace(configuration=lambda: {'test': True})
    text.family = 'INV-081'
    text.source_snapshot_sha256 = 'a'*64
    text.budget = TextBudget(batch_size=2, max_batch_target_bytes=10,
        max_materialized_target_bytes=10, max_materialized_rows=4,
        max_passes=2, chunk_size=1, recompute=False)
    text._train = tuple(range(3))
    text._binding = canonical(dict(budget=asdict(text.budget),
        snapshot=text.source_snapshot_sha256, codec=text.codec.configuration(),
        family=text.family))
    text.native_view_policy_sha256 = 'b'*64
    text.task_key = 'reviewed-native-text:INV-081:source_language_prior'
    return text


def test_reviewed_text_cursor_preflight_accepts_real_boundaries_and_refuses_forgery():
    text = _reviewed_text_route()
    binding = NativeRouteBinding(text, text.prepare, text.loss_unit, text.target_unit)
    for cursor in (text.initial_cursor, text._cursor(0, 2), text._cursor(1, 0),
                   text._cursor(2, 0)):
        require_native_route_cursor_map({text.task_key: binding}, {text.task_key: cursor})
    for cursor in (text._cursor(0, 1), text._cursor(2, 2), b'not-json',
                   text.initial_cursor + b' ', text._cursor(3, 0)):
        with pytest.raises(ValueError, match='cursor|pass'):
            require_native_route_cursor_map({text.task_key: binding}, {text.task_key: cursor})


def test_route_map_never_treats_missing_or_nonbytes_state_as_resumable(tmp_path):
    view = fixture(tmp_path)
    r = route(view)
    bindings = {r.task_key: _binding(r)}
    with pytest.raises(ValueError, match='cursor|task'):
        require_native_route_cursor_map(bindings, {})
    with pytest.raises(ValueError, match='cursor|task'):
        require_native_route_cursor_map(bindings, {r.task_key: 'not-bytes'})


def test_mcsa_cursor_preflight_covers_unselected_reviewed_role_route():
    from mora3.research.mcsa_campaign_route import MCSACampaignTrainRoute, MCSAFinitePassPolicy
    # A source-free test-only instance uses the real original route's cursor
    # validator and native binding method; it never constructs a data batch.
    mcsa = object.__new__(MCSACampaignTrainRoute)
    mcsa._route_sha256 = 'c'*64
    mcsa._policy = MCSAFinitePassPolicy(2, 2, 29)
    mcsa._ordinals = (0,)
    mcsa._configuration = canonical({'source_snapshot_sha256': 'a'*64})
    binding = NativeRouteBinding(mcsa, mcsa.prepare, mcsa.loss_unit, mcsa.target_unit)
    for cursor in (mcsa.initial_cursor, mcsa._cursor(1, 0), mcsa._cursor(2, 0)):
        require_native_route_cursor_map({mcsa.task_key: binding}, {mcsa.task_key: cursor})
    for cursor in (mcsa._cursor(0, 1), mcsa._cursor(3, 0), mcsa.initial_cursor+b' '):
        with pytest.raises(ValueError, match='cursor|pass'):
            require_native_route_cursor_map({mcsa.task_key: binding}, {mcsa.task_key: cursor})
