"""Finite original train repetition preserves guarded roles and exact resume."""

from dataclasses import replace

import pytest

from mora3.data.admet import _json, canonical
from mora3.research.primary_native_routes import (
    NativeRouteBinding, PreparedNativeTargets, repeat_native_train_route, select_native_route_targets,
)
from test_primary_native_routes import fixture, ids, route


def test_explicit_repeat_exact_source_order_guard_and_resume(tmp_path):
    native = route(fixture(tmp_path))
    once = tuple(native.iter_from(native.initial_cursor))
    base = NativeRouteBinding(native,lambda _:PreparedNativeTargets(1,None,lambda _:None),'loss','target')
    repeated = repeat_native_train_route(base,max_passes=3)
    actual = tuple(repeated.route.iter_from(repeated.route.initial_cursor))
    assert [ids(w) for w in actual]==[ids(w) for w in once]*3
    assert all(w.batch.original_split=='train' for w in actual)
    assert len({w.evidence_sha256 for w in actual})==len(actual)==12
    assert _json(actual[3].cursor_after)['pass_index']==0
    assert _json(actual[4].cursor_after)['pass_index']==1
    for index in (0,3,4,7,11):
        assert tuple(repeated.route.iter_from(actual[index].cursor_after))==actual[index+1:]
    with pytest.raises(ValueError,match='cursor'):
        tuple(repeated.route.iter_from(native.initial_cursor))
    with pytest.raises(ValueError,match='cursor'):
        tuple(repeat_native_train_route(base,max_passes=2).route.iter_from(actual[0].cursor_after))
    selected = select_native_route_targets(repeated,actual[3].cursor_after,max_windows=1)
    assert selected.window==actual[4] and selected.status=='observed'


def test_finite_repetition_does_not_hide_unavailable_source(tmp_path):
    native = route(fixture(tmp_path))
    base = NativeRouteBinding(native,lambda _:PreparedNativeTargets(0,'native_missing',None),'loss','target')
    repeated = repeat_native_train_route(base,max_passes=2)
    found = select_native_route_targets(repeated,repeated.route.initial_cursor,max_windows=9)
    assert found.status=='source-exhausted' and len(found.unavailable)==8
    bounded = select_native_route_targets(repeated,repeated.route.initial_cursor,max_windows=3)
    assert bounded.status=='unavailable-scan-limit' and len(bounded.unavailable)==3
    for value in (1,True,0,2.0):
        with pytest.raises(ValueError):
            repeat_native_train_route(base,max_passes=value)
    with pytest.raises(ValueError):
        repeat_native_train_route(repeated,max_passes=2)
    wire = _json(repeated.route.initial_cursor)
    for drift in ({'pass_index':True},{'pass_index':2},{'inner_cursor_hex':'abc'},{'role':'validation'}):
        with pytest.raises(ValueError,match='cursor'):
            tuple(repeated.route.iter_from(canonical(wire|drift)))
