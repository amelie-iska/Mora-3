"""Exact grouped metadata windows follow the native payload iterator."""
import json

import pytest

from mora3.research.finite_window_schedule import _grouped_native_windows
from test_primary_native_routes import fixture, route


@pytest.mark.parametrize('size',[2,3,4,8])
def test_grouped_native_counts_match_live_cursor_for_mixed_roles_tasks_and_shards(tmp_path,size):
    view=fixture(tmp_path)
    binding=route(view,batch_size=size)
    results=[
        _grouped_native_windows(binding,json.loads(view._sealed_by_path[path]),
                                root=tmp_path,max_metadata_bytes=100_000)
        for path in binding._paths
    ]
    actual=tuple(binding.iter_from(binding.initial_cursor))
    assert sum(x[0] for x in results)==len(actual)
    assert sum(x[1] for x in results)==sum(len(w.batch.source_row_ordinals) for w in actual)==4
    assert sum(x[2] for x in results)>0
    with pytest.raises(ValueError,match='budget'):
        _grouped_native_windows(binding,json.loads(view._sealed_by_path[binding._paths[0]]),
                                root=tmp_path,max_metadata_bytes=1)


def test_observed_target_replay_preserves_each_call_search_limit(tmp_path):
    from mora3.research.primary_native_routes import NativeRouteBinding, PreparedNativeTargets
    from mora3.research.runtime_capacity_gate import replay_observed_windows

    native=route(fixture(tmp_path),batch_size=2)
    def prepare(window):
        if window.batch.native_rows['task_id'][0].as_py()=='a0':
            return PreparedNativeTargets(0,'source missing',None)
        return PreparedNativeTargets(1,None,lambda _: pytest.fail('No model calls during preflight'))
    binding=NativeRouteBinding(native,prepare,'loss','native targets')
    refused=replay_observed_windows(binding,required_updates=2,per_update_scan_limit=1,remaining_budget=5)
    assert (refused['observed_updates'],refused['consumed_windows'],refused['reason'])==(0,1,'per-update-scan-limit')
    verified=replay_observed_windows(binding,required_updates=3,per_update_scan_limit=2,remaining_budget=5)
    assert (verified['observed_updates'],verified['consumed_windows'],verified['reason'])==(3,4,None)
    with pytest.raises(ValueError,match='budget'):
        replay_observed_windows(binding,required_updates=3,per_update_scan_limit=2,remaining_budget=3)
