import hashlib
import json

import pytest

from mora3.research.mup_campaign import _prefetch_runtime_options
from mora3.research.native_cpu_prefetch import NativeRouteCPUStager
from mora3.research.primary_native_routes import NativeRouteBinding


def test_runtime_options_are_pinned_and_match_planned_routes(tmp_path):
    path = tmp_path/'runtime.json'
    def write(option):
        raw = json.dumps(option).encode()
        path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()
    assert _prefetch_runtime_options(path,write({'older_profile':True}),{'route':None}) is None
    value = {'native_cpu_prefetch':{'workers':4,'task_keys':['route']}}
    sha = write(value)
    assert _prefetch_runtime_options(path,sha,{'route':None})==(4,('route',))
    with pytest.raises(ValueError):
        _prefetch_runtime_options(path,sha,{'other':None})
    write({'native_cpu_prefetch':{'workers':5,'task_keys':['route']}})
    with pytest.raises(ValueError):
        _prefetch_runtime_options(path,sha,{'route':None})


def test_future_consumption_respects_exact_window_budget(monkeypatch):
    from mora3.research import native_cpu_prefetch
    binding = object.__new__(NativeRouteBinding)
    calls = []
    def select(_binding,cursor,*,max_windows):
        calls.append((cursor,max_windows))
        return max_windows
    monkeypatch.setattr(native_cpu_prefetch,'select_native_route_targets',select)
    with NativeRouteCPUStager(workers=1) as pool:
        assert pool.prefetch(binding,b'first',max_windows=3)
        assert not pool.prefetch(binding,b'first',max_windows=3)
        assert pool.select(binding,b'first',max_windows=2)==2
        assert pool.select(binding,b'first',max_windows=3)==3
    assert sorted(calls)==[(b'first',2),(b'first',3)]
