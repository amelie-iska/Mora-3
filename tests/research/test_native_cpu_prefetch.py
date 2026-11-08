from threading import Event

import pytest

import mora3.research.native_cpu_prefetch as staging
from mora3.research.primary_native_routes import (
    NativeRouteBinding, PreparedNativeTargets, select_native_route_targets,
)
import pyarrow as pa
import pyarrow.parquet as pq
from mora3.data.admet import canonical, digest
from mora3.data.finite_union_native_rows import FiniteUnionNativeView, full9
from mora3.research.primary_native_routes import FiniteNativeTrainRoute, finite_view_policy_sha256

def fixture(tmp_path):
    path = 'source.parquet'
    rows = pa.table(dict(task_id=['t0','t1'], source_family=['native']*2,
        task_name=['fit']*2, native_schema=['schema/v1']*2, source_revision=['rev1']*2,
        split=['train']*2, native_payload_sha256=[digest(b'0'),digest(b'1')],
        native_payload=[b'0',b'1']))
    pq.write_table(rows,tmp_path/path)
    raw=(tmp_path/path).read_bytes()
    def artifact(name, data):
        (tmp_path/name).write_bytes(data)
        return dict(path=name,bytes=len(data),sha256=digest(data))
    descriptor=dict(source_shard=dict(path=path,rows=2,bytes=len(raw),sha256=digest(raw)),
        primary_retained_rows=2,source_full9=full9(tmp_path/path),
        original_group_sidecar=artifact('groups',b'groups'),
        disposition=dict(kind='original-row-ordinal-uint8',rows=2,reference=artifact('mask',b'\x00\x00')),
        source_task_schema_counts=[dict(source_family='native',task_name='fit',native_schema='schema/v1',
            source_revision='rev1',split='train',eligible_view_disposition=0,rows=2)])
    refs=artifact('descriptors.jsonl',canonical(descriptor)+b'\n')
    manifest=artifact('manifest.json',canonical(dict(schema='Mora3.ImmutableFiniteEligibleNativeUnion/v1',
        full_project_training_ready=False,canonical_master_pointer_unchanged=True,
        original_immutable_source_views=refs,source_taskviews=2,primary_retained_taskviews=2)))
    return FiniteUnionNativeView(tmp_path/'manifest.json',manifest['sha256'],root=tmp_path)

def route(view):
    return FiniteNativeTrainRoute(view,task_key='task',source_snapshot_sha256='a'*64,
        native_view_policy_sha256=finite_view_policy_sha256(view),source_family='native',
        task_name='fit',native_schema='schema/v1',source_revisions=('rev1',),
        source_paths=('source.parquet',),batch_size=2)

def ids(window): return window.batch.native_rows['task_id'].to_pylist()


def binding_for(r):
    return NativeRouteBinding(
        r, lambda _: PreparedNativeTargets(0, 'bounded_fixture', None),
        'native units', 'target positions',
    )


def test_admission_gates_source_work_and_consumes_exact_existing_selection(tmp_path, monkeypatch):
    r = route(fixture(tmp_path)); binding = binding_for(r); calls = []
    original = staging.select_native_route_targets
    def observed(*args, **kwargs):
        calls.append((args[0], args[1]))
        return original(*args, **kwargs)
    monkeypatch.setattr(staging, 'select_native_route_targets', observed)
    with staging.NativeRouteCPUStager(workers=1) as stager:
        assert calls == []
        ticket = stager.admit(binding, r.initial_cursor, max_windows=2)
        expected = select_native_route_targets(binding, r.initial_cursor, max_windows=2)
        got = stager.consume(ticket, binding, r.initial_cursor)
        assert got == expected
        assert got.status == 'source-exhausted'
        assert len(calls) == 1
        assert calls[0] == (binding, r.initial_cursor)


def test_literal_binding_cursor_freshness_and_discard_cleanup(tmp_path, monkeypatch):
    r = route(fixture(tmp_path)); binding = binding_for(r); entered = Event(); release = Event()
    def blocked(bound, cursor, *, max_windows):
        entered.set(); assert release.wait(5)
        return select_native_route_targets(bound, cursor, max_windows=max_windows)
    monkeypatch.setattr(staging, 'select_native_route_targets', blocked)
    stager = staging.NativeRouteCPUStager(workers=1)
    try:
        ticket = stager.admit(binding, r.initial_cursor, max_windows=1)
        assert entered.wait(5)
        with pytest.raises(ValueError, match='stale'):
            stager.consume(ticket, binding, r.initial_cursor + b' ')
        assert stager.discard(ticket)
        release.set()
        assert not stager.discard(ticket)
    finally:
        release.set(); stager.close()

def test_four_outstanding_limit_and_duplicate_cursor_guard(tmp_path, monkeypatch):
    r = route(fixture(tmp_path)); binding = binding_for(r); entered = Event(); release = Event()
    def blocked(*args, **kwargs):
        entered.set(); assert release.wait(5)
        return select_native_route_targets(*args, **kwargs)
    monkeypatch.setattr(staging, 'select_native_route_targets', blocked)
    stager = staging.NativeRouteCPUStager(workers=4)
    try:
        tickets = [stager.admit(binding, r.initial_cursor + str(i).encode(), max_windows=1) for i in range(3)]
        with pytest.raises(ValueError, match='already outstanding'):
            stager.admit(binding, r.initial_cursor + b'0', max_windows=1)
        tickets.append(stager.admit(binding, r.initial_cursor + b'3', max_windows=1))
        assert entered.wait(5)
        with pytest.raises(RuntimeError, match='limit'):
            stager.admit(binding, b'another-cursor', max_windows=1)
    finally:
        release.set(); stager.close()

def test_selector_exception_is_preserved_when_consumed(tmp_path, monkeypatch):
    r = route(fixture(tmp_path)); binding = binding_for(r); error = ValueError('route snapshot changed')
    def fail(*args, **kwargs): raise error
    monkeypatch.setattr(staging, 'select_native_route_targets', fail)
    with staging.NativeRouteCPUStager(workers=1) as stager:
        ticket = stager.admit(binding, r.initial_cursor, max_windows=1)
        with pytest.raises(ValueError, match='route snapshot changed') as caught:
            stager.consume(ticket, binding, r.initial_cursor)
        assert caught.value is error

def test_workers_and_closed_stager_are_bounded(tmp_path):
    for workers in (0, 5, True):
        with pytest.raises(ValueError): staging.NativeRouteCPUStager(workers=workers)
    r = route(fixture(tmp_path)); binding = binding_for(r)
    stager = staging.NativeRouteCPUStager(workers=1); stager.close()
    with pytest.raises(RuntimeError, match='closed'):
        stager.admit(binding, r.initial_cursor, max_windows=1)
