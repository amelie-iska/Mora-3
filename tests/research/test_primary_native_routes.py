import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from mora3.data.admet import canonical, digest
from mora3.data.finite_union_native_rows import FiniteUnionNativeView, full9
from mora3.research.primary_native_routes import (
    FiniteNativeTrainRoute, NativeRouteBinding, NativeMetricIdentity, PreparedNativeTargets,
    finite_view_policy_sha256, run_native_primary_update, select_native_route_targets,
)
from test_full_owner import _construct, actual_core_assets, cases


def artifact(root, name, raw):
    (root/name).write_bytes(raw)
    return dict(path=name,bytes=len(raw),sha256=digest(raw))


def fixture(tmp_path):
    descriptors=[]
    for path, ids, tasks, codes in (
        ('A.parquet', ['a0','a1','a2','a3','a4','a5'], ['fit','fit','other','fit','fit','fit'], [0,5,0,1,0,2]),
        ('B.parquet', ['b0','b1','b2'], ['fit','fit','fit'], [5,0,0]),
    ):
        splits = [('train','validation','test')[c] if c<=2 else 'train' for c in codes]
        rows=pa.table(dict(task_id=ids,source_family=['native']*len(ids),task_name=tasks,
            native_schema=['native-schema/v1']*len(ids),source_revision=['revision-1']*len(ids),
            split=splits,native_payload_sha256=[digest(k.encode()) for k in ids],
            native_payload=[k.encode() for k in ids]))
        rows=rows.replace_schema_metadata({b'original':b'exact'})
        pq.write_table(rows,tmp_path/path,row_group_size=2)
        raw=(tmp_path/path).read_bytes()
        descriptor=dict(source_shard=dict(path=path,rows=len(ids),bytes=len(raw),sha256=digest(raw)),
            primary_retained_rows=sum(c<=2 for c in codes),source_full9=full9(tmp_path/path),
            original_group_sidecar=artifact(tmp_path,path+'.groups',b'authoritative'),
            disposition=dict(kind='original-row-ordinal-uint8',rows=len(ids),reference=artifact(tmp_path,path+'.mask',bytes(codes))),
            source_task_schema_counts=[dict(source_family='native',task_name=t,native_schema='native-schema/v1',
                source_revision='revision-1',split=s,eligible_view_disposition=c,rows=1)
                for t,s,c in zip(tasks,splits,codes,strict=True)])
        descriptors.append(descriptor)
    refs=artifact(tmp_path,'descriptors.jsonl',b''.join(canonical(d)+b'\n' for d in descriptors))
    manifest=artifact(tmp_path,'manifest.json',canonical(dict(schema='Mora3.ImmutableFiniteEligibleNativeUnion/v1',
        full_project_training_ready=False,canonical_master_pointer_unchanged=True,original_immutable_source_views=refs,
        source_taskviews=9,primary_retained_taskviews=7)))
    return FiniteUnionNativeView(tmp_path/manifest['path'],manifest['sha256'],root=tmp_path)


def route(view, paths=('A.parquet','B.parquet'), **options):
    config=dict(task_key='fit-objective',source_snapshot_sha256='a'*64,
        native_view_policy_sha256=finite_view_policy_sha256(view),source_family='native',task_name='fit',
        native_schema='native-schema/v1',source_revisions=('revision-1',),source_paths=paths,batch_size=2)
    return FiniteNativeTrainRoute(view,**(config|options))


def ids(window):
    return window.batch.native_rows['task_id'].to_pylist()


def test_cursor_keeps_original_roles_task_axis_mask_and_exact_resume(tmp_path):
    view=fixture(tmp_path);r=route(view)
    windows=tuple(r.iter_from(r.initial_cursor))
    assert [ids(w) for w in windows]==[['a0'],['a4'],['b1'],['b2']]
    assert [w.batch.source_row_ordinals for w in windows]==[(0,),(4,),(1,),(2,)]
    assert all(w.batch.original_split=='train' and w.batch.native_rows.schema.metadata=={b'original':b'exact'}
               for w in windows)
    first, second, third, fourth=windows
    resumed=tuple(r.iter_from(second.cursor_after))
    assert [(ids(w),w.cursor_after,w.evidence_sha256) for w in resumed]==[
        (ids(w),w.cursor_after,w.evidence_sha256) for w in (third,fourth)]
    assert not tuple(r.iter_from(fourth.cursor_after))
    assert len({w.evidence_sha256 for w in windows})==4
    for w in windows:view.require_native_rows(w.batch,split='train')
    with pytest.raises(ValueError):
        tuple(route(view,paths=('B.parquet','A.parquet')).iter_from(first.cursor_after))
    with pytest.raises(ValueError):
        tuple(r.iter_from(r.initial_cursor+b' '))
    with pytest.raises(ValueError):
        route(view,source_revisions=('wrong-release',))


def test_source_fingerprint_changes_reject_before_read(tmp_path):
    view=fixture(tmp_path);r=route(view)
    view.effective_filter={'changed':True}
    with pytest.raises(ValueError,match='snapshot|view'):
        next(r.iter_from(r.initial_cursor))
    with pytest.raises(ValueError,match='snapshot|view'):
        route(view,native_view_policy_sha256='f'*64)


def test_explicit_unavailable_events_do_not_execute_model_or_consume_cursor(tmp_path):
    r=route(fixture(tmp_path));calls=[]
    def prepare(window):
        calls.append(ids(window)[0])
        if len(calls)==1:return PreparedNativeTargets(0,'original_field_missing',None)
        def evaluate(_model):
            pytest.fail('pure source selection must not evaluate model')
        return PreparedNativeTargets(3,None,evaluate)
    binding=NativeRouteBinding(r,prepare,'native_units','target positions')
    found=select_native_route_targets(binding,r.initial_cursor,max_windows=2)
    assert found.status=='observed' and found.targets.observed_positions==3
    assert ids(found.window)==['a4'] and calls==['a0','a4']
    assert len(found.unavailable)==1
    assert found.unavailable[0].reason=='original_field_missing'
    assert found.unavailable[0].original_source_rows==1
    missing=NativeRouteBinding(r,lambda _:PreparedNativeTargets(0,'unknown_source_value',None),
                               'native_units','target positions')
    bounded=select_native_route_targets(missing,r.initial_cursor,max_windows=1)
    assert bounded.status=='unavailable-scan-limit' and bounded.window is None
    exhausted=select_native_route_targets(missing,r.initial_cursor,max_windows=10)
    assert exhausted.status=='source-exhausted' and len(exhausted.unavailable)==4
    assert tuple(r.iter_from(found.window.cursor_after))[0].batch.source_row_ordinals == (1,)
    for args in ((0,None,None),(True,'wrong',None),(2,'source_missing',lambda _:None),(2,None,None)):
        with pytest.raises(ValueError):
            PreparedNativeTargets(*args)


def test_real_source_bound_owner_gate_precedes_route_data_model_and_optimizer(tmp_path,actual_core_assets):
    from mora3.research.primary_training import NativeMuUpdateEngine
    model=_construct(tmp_path,actual_core_assets)
    engine=object.__new__(NativeMuUpdateEngine)
    engine.model=model
    # No readable route or optimizer field is even installed: current whole
    # source gate must fire before either is consulted.
    before=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match='admission'):
        run_native_primary_update(engine,bindings=None,
            metric_identity=NativeMetricIdentity('run','trial','segment',5),max_windows=1)
    assert torch.equal(before,torch.get_rng_state())


def test_metrics_use_native_observed_counts_and_explicit_missing_disposition(tmp_path):
    from mora3.research.metric_records import read_history, write_history
    from mora3.research.primary_native_routes import _metrics

    r=route(fixture(tmp_path));identity=NativeMetricIdentity('run','trial','seg',10)
    missing=NativeRouteBinding(r,lambda _:PreparedNativeTargets(0,'original_missing',None),'native log loss','native coordinates')
    missing_selection=select_native_route_targets(missing,r.initial_cursor,max_windows=1)
    before=_metrics(identity,missing,missing_selection,step=0,evidence='f'*64)
    assert {m.name:m.value for m in before}=={'unavailable_native_source_rows':1,'native_objective_unavailable':None}
    assert before[-1].missing_reason=='unavailable-scan-limit'
    valid=NativeRouteBinding(r,lambda _:PreparedNativeTargets(2,None,lambda model:None),
                             'native log loss','native coordinates')
    selected=select_native_route_targets(valid,r.initial_cursor,max_windows=1)
    after=_metrics(identity,valid,selected,step=1,evidence='e'*64,update={'loss':2.3})
    assert {m.name:m.value for m in after}==dict(unavailable_native_source_rows=0,native_objective=2.3,
                                               observed_native_targets=2,native_source_rows_for_update=1)
    path=tmp_path/'history.jsonl'
    assert write_history(path,(*before,*after))==6
    assert read_history(path)==(*before,*after)
