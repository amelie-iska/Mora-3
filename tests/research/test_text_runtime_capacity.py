from dataclasses import replace
import json,pytest
from test_combined62_binding import combined, finite_fixture, full
from mora3.research import finite_window_schedule as capacity,runtime_capacity_gate as gate,source_text_campaign as text
from mora3.research.combined62_binding import text_plans
from mora3.data.admet import canonical

@pytest.fixture(scope='module')
def actual(full):
    root,args,pin=full
    sr=args['combined60_inputs']['finite_inputs']['source_snapshot']
    routes=text.build_reviewed_text_routes(root,source_snapshot=sr,budgets={k:text.TextBudget(8,1048576,134217728,20000,2,128,False) for k in text.SPECS})
    plans=text_plans(root,routes=routes,scan_limits={r.task_key:1 for r in routes})
    bindings=tuple(sorted((r.task_key,r.train_binding()) for r in routes))
    return root,args,sr,plans,bindings,pin

def test_exact_lifetime_and_bounded_observed_replay(actual):
    root,args,sr,plans,bindings,pin=actual
    counts=capacity.exact_capacities(root,bindings=bindings,plans=plans,source_snapshot=sr,max_identity_metadata_bytes=8*1024*1024)
    assert [(c.original_windows,c.source_window_upper_bound,c.per_update_scan_limit) for c in counts]==[(576,1152,1),(1674,3348,1)]
    rows=[]
    for key,b in bindings:
        before=b.route.initial_cursor
        proof=gate.replay_observed_windows(b,required_updates=3,per_update_scan_limit=1,remaining_budget=3)
        assert proof['observed_updates']==proof['consumed_windows']==3 and proof['reason'] is None
        assert b.route.initial_cursor==before
        last=((len(b.route._train)-1)//8)*8
        window=next(b.route.iter_from(b.route._cursor(1,last)))
        assert len(window.rows)==len(b.route._train)-last
        assert json.loads(window.cursor_after)['pass_index']==2
        assert not tuple(b.route.iter_from(window.cursor_after))
        with pytest.raises(ValueError,match='budget'):
            gate.replay_observed_windows(b,required_updates=3,per_update_scan_limit=1,remaining_budget=2)
        rows.append(dict(task_key=key,**proof))
    pin('TEST_ONLY_text_capacity_replay.json',canonical(dict(capacities=[c.__dict__ for c in counts],bounded_observed_replay=rows,model_constructed=False,training_authorized=False)))

def test_duplicate_and_metadata_bound_refused(actual):
    root,args,sr,plans,bindings,pin=actual
    with pytest.raises(ValueError,match='metadata budget'):
        capacity.exact_capacities(root,bindings=bindings,plans=plans,source_snapshot=sr,max_identity_metadata_bytes=1)
    route=bindings[0][1].route;original=route._train
    try:
        route._train=(original[0],)*len(original)
        with pytest.raises(ValueError,match='Duplicate'):
            capacity.exact_capacities(root,bindings=bindings,plans=plans,source_snapshot=sr,max_identity_metadata_bytes=8*1024*1024)
    finally:route._train=original

def test_runtime_gate_refuses_unavailable_passes_before_replay(actual,monkeypatch):
    root,args,sr,plans,bindings,pin=actual
    old=args['campaign'];keys={k for k,_ in bindings}
    cohort=replace(old.validation,tasks=tuple(t for t in old.validation.tasks if t.task_key in keys),train_only_dispositions=())
    campaign=replace(old,train_routes=plans,task_cycle=tuple(k for k,_ in bindings),validation=cohort,validation_adapter_sha256=tuple(x for x in old.validation_adapter_sha256 if x[0] in keys))
    cr=pin('TEST_ONLY_text_capacity_campaign.json',campaign.wire())
    request=gate.RuntimeCapacityRequest(sr,cr,8*1024*1024,5000)
    def forbidden(*a,**k):raise AssertionError('Metadata deficit must precede observed replay')
    monkeypatch.setattr(gate,'replay_observed_windows',forbidden)
    with pytest.raises(gate.WindowCapacityRefusal) as error:
        gate.require_runtime_window_capacity(root,campaign=campaign,bindings=dict(bindings),request=request)
    assert len(error.value.deficits)==1 and error.value.deficits[0]['available_windows']==1152
