"""Reviewed M-CSA source cursor checks without optimization or a model."""

import pytest

from mora3.data.admet import _json, canonical, digest
from mora3.research.mcsa_campaign_route import MCSACampaignTrainRoute, MCSAFinitePassPolicy
import test_mcsa_role_list_admission as admission_fixture


@pytest.fixture
def review_fixture():
    source = admission_fixture.AdmissionTests()
    source.setUp()
    try:
        path = source.root/'review.json'
        reference = admission_fixture.write(path,source.packet)
        yield source,path,reference['sha256']
    finally:
        source.doCleanups()


def route(review_fixture, *, passes=3, seed=29, batch_size=2, payload_max_tokens=8192):
    source,path,sha = review_fixture
    return MCSACampaignTrainRoute(
        path,sha,vocabulary=source.vocab,source_snapshot_sha256='a'*64,
        pass_policy=MCSAFinitePassPolicy(passes,batch_size,seed),reviewed_payload_max_tokens=payload_max_tokens,
    )


def test_reviewed_roles_and_finite_exact_pass_recovery(review_fixture):
    source,_,_ = review_fixture
    run = route(review_fixture)
    windows = tuple(run.iter_from(run.initial_cursor))
    assert run.records_per_pass==1 and run.symbols_per_pass==3 and run.maximum_updates==3
    assert len(windows)==3 and len({window.evidence_sha256 for window in windows})==3
    assert [window.pass_index for window in windows]==[0,1,2]
    assert all(window.original_candidate_row_ordinals==(0,) for window in windows)
    assert all(window.batch.audit[0]['proposed_split']=='train' and not window.batch.task_admitted for window in windows)
    assert all(window.batch.canonical_sequences==('ACD',) and window.observed_targets==3 for window in windows)
    assert tuple(run.iter_from(windows[0].cursor_after))==windows[1:]
    assert tuple(run.iter_from(windows[-1].cursor_after))==()
    run.admission.require_batch(windows[-1].batch,model_binding=source.binding,split='train')
    with pytest.raises(ValueError):
        run.admission.require_batch(windows[0].batch,model_binding=dict(source.binding,payload_max_tokens=37000),split='train')


def test_cursor_refuses_route_policy_and_position_drift(review_fixture):
    first = route(review_fixture)
    cursor = next(first.iter_from(first.initial_cursor)).cursor_after
    for options in ({'passes':2},{'seed':30},{'batch_size':1}):
        with pytest.raises(ValueError,match='cursor'):
            tuple(route(review_fixture,**options).iter_from(cursor))
    wire = _json(cursor)
    for changes in ({'next_offset':1},{'pass_index':4},{'role':'validation'},{'next_offset':True}):
        with pytest.raises(ValueError,match='cursor'):
            tuple(first.iter_from(canonical(wire|changes)))
    with pytest.raises(ValueError,match='cursor'):
        tuple(first.iter_from(cursor+b' '))


def test_hash_order_keeps_all_source_ordinals_and_is_deterministic():
    policy = MCSAFinitePassPolicy(4,2,29)
    identities = (0,3,4,9,12,44,50,62)
    orders = [policy.order(identities,n) for n in range(4)]
    assert all(sorted(order)==list(identities) for order in orders)
    assert orders==[policy.order(identities,n) for n in range(4)]
    assert len(set(orders))==4
    assert policy.order(identities,0)!=MCSAFinitePassPolicy(4,2,30).order(identities,0)
    for bad in ((True,2,29),(1,0,29),(2,2,-1),(2,2,True)):
        with pytest.raises(ValueError):
            MCSAFinitePassPolicy(*bad)
    with pytest.raises(ValueError):
        policy.order(identities,4)


def test_immutable_review_refuses_source_drift_after_construction(review_fixture):
    source,_,_ = review_fixture
    run = route(review_fixture)
    candidate = source.root/'candidates.jsonl'
    candidate.write_bytes(candidate.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='identity changed'):
        tuple(run.iter_from(run.initial_cursor))


def test_distinct_explicit_review_capacities_preserve_original_roles(review_fixture):
    source,path,_ = review_fixture
    narrow = route(review_fixture)
    narrow_window = next(narrow.iter_from(narrow.initial_cursor))
    with pytest.raises(ValueError,match='literal reviewed'):
        route(review_fixture,payload_max_tokens=37000)

    binding = dict(source.binding,payload_max_tokens=37000)
    coordinator = dict(source.packet['coordinator_review'],approved_model_binding_sha256=digest(canonical(binding)))
    wide_packet = dict(source.packet,approved_model_binding=binding,coordinator_review=coordinator)
    wide_path = path.with_name('wide-review.json')
    wide_reference = admission_fixture.write(wide_path,wide_packet)
    wide_fixture = source,wide_path,wide_reference['sha256']
    with pytest.raises(ValueError,match='literal reviewed'):
        route(wide_fixture)
    wide = route(wide_fixture,payload_max_tokens=37000)
    wide_window = next(wide.iter_from(wide.initial_cursor))
    assert narrow.binding()['reviewed_payload_max_tokens']==8192
    assert wide.binding()['reviewed_payload_max_tokens']==37000
    assert narrow_window.batch==wide_window.batch
    assert narrow_window.cursor_after!=wide_window.cursor_after
    with pytest.raises(ValueError,match='cursor'):
        tuple(wide.iter_from(narrow_window.cursor_after))
    with pytest.raises(ValueError,match='model task semantics'):
        wide.admission.require_batch(wide_window.batch,model_binding=source.binding,split='train')
