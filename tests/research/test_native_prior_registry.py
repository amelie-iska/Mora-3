from pathlib import Path
import pytest
from mora3.data.admet import canonical,digest
from mora3.data.finite_union_native_rows import FiniteUnionNativeView,full9
from mora3.model.annotation_vocabulary import AnnotationVocabulary
from mora3.research.native_prior_campaign import _SOURCE_ROUTES
from mora3.research.mup_campaign import executable_adapter_sha256
from mora3.research.native_prior_registry import PriorBudget,ROUTES,checked_budgets,_construct,build_native_prior_registry

def budgets():return {r:PriorBudget(1,4096,128,3) for r in ROUTES}

@pytest.mark.parametrize('mutation',['missing','unsupported','bool','untyped'])
def test_refuses_before_filesystem(mutation):
    b=budgets()
    if mutation=='missing':b.pop(ROUTES[0])
    if mutation=='unsupported':b['unsupported']=b.pop(ROUTES[0])
    if mutation=='untyped':b[ROUTES[0]]={}
    if mutation=='bool':
        with pytest.raises(ValueError):PriorBudget(True,1,1,1)
        return
    with pytest.raises(ValueError):
        build_native_prior_registry('/no-such-root',source_snapshot={},expected_view_policy_sha256='a'*64,budgets=b)

def fixture(tmp):
    ds=[]
    for route,(family,task,revision) in sorted(_SOURCE_ROUTES.items()):
        p=tmp/(route+'.source');p.write_bytes(b'test-only-payload-never-consumed')
        ds.append(dict(source_shard=dict(path=p.name,sha256=digest(p.read_bytes()),bytes=p.stat().st_size,rows=1),
            source_full9=full9(p),primary_retained_rows=1,
            source_task_schema_counts=[dict(source_family=family,task_name=task,native_schema='mora3.CorpusTask/v1',
                source_revision=revision,split='train',eligible_view_disposition=0,rows=1)]))
    raw=b''.join(canonical(d)+b'\n' for d in ds);(tmp/'descriptors').write_bytes(raw)
    m=canonical(dict(schema='Mora3.ImmutableFiniteEligibleNativeUnion/v1',full_project_training_ready=False,
        canonical_master_pointer_unchanged=True,source_taskviews=7,primary_retained_taskviews=7,
        original_immutable_source_views=dict(path='descriptors',bytes=len(raw),sha256=digest(raw))))
    (tmp/'manifest').write_bytes(m)
    return FiniteUnionNativeView(tmp/'manifest',digest(m),root=tmp)

def test_all_seven_real_adapter_factories_exact_cursor_and_code_hash(tmp_path):
    view=fixture(tmp_path)
    vocab=AnnotationVocabulary('{"learned_hex":[],"controls":["pad","node","edge"]}')
    root=Path.cwd()
    bindings,plans=_construct(root,view,vocab,'a'*64,checked_budgets(budgets()))
    assert len(bindings)==len(plans)==7
    for (key,binding),plan in zip(bindings,plans,strict=True):
        assert key==plan.task_key
        assert plan.initial_cursor_sha256==digest(binding.route.initial_cursor)
        assert plan.adapter_sha256==executable_adapter_sha256(binding.prepare,root=root)
        assert plan.max_windows==3
        assert binding.route.source_snapshot_sha256=='a'*64

def test_wrong_release_refuses_real_factory(tmp_path):
    view=fixture(tmp_path)
    view.descriptors[0]['source_task_schema_counts'][0]['source_revision']='stale'
    vocab=AnnotationVocabulary('{"learned_hex":[],"controls":["pad","node","edge"]}')
    with pytest.raises(ValueError):_construct(Path.cwd(),view,vocab,'a'*64,checked_budgets(budgets()))

def test_stale_snapshot_refused_before_source_reader(tmp_path):
    (tmp_path/'snapshot').write_bytes(b'changed')
    with pytest.raises(ValueError,match='digest'):
        build_native_prior_registry(tmp_path,source_snapshot=dict(path='snapshot',sha256='a'*64),
            expected_view_policy_sha256='b'*64,budgets=budgets())

def test_public_factory_uses_v2_loader_and_checks_policy(tmp_path,monkeypatch):
    import json
    from mora3.research import native_prior_registry as m
    view=fixture(tmp_path);view.effective_filter={'test_only':'fixture'}
    def put(name,raw):
        (tmp_path/name).write_bytes(raw);return (name,digest(raw))
    vocab=put('vocabulary',b'{"learned_hex":[],"controls":["pad","node","edge"]}')
    fit=put('fit',canonical(dict(status='complete-source-fit',train_views=602,target_fields_used=False,
        heldout_rows_used=False,payload_vocabulary_artifact=dict(sha256=vocab[1]))))
    vf=put('filter',canonical(view.effective_filter));snap=put('snapshot',b'test-only-snapshot')
    monkeypatch.setattr(m,'PINS',dict(payload_vocabulary=vocab,payload_fit=fit))
    monkeypatch.setattr(m,'CURRENT_VIEW_PINS',dict(effective_source_view=vf))
    calls=[]
    def loader(path,expected_sha256,*,root):
        calls.append((path,expected_sha256,root));return view
    monkeypatch.setattr(m,'load_study_filtered_native_view',loader)
    kwargs=dict(source_snapshot=dict(path=snap[0],sha256=snap[1]),budgets=budgets())
    with pytest.raises(ValueError,match='effective view'):
        m.build_native_prior_registry(tmp_path,expected_view_policy_sha256='b'*64,**kwargs)
    assert len(calls)==1
    actual_root=Path.cwd()
    monkeypatch.setattr(m,'executable_adapter_sha256',lambda method,root:executable_adapter_sha256(method,root=actual_root))
    result=m.build_native_prior_registry(tmp_path,expected_view_policy_sha256=m.finite_view_policy_sha256(view),**kwargs)
    assert len(result.plans)==7
    assert not result.complete_campaign and not result.allocation_authorized and not result.training_authorized
    (tmp_path/(ROUTES[0]+'.source')).write_bytes(b'stale')
    with pytest.raises(ValueError,match='fingerprint'):
        m.build_native_prior_registry(tmp_path,expected_view_policy_sha256=m.finite_view_policy_sha256(view),**kwargs)
