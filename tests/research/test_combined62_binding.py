from dataclasses import replace
from decimal import Decimal
import pytest
from test_combined60_binding import combined, finite_fixture
from mora3.research import combined62_binding as m,source_text_campaign as text,matched_validation as v

@pytest.fixture(scope='module')
def full(combined):
    root,prior,pin=combined
    budget=text.TextBudget(8,1048576,134217728,20000,1,128,False)
    routes=text.build_reviewed_text_routes(root,source_snapshot=prior['finite_inputs']['source_snapshot'],budgets={k:budget for k in text.SPECS})
    scan={r.task_key:1 for r in routes};plans=m.text_plans(root,routes=routes,scan_limits=scan)
    pending=m.legacy.prepare_three_evaluators(root,finite_inputs=prior['finite_inputs'],mcsa_bridge=prior['mcsa_bridge'],read_batch_size=8,max_selected_bytes=8*1024*1024)
    sources=v.TextCampaignValidationSources(pending.sources,tuple(sorted(routes,key=lambda r:r.task_key)),16*1024*1024)
    tasks=list(prior['campaign'].validation.tasks)
    for r in routes:
        coords=tuple(v.TextValidationSourceRow(**(c|{'native_group_key_digests':tuple(c['native_group_key_digests'])})) for c in r.validation_coordinates)
        tasks.append(v.MatchedValidationTask(r.task_key,r.family,text.SPECS[r.family][0],'Mora3.ReviewedTextRow/v1',coords,sum(c.observed_tokens_and_eos for c in coords),Decimal('1'),Decimal('1'),r.loss_unit,r.target_unit))
    cohort=replace(prior['campaign'].validation,view_policy_sha256=v.matched_validation_source_policy_sha256(sources),tasks=tuple(sorted(tasks,key=lambda t:t.task_key)))
    hashes=tuple(sorted(pending.hashes+tuple((r.task_key,m.executable_adapter_sha256(r.evaluate_validation,root=root)) for r in routes)))
    campaign=replace(prior['campaign'],validation=cohort,validation_adapter_sha256=hashes,train_routes=tuple(sorted(prior['campaign'].train_routes+plans,key=lambda p:p.task_key)),task_cycle=prior['campaign'].task_cycle+tuple(r.task_key for r in routes))
    args=dict(combined60_inputs=prior,text_routes=routes,text_scan_limits=scan,max_selected_text_bytes=16*1024*1024,campaign=campaign,campaign_reference=pin('TEST_ONLY_combined62.json',campaign.wire()))
    pin('TEST_ONLY_five_scope_v2.json',cohort.wire())
    return root,args,pin

def test_public_real_assets(full):
    root,args,pin=full;r=m.bind_combined62(root,**args)
    assert len(r.bindings)==62 and len(r.evaluators)==5
    assert sum(t.observed_targets for t in args['campaign'].validation.tasks)==5560367
    assert not r.training_authorized and not r.allocation_authorized and not r.complete_campaign
    prior = args['combined60_inputs']['campaign'].validation
    current = args['campaign'].validation
    for name, cohort in (('v1', prior), ('v2', current)):
        ref = pin(f'TEST_ONLY_cohort_roundtrip_{name}.json', cohort.wire())
        assert v.MatchedValidationCohort.load(root / ref['path'], ref['sha256']).wire() == cohort.wire()
    assert [p.max_windows for p in r.plans if p.task_key.startswith('reviewed-native-text:')]==[1,1]

@pytest.mark.parametrize('kind',['scan','prior_reference','cycle','coordinate','snapshot'])
def test_exact_refusals(full,kind):
    root,original,pin=full;args=dict(original);campaign=args['campaign']
    if kind=='scan':args['text_scan_limits']={}
    elif kind=='prior_reference':
        t=campaign.validation.tasks[0];tasks=tuple(replace(x,initial_reference_loss=Decimal('2')) if x is t else x for x in campaign.validation.tasks)
        campaign=replace(campaign,validation=replace(campaign.validation,tasks=tasks))
    elif kind=='cycle':campaign=replace(campaign,task_cycle=tuple(reversed(campaign.task_cycle)))
    elif kind=='snapshot':campaign=replace(campaign,source_role_schedule_sha256='a'*64)
    else:
        t=next(t for t in campaign.validation.tasks if isinstance(t.coordinates[0],v.TextValidationSourceRow));coords=(replace(t.coordinates[0],condition_sha256='a'*64),)+t.coordinates[1:]
        campaign=replace(campaign,validation=replace(campaign.validation,tasks=tuple(replace(x,coordinates=coords) if x is t else x for x in campaign.validation.tasks)))
    args.update(campaign=campaign,campaign_reference=pin('TEST_ONLY_bad62_'+kind+'.json',campaign.wire()))
    with pytest.raises(ValueError):m.bind_combined62(root,**args)

def test_window_evidence_refusal(full):
    r=full[1]['text_routes'][0];w=next(r.validation_batches())
    with pytest.raises(ValueError,match='evidence'):r.observed_targets(replace(w,evidence_sha256='a'*64))

def test_validation_authorization_control_flow():
    # Structural regression: no owner/model allocated or neural call made here.
    import ast,inspect,textwrap
    tree=ast.parse(textwrap.dedent(inspect.getsource(text.ReviewedTextCampaignRoute._objective)))
    gates=[n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=='not validation']
    assert len(gates)==1 and 'owner.require_training_admission()' in ast.unparse(gates[0])
    source=ast.unparse(tree)
    assert 'self.budget.batch_size > branch.max_batch_size' in source
    assert 'self.budget.max_batch_target_bytes > branch.max_total_target_bytes' in source
    assert 'owner.get_extra_state()' in source and 'self.observed_targets(window)' in source


def test_combined62_rejects_alternate_code_root():
    with pytest.raises(TypeError, match='text_code_root'):
        m.bind_combined62(
            '.', combined60_inputs={}, text_routes=(), text_scan_limits={},
            max_selected_text_bytes=1, campaign=None, campaign_reference=None,
            text_code_root='.',
        )
