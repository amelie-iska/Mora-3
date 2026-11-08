from dataclasses import replace
from pathlib import Path
import pytest
from mora3.data.admet import digest
from mora3.research.mup_campaign import NativeTrainRoutePlan
from mora3.research.primary_native_routes import finite_view_policy_sha256
from mora3.research.finite_runtime_aggregator import SUPPORTED,_check_axes,_merge_exact,prior,reaction,assay

@pytest.mark.parametrize('kind',['missing','duplicate','unsupported','unsorted'])
def test_exact_scope_refuses(kind):
    keys=list(SUPPORTED)
    if kind=='missing':keys.pop()
    if kind=='duplicate':keys[-1]=keys[0]
    if kind=='unsupported':keys[-1]='unknown'
    if kind=='unsorted':keys.reverse()
    with pytest.raises(ValueError):_check_axes(tuple(NativeTrainRoutePlan(k,'a'*64,'b'*64,2) for k in keys))

@pytest.fixture(scope='module')
def constructed():
    root=Path.cwd();snapshot=digest(b'TEST ONLY scoped registry merge identity; not full-owner authority')
    view,sources,vocab,refs,raws=assay.load_original_assay_assets(root)
    ab=assay.checked_budgets({k:assay.AssayBudget(1,128,2) for k in assay.ROUTES})
    a=assay._construct(root,view,sources,vocab,snapshot,ab)
    pb=prior.checked_budgets({k:prior.PriorBudget(1,4096,128,2) for k in prior.ROUTES})
    p=prior._construct(root,view,vocab,snapshot,pb)
    source,codec,_,_=reaction.load_original_reactzyme_assets(root)
    binding,plan=reaction._construct(root,source,codec,snapshot,reaction.ReactZymeBudget(1,128,2))
    r=(((plan.task_key,binding),),(plan,))
    groups=(p,r,a);plans=tuple(sorted(p[1]+r[1]+a[1],key=lambda x:x.task_key))
    return root,snapshot,finite_view_policy_sha256(view),groups,plans

def test_real_factory_union_under_explicit_test_identity(constructed):
    root,snapshot,policy,groups,plans=constructed
    result=_merge_exact(root,plans,groups,snapshot=snapshot,view_policy=policy)
    assert tuple(k for k,_ in result)==SUPPORTED and len(result)==18

@pytest.mark.parametrize('kind',['snapshot','view','cursor','adapter','budget','duplicate','missing'])
def test_exact_binding_drift_refused(constructed,kind):
    root,snapshot,policy,groups,plans=constructed
    if kind=='snapshot':snapshot='b'*64
    if kind=='view':policy='c'*64
    if kind in ('cursor','adapter','budget'):
        changes={'cursor':dict(initial_cursor_sha256='d'*64),'adapter':dict(adapter_sha256='e'*64),'budget':dict(max_windows=3)}[kind]
        plans=(replace(plans[0],**changes),)+plans[1:]
    if kind=='duplicate':groups=groups+(groups[0],)
    if kind=='missing':groups=groups[:-1]
    with pytest.raises(ValueError):_merge_exact(root,plans,groups,snapshot=snapshot,view_policy=policy)

@pytest.mark.parametrize('kind',['view','budget','snapshot'])
def test_public_preflight_refuses_before_factory_calls(constructed,monkeypatch,kind):
    from decimal import Decimal
    from mora3.research.matched_validation import MatchedValidationCohort,MatchedValidationTask,ValidationSourceRow
    from mora3.research.mup_campaign import FrozenPrimaryProxyCampaign
    from mora3.research.finite_runtime_aggregator import assemble_scoped_runtime_routes
    root,snapshot,policy,groups,plans=constructed
    task=MatchedValidationTask(SUPPORTED[0],'TEST_ONLY','TEST_ONLY','TEST_ONLY',
        (ValidationSourceRow('TEST_ONLY','a'*64,0,'TEST_ONLY','b'*64,'TEST_ONLY'),),1,Decimal('1'),Decimal('1'),'TEST_ONLY','TEST_ONLY')
    cohort=MatchedValidationCohort(snapshot,policy,'c'*64,'d'*64,1,1,1000,(task,),
        tuple((key,'TEST ONLY disposition') for key in SUPPORTED[1:]))
    campaign=FrozenPrimaryProxyCampaign(snapshot,'e'*64,36,9750884,'1'*64,'2'*64,'3'*64,'4'*64,
        cohort,((SUPPORTED[0],'5'*64),),plans,SUPPORTED,1.,('TEST_ONLY.weight',),Decimal('2'),Decimal('20'))
    pb={k:prior.PriorBudget(1,4096,128,2) for k in prior.ROUTES}
    ab={k:assay.AssayBudget(1,128,2) for k in assay.ROUTES}
    if kind=='budget':pb[prior.ROUTES[0]]=prior.PriorBudget(1,4096,128,3)
    def forbidden(*a,**kw):pytest.fail('factory called before invalid preflight refused')
    monkeypatch.setattr(prior,'build_native_prior_registry',forbidden)
    with pytest.raises(ValueError):
        assemble_scoped_runtime_routes('/missing',campaign=campaign,
            campaign_reference=dict(path='campaign.json',sha256=campaign.sha256),
            source_snapshot=dict(path='snapshot.json',sha256='f'*64 if kind=='snapshot' else snapshot),
            expected_view_policy_sha256='f'*64 if kind=='view' else policy,
            prior_budgets=pb,reactzyme_budget=reaction.ReactZymeBudget(1,128,2),assay_budgets=ab)
