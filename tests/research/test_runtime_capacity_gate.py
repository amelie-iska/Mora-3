"""Real pinned coordinates; ALL campaign/snapshot/baseline identities TEST ONLY."""
from pathlib import Path
from decimal import Decimal
from dataclasses import replace
import tempfile,json
import pytest
from mora3.research import finite59_assembly as m
from mora3.research.matched_validation import MatchedValidationCohort,MatchedValidationTask,ValidationSourceRow
from mora3.research.finite_view_compatibility import ENROLLMENT,SOURCE

@pytest.fixture(scope='module')
def finite_fixture():
    root=Path.cwd();out=Path(tempfile.mkdtemp(prefix='TEST_ONLY_NOT_AUTHORITY_',dir=root/'outputs/implementation/readiness-20261108-runtime-capacity-integration-v2'))
    def pin(name,raw):
        path=out/name;path.write_bytes(raw);return dict(path=str(path.relative_to(root)),sha256=m.digest(raw))
    sr=pin('TEST_ONLY_source_snapshot.json',m.canonical(dict(schema='TEST_ONLY_NOT_A_FULL_SOURCE_SNAPSHOT',training_authorized=False)))
    kw=dict(root=root,source_snapshot=sr,expected_view_policy_sha256=m.BASE_POLICY)
    p=m.prior.build_native_prior_registry(**kw,budgets={k:m.prior.PriorBudget(1,4096,128,2) for k in m.prior.ROUTES})
    r=m.reaction.build_original_reactzyme_registry(**kw,budget=m.reaction.ReactZymeBudget(1,128,2))
    a=m.assay.build_original_assay_registry(**kw,budgets={k:m.assay.AssayBudget(1,128,2) for k in m.assay.ROUTES})
    d=m.pdb.build_original_pdb_registry(**kw,budget=m.pdb.PDBRouteBudget(1,2,1,1,4000000))
    u=m.uniprot.build_full_record_registry(root,source_snapshot=sr,expected_view_policy_sha256=m.INVERSE_POLICY,budgets={k:m.uniprot.FullRecordBudget(1,128,2) for k in m.uniprot.ROUTES})
    n=m.annotation.build_original_annotation_registry(**kw,budgets={k:m.annotation.AnnotationRouteBudget(1,2) for k in m.annotation.ROUTES})
    g=m.rg.build_rfam_goa_registry(root,source_snapshot=sr,rfam_budgets={k:m.rg.RfamBudget(1,2) for k in m.rg.RFAM_KEYS},goa_budget=m.rg.GOABudget(2,1))
    o=m.rosetta.build_rosetta_registry(root,source_snapshot=sr,budgets={k:m.rosetta.RosettaBudget(2,128 if m.rosetta.SPECS[k][0]=='LINCS-Pilot1' else 1) for k in m.rosetta.KEYS})
    e=m.remaining.build_remaining_eight_registry(root,source_snapshot=sr,budgets={k:m.remaining.RouteBudget(1,2,128 if m.remaining.SPECS[k][0]=='UniProt-function-annotations' else 1) for k in m.remaining.KEYS})
    plans=tuple(sorted(p.plans+(r.plan,)+a.plans+(d.plan,)+u.plans+n.plans+g.plans+o.plans+e.plans,key=lambda x:x.task_key))
    proof=json.loads((root/ENROLLMENT[0]).read_bytes());coordinates=tuple(ValidationSourceRow(**r['coordinate']) for r in proof['inverse_validation_source_coordinates'])
    witness,=tuple(w for w in proof['native_route_binding_witnesses'] if w['direction']==m.INVERSE)
    task=MatchedValidationTask(m.INVERSE,SOURCE,m.INVERSE,'mora3.CorpusTask/v1',coordinates,5225,Decimal('1'),Decimal('1'),witness['loss_unit'],witness['target_unit'])
    fake=m.digest(b'TEST ONLY nonexistent authority/reference/resource evidence')
    cohort=MatchedValidationCohort(sr['sha256'],m.INVERSE_POLICY,fake,fake,1,1,10000000,(task,),tuple((k,'TEST ONLY disposition; not an accepted production role') for k in m.AXES if k!=m.INVERSE))
    campaign=m.FrozenPrimaryProxyCampaign(sr['sha256'],fake,2,1,fake,fake,fake,fake,cohort,((m.INVERSE,fake),),plans,m.AXES,1.0,(),Decimal('1'),Decimal('1'))
    cr=pin('TEST_ONLY_campaign.json',campaign.wire());pin('TEST_ONLY_cohort.json',cohort.wire())
    kwargs=dict(campaign=campaign,campaign_reference=cr,source_snapshot=sr,prior_result=p,reactzyme_result=r,assay_result=a,pdb_result=d,uniprot_result=u,annotation_result=n,rfam_goa_result=g,pdb_resource_binding_sha256=d.resource_binding_sha256,rfam_goa_budget_binding_sha256=g.budget_binding_sha256,rosetta_result=o,remaining_result=e)
    return root,kwargs,pin




from mora3.research import runtime_capacity_gate as gate
from mora3.research import mcsa_campaign_bridge as mcsa
from mora3.data.mcsa_role_list_admission import TASK as MCSA

def test_actual60_full_horizon_refuses_before_any_engine(finite_fixture):
    root,inputs,pin=finite_fixture;cycle=tuple(sorted(m.AXES+(MCSA,)))
    bridge=mcsa.build_mcsa_bridge(root,source_snapshot=inputs['source_snapshot'],pass_policy=mcsa.MCSAFinitePassPolicy(3,16,2026110201),max_windows=114,task_cycle=cycle,total_updates=5000,max_validation_bytes=8*1024*1024)
    bindings={}
    for name in ('prior_result','assay_result','uniprot_result','annotation_result','rfam_goa_result','rosetta_result','remaining_result'):bindings.update(inputs[name].bindings)
    for name in ('reactzyme_result','pdb_result'):
        r=inputs[name];bindings[r.plan.task_key]=r.binding
    bindings[MCSA]=bridge.binding
    cohort=replace(inputs['campaign'].validation,train_only_dispositions=tuple(sorted(inputs['campaign'].validation.train_only_dispositions+((MCSA,'TEST ONLY metadata role; no production authority'),))))
    campaign=replace(inputs['campaign'],train_routes=tuple(sorted(inputs['campaign'].train_routes+(bridge.plan,),key=lambda p:p.task_key)),task_cycle=cycle,validation=cohort)
    cr=pin('TEST_ONLY_immutable_campaign.json',campaign.wire());before=campaign.wire()
    request=gate.RuntimeCapacityRequest(inputs['source_snapshot'],cr,64*1024*1024,5000)
    with pytest.raises(gate.WindowCapacityRefusal) as error:gate.require_runtime_window_capacity(root,campaign=campaign,bindings=bindings,request=request)
    deficits={d['task_key']:d for d in error.value.deficits}
    assert deficits[m.pdb.TASK]['available_windows']==2
    assert all(d['required_updates']>d['available_windows'] for d in deficits.values())
    assert len(deficits)<59
    assert campaign.wire()==before
    pin('actual-capacity-refusal.json',m.canonical(dict(schema='Mora3.TestOnlyActual60FullHorizonCapacityRefusal/v1',horizon=5000,deficits=error.value.deficits,campaign_sha256=campaign.sha256,optimizer_created=False,reservation_created=False)))

def test_exact_cycle_multiplicity_and_tail():
    assert dict(gate.scheduled_calls(('a','b','a'),8))=={'a':5,'b':3}

@pytest.mark.parametrize('admission_refuses',[False,True])
def test_staged_worker_gate_precedes_all_mutations(tmp_path,monkeypatch,admission_refuses):
    from mora3.research import mup_campaign_worker as worker
    events=[]
    class Model:
        def require_training_admission(self,**scope):
            if not scope:events.append('admission')
            if admission_refuses:raise ValueError('owner refused')
    class Engine:model=Model()
    class Campaign:sha256='a'*64
    monkeypatch.setattr(worker,'NativeMuUpdateEngine',Engine);monkeypatch.setattr(worker,'SourceBoundOwner',Model);monkeypatch.setattr(worker,'FrozenPrimaryProxyCampaign',Campaign)
    def refuse(*a,**kw):events.append('capacity');raise gate.WindowCapacityRefusal(({'task_key':'TEST ONLY'},))
    monkeypatch.setattr(worker,'require_runtime_window_capacity',refuse)
    def forbidden(*a,**kw):raise AssertionError('Mutation/runtime execution reached')
    for name in ('reserve_next_attempt','run_primary_proxy_trial','DurableMetricJournal','_require_runtime','recover_terminal_prefix'):monkeypatch.setattr(worker,name,forbidden)
    with pytest.raises(ValueError):worker.execute_one_primary_proxy_trial(tmp_path,engine=Engine(),campaign=Campaign(),trial_id='TEST ONLY',pins=(),bindings={},validation_view=None,validation_evaluators={},metric_identity=None,root=tmp_path,journal_path=tmp_path/'unused',capacity_request=None)
    assert events==(['admission'] if admission_refuses else ['admission','capacity'])
    assert not list(tmp_path.iterdir())
