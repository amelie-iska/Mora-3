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


from mora3.research import combined60_binding as c
from mora3.data.mcsa_role_list_admission import TASK as MCSA,FAMILY as MCSA_FAMILY,NATIVE as MCSA_SCHEMA
from mora3.research.mup_sweep_selection import MAX_UPDATES

@pytest.fixture(scope='module')
def combined(finite_fixture):
    root,inputs,pin=finite_fixture
    cycle=tuple(sorted(m.AXES+(MCSA,)))
    bridge=c.mcsa.build_mcsa_bridge(root,source_snapshot=inputs['source_snapshot'],pass_policy=c.mcsa.MCSAFinitePassPolicy(3,16,2026110201),max_windows=114,task_cycle=cycle,total_updates=MAX_UPDATES,max_validation_bytes=8*1024*1024)
    pending=c.prepare_three_evaluators(root,finite_inputs=inputs,mcsa_bridge=bridge,read_batch_size=8,max_selected_bytes=8*1024*1024)
    coords=dict(pending.coordinates);units=dict(pending.units)
    tasks=(inputs['campaign'].validation.tasks[0],
        MatchedValidationTask(c.FREE,'FreeSolv',c.FREE,'mora3.CorpusTask/v1',coords[c.FREE],42,Decimal('1'),Decimal('1'),*units[c.FREE]),
        MatchedValidationTask(MCSA,MCSA_FAMILY,MCSA,MCSA_SCHEMA,coords[MCSA],2043,Decimal('1'),Decimal('1'),*units[MCSA]))
    cohort=replace(inputs['campaign'].validation,view_policy_sha256=c.matched_validation_source_policy_sha256(pending.sources),read_batch_size=8,max_selected_arrow_bytes=8*1024*1024,tasks=tuple(sorted(tasks,key=lambda t:t.task_key)),train_only_dispositions=tuple((k,'TEST ONLY; no production role authority') for k in cycle if k not in coords))
    campaign=replace(inputs['campaign'],validation=cohort,validation_adapter_sha256=pending.hashes,train_routes=tuple(sorted(inputs['campaign'].train_routes+(bridge.plan,),key=lambda p:p.task_key)),task_cycle=cycle)
    ref=pin('TEST_ONLY_combined_campaign.json',campaign.wire());pin('TEST_ONLY_combined_cohort.json',cohort.wire())
    return root,dict(finite_inputs=inputs,mcsa_bridge=bridge,campaign=campaign,campaign_reference=ref),pin

def test_public_combined_actual_three_enrollments(combined):
    root,args,_=combined;result=c.bind_combined60(root,**args)
    assert len(result.bindings)==60 and len(result.validation.evaluators)==3
    assert dict(result.validation.denominators)=={m.INVERSE:5225,c.FREE:42,MCSA:2043}
    assert not result.training_authorized and not result.allocation_authorized

@pytest.mark.parametrize('kind',['freesolv_denominator','freesolv_train_substitution','mcsa_capacity','mixed_view'])
def test_public_combined_refuses_drift(combined,kind):
    root,original,pin=combined;args=dict(original);campaign=args['campaign']
    if kind.startswith('freesolv'):
        task=next(t for t in campaign.validation.tasks if t.task_key==c.FREE)
        if kind=='freesolv_denominator':task=replace(task,observed_targets=43)
        else:task=replace(task,coordinates=(replace(task.coordinates[0],source_path=task.coordinates[0].source_path.replace('/validation/','/train/')),)+task.coordinates[1:])
        campaign=replace(campaign,validation=replace(campaign.validation,tasks=tuple(task if t.task_key==c.FREE else t for t in campaign.validation.tasks)))
    elif kind=='mixed_view':campaign=replace(campaign,validation=replace(campaign.validation,view_policy_sha256=m.INVERSE_POLICY))
    else:
        cycle = (MCSA,)*8 + tuple(key for key in campaign.task_cycle if key!=MCSA)
        campaign=replace(campaign,task_cycle=cycle)
    args.update(campaign=campaign,campaign_reference=pin('TEST_ONLY_bad_'+kind+'.json',campaign.wire()))
    with pytest.raises(ValueError):c.bind_combined60(root,**args)


def test_mcsa_scan_limit_is_not_its_lifetime_window_capacity(combined):
    root,original,pin=combined
    args=dict(original)
    bridge=args['mcsa_bridge']
    updated=replace(bridge,plan=replace(bridge.plan,max_windows=1))
    args['mcsa_bridge']=updated
    campaign=replace(args['campaign'],train_routes=tuple(
        updated.plan if p.task_key==MCSA else p for p in args['campaign'].train_routes
    ))
    args.update(campaign=campaign,campaign_reference=pin('TEST_ONLY_scan_one.json',campaign.wire()))
    assert dict((p.task_key,p.max_windows) for p in c.bind_combined60(root,**args).plans)[MCSA]==1
