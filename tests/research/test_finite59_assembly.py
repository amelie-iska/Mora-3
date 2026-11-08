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
def assembled():
    root=Path.cwd();out=Path(tempfile.mkdtemp(prefix='TEST_ONLY_NOT_AUTHORITY_',dir=root/'outputs/implementation/readiness-20261108-finite59-runtime-integration-v1'))
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

def test_public_real59_pinned_coordinates_TEST_ONLY(assembled):
    root,kwargs,_=assembled
    result=m.bind_finite59_results(root,**kwargs)
    assert len(result.bindings)==59 and len(result.plans)==59
    assert sum(p==m.BASE_POLICY for _,p in result.per_route_view_policies)==40
    assert result.validation_compatibility['base18']['original_validation_coordinates_verified']==114
    assert not result.training_authorized and not result.allocation_authorized and not result.complete_campaign

@pytest.mark.parametrize('kind',['denominator','missing','budget','snapshot'])
def test_public_refuses_drift(assembled,kind):
    root,original,pin=assembled;kw=dict(original)
    if kind=='denominator':
        c=kw['campaign'];task=replace(c.validation.tasks[0],observed_targets=5226)
        c=replace(c,validation=replace(c.validation,tasks=(task,)))
        kw.update(campaign=c,campaign_reference=pin('TEST_ONLY_bad_denominator.json',c.wire()))
    elif kind=='missing':kw['remaining_result']=replace(kw['remaining_result'],bindings=kw['remaining_result'].bindings[:-1],plans=kw['remaining_result'].plans[:-1])
    elif kind=='budget':kw['rfam_goa_budget_binding_sha256']='a'*64
    else:kw['source_snapshot']=dict(path=kw['source_snapshot']['path'],sha256='a'*64)
    with pytest.raises(ValueError):m.bind_finite59_results(root,**kw)
