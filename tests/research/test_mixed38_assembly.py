from dataclasses import replace
from pathlib import Path
import pytest
from mora3.data.admet import digest
from mora3.research.mixed38_assembly import AXES,POLICIES,BASE_POLICY,INVERSE_POLICY,prior,reaction,assay,pdb,uniprot,annotation,rg,_merge

@pytest.fixture(scope='module')
def constructed():
    root=Path.cwd();snapshot=digest(b'TEST ONLY mixed38 metadata; not production authority')
    view,sources,vocab,_,_=assay.load_original_assay_assets(root)
    a=assay._construct(root,view,sources,vocab,snapshot,assay.checked_budgets({k:assay.AssayBudget(1,128,2) for k in assay.ROUTES}))
    p=prior._construct(root,view,vocab,snapshot,prior.checked_budgets({k:prior.PriorBudget(1,4096,128,2) for k in prior.ROUTES}))
    source,codec,_,_=reaction.load_original_reactzyme_assets(root);rb,rp=reaction._construct(root,source,codec,snapshot,reaction.ReactZymeBudget(1,128,2))
    pv,axes,avail,_,_=pdb.load_original_pdb_assets(root);db,dp=pdb._construct(root,pv,axes,snapshot,pdb.PDBRouteBudget(1,2,1,1,4000000))
    uv,us,uc,_,_,_=uniprot.load_full_record_assets(root)
    u=uniprot._construct(root,uv,us,uc,snapshot,uniprot.checked_budgets({k:uniprot.FullRecordBudget(1,128,2) for k in uniprot.ROUTES}))
    nv,ns,_,_,_=annotation.load_original_annotation_assets(root)
    n=annotation._construct(root,nv,ns,snapshot,annotation.checked_budgets({k:annotation.AnnotationRouteBudget(1,2) for k in annotation.ROUTES}))
    gb=rg.GOABudget(2,1);rv,pool,_,_,_=rg.load_original_rfam_goa_assets(root,goa_budget=gb)
    g=rg._construct(root,rv,pool,snapshot,{k:rg.RfamBudget(1,2) for k in rg.RFAM_KEYS},gb)
    groups=(p,(((rp.task_key,rb),),(rp,)),a,(((dp.task_key,db),),(dp,)),u,n,g)
    plans=tuple(sorted(tuple(x for _,ps in groups for x in ps),key=lambda x:x.task_key))
    return root,snapshot,groups,plans

def test_real38_metadata_routes_keep_per_axis_policy(constructed):
    root,snapshot,groups,plans=constructed
    result=_merge(root,plans,groups,snapshot=snapshot)
    assert len(result)==38
    assert sum(POLICIES[k]==BASE_POLICY for k in AXES)==23
    assert sum(POLICIES[k]==INVERSE_POLICY for k in AXES)==15
    assert all(b.route.native_view_policy_sha256==POLICIES[k] for k,b in result)

@pytest.mark.parametrize('kind',['missing','duplicate','snapshot','plan','inverse_guard'])
def test_mixed_scope_drift_refused(constructed,kind):
    root,snapshot,groups,plans=constructed
    if kind=='missing':groups=groups[:-1]
    if kind=='duplicate':groups=groups+(groups[-1],)
    if kind=='snapshot':snapshot='a'*64
    if kind=='plan':plans=(replace(plans[0],max_windows=3),)+plans[1:]
    if kind=='inverse_guard':
        binding=groups[4][0][0][1];route=binding.route;old=route.native_view_policy_sha256
        route.native_view_policy_sha256=BASE_POLICY
        try:
            with pytest.raises(ValueError):_merge(root,plans,groups,snapshot=snapshot)
        finally:route.native_view_policy_sha256=old
        return
    with pytest.raises(ValueError):_merge(root,plans,groups,snapshot=snapshot)


def test_live_cache_budget_is_cross_bound(constructed):
    from mora3.research.mixed38_assembly import _rfam_goa_budget_digest
    _,_,groups,_=constructed
    bindings,plans=groups[-1]
    before=_rfam_goa_budget_digest(bindings,plans)
    pool=dict(bindings)[rg.GOA_TASK].prepare.__self__.pool
    old=pool.max_cached_kb_parts
    try:
        pool.max_cached_kb_parts=old+1
        assert _rfam_goa_budget_digest(bindings,plans)!=before
    finally:pool.max_cached_kb_parts=old
