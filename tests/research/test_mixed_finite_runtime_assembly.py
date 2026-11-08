from dataclasses import replace
from pathlib import Path
import pytest
from mora3.data.admet import digest
from mora3.research.mixed_finite_runtime_assembly import AXES,POLICIES,BASE_POLICY,INVERSE_POLICY,prior,reaction,assay,pdb,uniprot,_merge

@pytest.fixture(scope='module')
def constructed():
    root=Path.cwd();snapshot=digest(b'TEST ONLY mixed32 metadata; not production authority')
    view,sources,vocab,_,_=assay.load_original_assay_assets(root)
    a=assay._construct(root,view,sources,vocab,snapshot,assay.checked_budgets({k:assay.AssayBudget(1,128,2) for k in assay.ROUTES}))
    p=prior._construct(root,view,vocab,snapshot,prior.checked_budgets({k:prior.PriorBudget(1,4096,128,2) for k in prior.ROUTES}))
    source,codec,_,_=reaction.load_original_reactzyme_assets(root);rb,rp=reaction._construct(root,source,codec,snapshot,reaction.ReactZymeBudget(1,128,2))
    pv,axes,avail,_,_=pdb.load_original_pdb_assets(root);db,dp=pdb._construct(root,pv,axes,snapshot,pdb.PDBRouteBudget(1,2,1,1,4000000))
    uv,us,uc,_,_,_=uniprot.load_full_record_assets(root)
    u=uniprot._construct(root,uv,us,uc,snapshot,uniprot.checked_budgets({k:uniprot.FullRecordBudget(1,128,2) for k in uniprot.ROUTES}))
    groups=(p,(((rp.task_key,rb),),(rp,)),a,(((dp.task_key,db),),(dp,)),u)
    plans=tuple(sorted(tuple(x for _,ps in groups for x in ps),key=lambda x:x.task_key))
    return root,snapshot,groups,plans

def test_real32_metadata_routes_keep_per_axis_policy(constructed):
    root,snapshot,groups,plans=constructed
    result=_merge(root,plans,groups,snapshot=snapshot)
    assert len(result)==32
    assert sum(POLICIES[k]==BASE_POLICY for k in AXES)==19
    assert sum(POLICIES[k]==INVERSE_POLICY for k in AXES)==13
    assert all(b.route.native_view_policy_sha256==POLICIES[k] for k,b in result)

@pytest.mark.parametrize('kind',['missing','duplicate','snapshot','plan','inverse_guard'])
def test_mixed_scope_drift_refused(constructed,kind):
    root,snapshot,groups,plans=constructed
    if kind=='missing':groups=groups[:-1]
    if kind=='duplicate':groups=groups+(groups[-1],)
    if kind=='snapshot':snapshot='a'*64
    if kind=='plan':plans=(replace(plans[0],max_windows=3),)+plans[1:]
    if kind=='inverse_guard':
        binding=groups[-1][0][0][1];route=binding.route;old=route.native_view_policy_sha256
        route.native_view_policy_sha256=BASE_POLICY
        try:
            with pytest.raises(ValueError):_merge(root,plans,groups,snapshot=snapshot)
        finally:route.native_view_policy_sha256=old
        return
    with pytest.raises(ValueError):_merge(root,plans,groups,snapshot=snapshot)
