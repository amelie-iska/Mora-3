from pathlib import Path
import pytest
from mora3.data.admet import digest
from mora3.research.mup_campaign import executable_adapter_sha256
from mora3.research.finite_assay_registry import AssayBudget,ROUTES,EXCLUDED,checked_budgets,build_original_assay_registry,load_original_assay_assets,_construct,PartialAssayRegistry

def budgets():return {r:AssayBudget(1,128,2) for r in ROUTES}

@pytest.mark.parametrize('values',[(True,1,1),(1,0,1),(1,1,-1)])
def test_budget_refusal(values):
    with pytest.raises(ValueError):AssayBudget(*values)

@pytest.mark.parametrize('kind',['missing','unsupported','untyped'])
def test_no_partial_silent_coverage(kind):
    b=budgets();key=next(iter(b))
    if kind=='missing':b.pop(key)
    if kind=='unsupported':b['inactive-expansion']=b.pop(key)
    if kind=='untyped':b[key]={}
    with pytest.raises(ValueError):
        build_original_assay_registry('/missing',source_snapshot={},expected_view_policy_sha256='a'*64,budgets=b)

def test_stale_external_snapshot(tmp_path):
    (tmp_path/'snapshot').write_bytes(b'stale')
    with pytest.raises(ValueError,match='digest'):
        build_original_assay_registry(tmp_path,source_snapshot=dict(path='snapshot',sha256='a'*64),expected_view_policy_sha256='b'*64,budgets=budgets())

def test_real_original_assets_construct_all_ten_under_test_only_identity():
    root=Path.cwd();view,sources,vocab,refs,raws=load_original_assay_assets(root)
    ordered=checked_budgets(budgets())
    bindings,plans=_construct(root,view,sources,vocab,digest(b'TEST ONLY: no authenticated full-owner snapshot'),ordered)
    assert len(bindings)==len(plans)==10
    for (task,binding),plan in zip(bindings,plans,strict=True):
        assert task==plan.task_key
        assert plan.initial_cursor_sha256==digest(binding.route.initial_cursor)
        assert plan.adapter_sha256==executable_adapter_sha256(binding.prepare,root=root)
    result=PartialAssayRegistry(bindings,plans,ordered,tuple(sorted(refs.items())))
    assert not result.complete_campaign and not result.allocation_authorized and not result.training_authorized
    assert len(EXCLUDED)==4
    # No windows, target preparation, owner allocation or model execution.
