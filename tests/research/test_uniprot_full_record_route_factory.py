from pathlib import Path
import pytest
from mora3.data.admet import digest
from mora3.research.mup_campaign import executable_adapter_sha256
from mora3.research.uniprot_full_record_route_factory import (
    ROUTES, TASK, VIEW, FullRecordBudget, PartialFullRecordRegistry, _construct,
    build_full_record_registry, checked_budgets, load_full_record_assets,
)

def budgets():return {r:FullRecordBudget(1,128,2) for r in ROUTES}

@pytest.mark.parametrize('kind',['missing','unsupported','field_batch','base_view'])
def test_refuses_before_source_access(kind):
    b=budgets();policy=VIEW
    if kind=='missing':b.pop(ROUTES[0])
    if kind=='unsupported':b['unsupported']=b.pop(ROUTES[0])
    if kind=='field_batch':b[TASK+':comments']=FullRecordBudget(2,128,2)
    if kind=='base_view':policy='c3812ecfcfea8a01f81769bf5ef6fb3880e04a0ac054a6349e9272db14765393'
    with pytest.raises(ValueError):build_full_record_registry('/missing',source_snapshot={},expected_view_policy_sha256=policy,budgets=b)

def test_real_guarded_source_all_thirteen_adapters_no_model():
    root=Path.cwd();view,source,codec,availability,refs,raws=load_full_record_assets(root)
    assert availability['inverse_train_rows']==76471 and availability['full_record_prediction_train_rows']==106675
    assert availability['validation_rows']==availability['test_rows']==0
    bindings,plans=_construct(root,view,source,codec,digest(b'TEST ONLY: no production owner authority'),checked_budgets(budgets()))
    assert len(bindings)==len(plans)==13
    for (task,binding),plan in zip(bindings,plans,strict=True):
        assert binding.route.native_view_policy_sha256==VIEW
        assert plan.task_key==task
        assert plan.initial_cursor_sha256==digest(binding.route.initial_cursor)
        assert plan.adapter_sha256==executable_adapter_sha256(binding.prepare,root=root)
    result=PartialFullRecordRegistry(bindings,plans,checked_budgets(budgets()),availability,tuple(sorted(refs.items())))
    assert not result.training_authorized and not result.complete_campaign and not result.allocation_authorized
