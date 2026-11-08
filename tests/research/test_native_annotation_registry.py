from pathlib import Path
import pytest
from mora3.data.admet import digest
from mora3.research.mup_campaign import executable_adapter_sha256
from mora3.research.native_annotation_registry import AnnotationRouteBudget,ROUTES,VIEW,checked_budgets,load_original_annotation_assets,_construct,build_original_annotation_registry,PartialAnnotationRegistry

def budgets():return {k:AnnotationRouteBudget(1,2) for k in ROUTES}

@pytest.mark.parametrize('kind',['missing','unsupported','bool','wrong_view'])
def test_fail_closed_scope_and_budget(kind):
    b=budgets();policy=VIEW
    if kind=='missing':b.pop(ROUTES[0])
    if kind=='unsupported':b['unqualified-source']=b.pop(ROUTES[0])
    if kind=='wrong_view':policy='a'*64
    if kind=='bool':
        with pytest.raises(ValueError):AnnotationRouteBudget(True,2)
        return
    with pytest.raises(ValueError):build_original_annotation_registry('/missing',source_snapshot={},expected_view_policy_sha256=policy,budgets=b)

def test_three_actual_pinned_native_train_routes_without_model():
    root=Path.cwd();view,sources,availability,refs,raws=load_original_annotation_assets(root)
    assert sorted(v['train_rows'] for v in availability.values())==[1653,2706,474656]
    assert all(v['validation_rows']==v['test_rows']==0 for v in availability.values())
    ordered=checked_budgets(budgets())
    bindings,plans=_construct(root,view,sources,digest(b'TEST ONLY source construction identity'),ordered)
    for (key,binding),plan in zip(bindings,plans,strict=True):
        assert key==plan.task_key and binding.route.native_view_policy_sha256==VIEW
        assert plan.initial_cursor_sha256==digest(binding.route.initial_cursor)
        assert plan.adapter_sha256==executable_adapter_sha256(binding.prepare,root=root)
    result=PartialAnnotationRegistry(bindings,plans,ordered,availability,tuple(sorted(refs.items())))
    assert not result.training_authorized and not result.allocation_authorized and not result.complete_campaign
