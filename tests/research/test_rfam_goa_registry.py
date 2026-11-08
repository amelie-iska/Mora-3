from pathlib import Path
import pytest
from mora3.data.admet import digest
from mora3.research.mup_campaign import executable_adapter_sha256
from mora3.research.rfam_goa_registry import (
    GOA_TASK, GOA_VIEW, RFAM_KEYS, RFAM_VIEW, GOABudget, RfamBudget, build_rfam_goa_registry, load_original_rfam_goa_assets,
)
from mora3.research.rfam_goa_registry import _construct

@pytest.mark.parametrize('kind',['missing','unsupported','zero_cache','bool_batch'])
def test_no_implicit_budgets(kind):
    rb={k:RfamBudget(1,2) for k in RFAM_KEYS};gb=GOABudget(2,1)
    if kind=='missing':rb.pop(RFAM_KEYS[0])
    if kind=='unsupported':rb['unknown']=rb.pop(RFAM_KEYS[0])
    if kind=='zero_cache':
        with pytest.raises(ValueError):GOABudget(2,0)
        return
    if kind=='bool_batch':
        with pytest.raises(ValueError):RfamBudget(True,2)
        return
    with pytest.raises(ValueError):build_rfam_goa_registry('/missing',source_snapshot={},rfam_budgets=rb,goa_budget=gb)

def test_actual_three_native_axes_three_routes_no_model():
    root=Path.cwd();gb=GOABudget(2,1);rb={k:RfamBudget(1,2) for k in RFAM_KEYS}
    rv,pool,availability,refs,raws=load_original_rfam_goa_assets(root,goa_budget=gb)
    bindings,plans=_construct(root,rv,pool,digest(b'TEST ONLY no production source authority'),rb,gb)
    assert len(bindings)==len(plans)==availability['distinct_native_source_axes']==3
    assert availability['validation_rows']==availability['test_rows']==0
    for (key,binding),plan in zip(bindings,plans,strict=True):
        assert key==plan.task_key
        assert plan.initial_cursor_sha256==digest(binding.route.initial_cursor)
        assert plan.adapter_sha256==executable_adapter_sha256(binding.prepare,root=root)
        assert binding.route.native_view_policy_sha256==(GOA_VIEW if key==GOA_TASK else RFAM_VIEW)
