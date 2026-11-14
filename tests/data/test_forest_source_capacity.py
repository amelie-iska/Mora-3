"""Explicit aggregate-prefix envelope; all fixtures remain synthetic."""
import pytest
from test_forest_source_exports import encoder,exported


def test_explicit_larger_aggregate_bound_keeps_complete_existing_export_identical():
    e=encoder();default=exported(e);large=exported(e,max_total_prefix_tokens=40_000_000)
    assert large.raw==default.raw
    assert large.document()['total_native_prefix_tokens']==default.document()['total_native_prefix_tokens']


@pytest.mark.parametrize('budget',(True,0,64_000_001,1.2))
def test_invalid_aggregate_envelope_fails_before_neural_call(budget):
    e=encoder();calls=[];handle=e.geometry.register_forward_pre_hook(lambda *a:calls.append(True))
    try:
        with pytest.raises(ValueError):exported(e,max_total_prefix_tokens=budget)
        assert not calls
    finally:handle.remove()


@pytest.mark.parametrize('key,value',(('payload_max_tokens',True),('payload_max_tokens',262145),('max_total_prefix_tokens',64_000_001)))
def test_reference_tool_rejects_bad_capacity_before_creating_output(tmp_path,key,value):
    from pathlib import Path
    import runpy
    run=runpy.run_path(str(Path(__file__).resolve().parents[2]/'tools/export_source_forest_reference.py'))['run']
    destination=tmp_path/'must-not-exist'
    with pytest.raises(ValueError):run(destination,**{key:value})
    assert not destination.exists()
