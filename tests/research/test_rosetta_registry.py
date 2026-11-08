from pathlib import Path
import pytest
from mora3.research.rosetta_registry import (
    KEYS, SPECS, VIEW, RosettaBudget, checked_budgets, digest, load_assets,
)

@pytest.mark.parametrize('budgets',[{}, {k:RosettaBudget(1,1) for k in KEYS[:-1]}, {k:RosettaBudget(1,2) for k in KEYS}])
def test_incomplete_or_unsupported_budgets_fail(budgets):
    with pytest.raises(ValueError):checked_budgets(budgets)

def test_real_pinned_ten_metadata_routes():
    root=Path.cwd();view,old,codec,availability,refs,raws=load_assets(root)
    budgets=checked_budgets({k:RosettaBudget(2,128 if SPECS[k][0]=='LINCS-Pilot1' else 1) for k in KEYS})
    bindings,plans=_construct(root,view,old,codec,digest(b'TEST ONLY Rosetta metadata snapshot'),budgets)
    assert tuple(k for k,_ in bindings)==KEYS and len(plans)==10
    assert all(b.route.native_view_policy_sha256==VIEW for _,b in bindings)
    assert len({(x[2],x[3]) for x in SPECS.values()})==10
    assert all(not a['validation_evaluator_bound'] for a in availability.values())
from mora3.research.rosetta_registry import _construct

@pytest.mark.parametrize('snapshot',[{'path':'../escape','sha256':'a'*64},{'path':'README.md','sha256':'a'*64}])
def test_snapshot_refused_before_source_construction(snapshot,monkeypatch):
    from mora3.research import rosetta_registry as module
    def forbidden(*args,**kwargs):raise AssertionError('source construction must not be reached')
    monkeypatch.setattr(module,'load_assets',forbidden)
    with pytest.raises(ValueError):
        module.build_rosetta_registry(Path.cwd(),source_snapshot=snapshot,budgets={k:RosettaBudget(1,1) for k in KEYS})
