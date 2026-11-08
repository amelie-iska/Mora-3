from pathlib import Path
import pytest
from mora3.research import remaining_eight_registry as m
@pytest.mark.parametrize('budgets',[{}, {k:m.RouteBudget(1,2,1) for k in m.KEYS[:-1]}, {k:m.RouteBudget(2,2,1) for k in m.KEYS}])
def test_missing_or_unsupported_budget(budgets):
    with pytest.raises(ValueError):m.checked_budgets(budgets)
def test_real_native_metadata():
    root=Path.cwd();assets=m.load_assets(root)
    budgets=m.checked_budgets({k:m.RouteBudget(1,2,128 if m.SPECS[k][0]=='UniProt-function-annotations' else 1) for k in m.KEYS})
    bindings,plans=m._construct(root,assets,m.digest(b'TEST ONLY remaining eight source snapshot'),budgets)
    assert len(bindings)==11 and len(set(m.SPECS.values()))==8
    assert tuple(k for k,_ in bindings)==m.KEYS
    assert all(b.route.native_view_policy_sha256==(m.INVERSE if m.SPECS[k][0]=='UniProt-function-annotations' else m.BASE) for k,b in bindings)
    assert all(not r['validation_bound'] for r in assets[-3].values())
@pytest.mark.parametrize('snapshot',[{'path':'../escape','sha256':'a'*64},{'path':'README.md','sha256':'a'*64}])
def test_bad_snapshot_refused_before_native_assets(snapshot,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Native asset construction must not be reached')
    monkeypatch.setattr(m,'load_assets',forbidden)
    with pytest.raises(ValueError):m.build_remaining_eight_registry(Path.cwd(),source_snapshot=snapshot,budgets={k:m.RouteBudget(1,2,1) for k in m.KEYS})
