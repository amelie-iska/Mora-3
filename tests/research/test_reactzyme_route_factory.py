import pytest
from mora3.research.reactzyme_route_factory import ReactZymeBudget,build_original_reactzyme_registry,_receipt_check

@pytest.mark.parametrize('values',[(True,1,1),(1,0,1),(1,1,-1),(1,1,1.0)])
def test_invalid_budgets(values):
    with pytest.raises(ValueError):ReactZymeBudget(*values)

def test_no_implicit_budget_or_snapshot(tmp_path):
    with pytest.raises(ValueError,match='budget'):
        build_original_reactzyme_registry(tmp_path,source_snapshot={},expected_view_policy_sha256='a'*64,budget=None)
    with pytest.raises(ValueError):
        build_original_reactzyme_registry(tmp_path,source_snapshot={},expected_view_policy_sha256='a'*64,budget=ReactZymeBudget(1,128,2))

def test_expansion_receipt_cannot_substitute():
    with pytest.raises(ValueError,match='original'):_receipt_check({'schema':'inactive-expansion'},None)

def test_stale_snapshot_before_reader(tmp_path):
    (tmp_path/'snapshot').write_bytes(b'stale')
    with pytest.raises(ValueError,match='digest'):
        build_original_reactzyme_registry(tmp_path,source_snapshot=dict(path='snapshot',sha256='a'*64),
            expected_view_policy_sha256='b'*64,budget=ReactZymeBudget(1,128,2))

def test_real_readonly_original_assets_and_binding_under_explicit_test_identity():
    from pathlib import Path
    from mora3.data.admet import digest
    from mora3.research.mup_campaign import executable_adapter_sha256
    from mora3.research.reactzyme_route_factory import load_original_reactzyme_assets,_construct,PartialReactZymeRegistry
    root=Path.cwd()
    source,codec,refs,raws=load_original_reactzyme_assets(root)
    assert source.axes.guard_clear_role_counts==(40426,0,0)
    assert len(source._unresolved)==16 and len(source.source_paths)==161
    # A test-only identity verifies factory wiring; it is never a production request.
    snapshot=digest(b'TEST ONLY: not an authenticated full-owner snapshot')
    budget=ReactZymeBudget(1,128,2)
    binding,plan=_construct(root,source,codec,snapshot,budget)
    assert plan.initial_cursor_sha256==digest(binding.route.initial_cursor)
    assert plan.adapter_sha256==executable_adapter_sha256(binding.prepare,root=root)
    assert binding.prepare.__self__.source is source
    assert binding.prepare.__self__.direction=='enzyme_product'
    result=PartialReactZymeRegistry(binding,plan,budget,tuple(sorted(refs.items())))
    assert not result.training_authorized and not result.complete_campaign and not result.allocation_authorized
    assert 'No original validation/test roles' in result.validation_obligation
    # No source windows or targets are consumed and evaluate is never called.
