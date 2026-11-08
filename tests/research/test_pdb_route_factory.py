from pathlib import Path
import pytest
from mora3.data.admet import digest
from mora3.research.mup_campaign import executable_adapter_sha256
from mora3.research.pdb_route_factory import PDBRouteBudget,PartialPDBRegistry,load_original_pdb_assets,_resource_binding,_construct,build_original_pdb_registry

@pytest.mark.parametrize('values',[(True,1,1,1,1),(1,0,1,1,1),(1,1,-1,1,1),(1,1,1,1,0)])
def test_no_implicit_invalid_resource_budgets(values):
    with pytest.raises(ValueError):PDBRouteBudget(*values)

def test_stale_snapshot_before_source_read(tmp_path):
    (tmp_path/'snapshot').write_bytes(b'stale')
    with pytest.raises(ValueError,match='digest'):
        build_original_pdb_registry(tmp_path,source_snapshot=dict(path='snapshot',sha256='a'*64),expected_view_policy_sha256='b'*64,budget=PDBRouteBudget(1,1,1,1,4000000))

def test_actual_complete_original_route_metadata_no_model():
    root=Path.cwd();view,axes,availability,refs,raws=load_original_pdb_assets(root)
    assert availability['train_packets']==2 and availability['masked_atoms']==128
    assert availability['observed_target_cartesian_components']==384 and axes.max_atoms==1821
    assert availability['validation_packets']==availability['test_packets']==0
    budget=PDBRouteBudget(1,1,1,1,4000000)
    resource=_resource_binding(axes,availability,budget)
    with pytest.raises(ValueError,match='attention'):_resource_binding(axes,availability,PDBRouteBudget(1,1,1,1,100))
    binding,plan=_construct(root,view,axes,digest(b'TEST ONLY: not a production owner snapshot'),budget)
    assert plan.initial_cursor_sha256==digest(binding.route.initial_cursor)
    assert plan.adapter_sha256==executable_adapter_sha256(binding.prepare,root=root)
    result=PartialPDBRegistry(binding,plan,budget,resource,availability,tuple(sorted(refs.items())))
    assert not result.training_authorized and not result.allocation_authorized and not result.complete_campaign
    assert len(result.unresolved)==3
