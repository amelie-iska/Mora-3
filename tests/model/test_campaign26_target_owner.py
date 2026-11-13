"""The actual current full owner must refuse before target execution side effects."""
import torch
import pytest
from test_full_owner import actual_core_assets,cases,_construct
from mora3.research.primary_training import NativeMuUpdateEngine
from mora3.research.campaign26_transfer import require_transfer_target


def test_current_full_owner_admission_is_not_bypassed_by_transfer_artifact(tmp_path,actual_core_assets):
    torch.manual_seed(20261110)
    owner=_construct(tmp_path,actual_core_assets)
    engine=object.__new__(NativeMuUpdateEngine);engine.model=owner
    # There is deliberately no optimizer on this object. Native admission must
    # refuse before accessing it or interpreting a purported transfer artifact.
    values={n:p.detach().clone()for n,p in owner.named_parameters()}
    rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match='not full corpus'):
        require_transfer_target(engine,{'untrusted':'not examined before admission'},'full80m-2026111001')
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(values[n],p)and p.grad is None for n,p in owner.named_parameters())


def test_target_configuration_matches_actual_factory_wire_without_issuing_admission(tmp_path,actual_core_assets):
    import copy
    from dataclasses import replace
    from mora3.model.full_owner import PrimaryConfiguration
    from mora3.research.campaign26_transfer import require_primary_target_configuration
    owner=_construct(tmp_path,actual_core_assets)
    config=PrimaryConfiguration(8,8,'a'*64)
    before=copy.deepcopy(owner.get_extra_state())
    result=require_primary_target_configuration(owner,config)
    assert result==before and result['training_admission']is False
    assert 'heads'in result['primary_configuration']
    assert 'source_snapshot_sha256'not in result['primary_configuration']
    for change in(dict(width=12),dict(source_snapshot_sha256='b'*64),dict(attn_mult=.5),dict(readout_mult=.5)):
        with pytest.raises(ValueError):require_primary_target_configuration(owner,replace(config,**change))
    assert owner.get_extra_state()==before
