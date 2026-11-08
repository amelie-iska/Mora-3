import copy
from dataclasses import replace

import pytest
import torch

from mora3.model.full_owner import PrimaryConfiguration, construct_primary_owner
from mora3.model.full_owner_mup import full_owner_mup_parameter_roles, full_owner_mup_width_family
from mora3.model.morphology import MorphologyInstance
from test_full_owner import actual_core_assets, cases
from test_full_owner_mup import AXES
from test_goa_objective import view
from test_native_grayscale_flow import condition


def test_grayscale_forward_opt_in_binds_exact_shared_owner_roles_and_checkpoint(tmp_path, actual_core_assets):
    envelope = condition().source_envelope
    sources = replace(actual_core_assets, goa_packets=(view(tmp_path),), grayscale_source_envelope=envelope)
    configuration = PrimaryConfiguration(8, 8, 'a' * 64, readout_mult=1.5)
    original = construct_primary_owner(configuration, sources)
    assert original.gray_forward is None and 'gray_forward' not in original.get_extra_state()
    with pytest.raises(ValueError, match='source envelope'):
        construct_primary_owner(configuration, replace(sources, grayscale_source_envelope=None,
                                                       grayscale_forward_enabled=True))
    enabled = replace(sources, grayscale_forward_enabled=True)
    members = tuple(construct_primary_owner(replace(configuration, width=width), enabled)
                    for width in (8, 12, 16))
    base, _, target = members
    for member in members:
        branch = member.gray_forward
        assert type(branch) is MorphologyInstance and branch.source_envelope == envelope
        assert branch.backbone is member.core.parent.parent.annotation.backbone
        assert member.get_extra_state()['gray_forward'] == branch.get_extra_state()
    family = full_owner_mup_width_family(*members, **AXES)
    assert family
    roles = full_owner_mup_parameter_roles(target, **AXES)
    parameters = dict(target.named_parameters(remove_duplicate=False))
    local = 'gray_forward.backbone.feature.weight'
    shared = 'core.parent.parent.annotation.backbone.feature.weight'
    assert roles[local] == roles[shared] and parameters[local] is parameters[shared]
    state = copy.deepcopy(base.state_dict())
    clone = copy.deepcopy(base)
    clone.load_state_dict(state)
    assert clone.get_extra_state() == base.get_extra_state()
    with pytest.raises(ValueError):
        base.load_state_dict(original.state_dict())
    with pytest.raises(ValueError, match='not full corpus'):
        base.require_training_admission()
