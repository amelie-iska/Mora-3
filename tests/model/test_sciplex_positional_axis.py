"""Repeated literal SciPlex gene labels retain all ordered source columns."""

import pytest

from mora3.model.flow_mup import flow_mup_parameter_roles
from mora3.model.parameterization import response_parameter_ledger
from mora3.model.response_flow import ResponseFlowDenoiserV2
from mora3.model.sciplex import SciPlexSharedModel
from mora3.model.sciplex_mup import sciplex_mup_parameter_roles, sciplex_mup_width_family


def make(width, labels):
    return SciPlexSharedModel(
        labels, width=width, scaling_policy='mup-v1', initialization_policy='mup-transformer-v1',
        parameterization_policy='mup-sciplex-v1', base_width=8,
    )


def test_exact_native_duplicate_positions_match_response_and_flow_operators():
    labels = ('A', 'B', 'A', 'C')
    models = tuple(make(width, labels) for width in (8, 12, 16))
    family = sciplex_mup_width_family(*models)
    assert family.total_numel == tuple(sum(p.numel() for p in m.parameters()) for m in models)
    native = models[-1]
    assert native.axis_ids == labels and native.expression.weight.shape == (16, 4)
    assert native.response_head.weight.shape == (4, 16)
    roles = sciplex_mup_parameter_roles(native)
    axis = roles['expression.weight'].axes[1]
    assert roles['response_head.weight'].axes[0] == axis
    flow = ResponseFlowDenoiserV2(native, parameterization_policy='mup-response-flow-v1')
    roles = flow_mup_parameter_roles(flow)
    assert roles['state.weight'].axes[1] == roles['output.weight'].axes[0] == axis
    assert roles['native.response_head.bias'].axes[0] == axis
    moved = make(16, ('A', 'A', 'B', 'C'))
    assert sciplex_mup_parameter_roles(moved)['expression.weight'].axes[1] != axis
    with pytest.raises(ValueError):
        sciplex_mup_width_family(*models[:2], moved)


def test_legacy_inventory_preserves_unique_label_contract():
    duplicate = SciPlexSharedModel(('A', 'A'))
    with pytest.raises(ValueError, match='unique'):
        response_parameter_ledger(duplicate)
    unique = make(8, ('A', 'B'))
    historical = response_parameter_ledger(unique)
    old = next(row.axes[1] for row in historical.parameters if 'expression.weight' in row.aliases)
    assert sciplex_mup_parameter_roles(unique)['expression.weight'].axes[1] == old
