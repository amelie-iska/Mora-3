"""Whole cooperative owner width roles for the source mask-layer count head."""
from dataclasses import replace

from test_full_owner import actual_core_assets, cases
from test_full_owner_mup import AXES, members
from mora3.model.full_owner_mup import full_owner_mup_parameter_roles, full_owner_mup_width_family
from mora3.model.mu_optimizer import MuAdamW
from mora3.model.raster_source_envelope import RasterSourceEnvelope


def test_count_and_mask_source_axes_in_registered_whole_owner(tmp_path, actual_core_assets):
    envelope = RasterSourceEnvelope('c' * 64, 512, 512, 297)
    assets = replace(
        actual_core_assets,
        raster_source_envelope=envelope,
        color_input_policy='source-envelope-v2',
        instance_slot_policy='source-layer-list-v4',
        instance_slots=envelope.max_instances,
    )
    models = members(tmp_path, assets)
    family = full_owner_mup_width_family(*models, **AXES)
    target = models[-1]
    roles = full_owner_mup_parameter_roles(target, **AXES)
    names = dict(target.named_parameters(remove_duplicate=False))
    count_weight = 'core.instance.color.count.weight'
    count_bias = 'core.instance.color.count.bias'
    mask_weight = 'core.instance.color.head.projection.weight'
    assert names[count_weight] is target.core.instance.color.count.weight
    assert names[count_weight].shape == (298, 16)
    assert names[count_bias].shape == (298,)
    assert roles[count_weight].role == 'output' and roles[count_bias].role == 'vector'
    assert roles[count_weight].axes[0] == roles[count_bias].axes[0]
    assert roles[count_weight].axes[0] != roles[mask_weight].axes[0]
    assert roles[count_weight].axes[0].kind == 'finite'
    assert set(names) == set(roles)
    assert family.total_numel == tuple(sum(p.numel() for p in model.parameters()) for model in models)
    optimizer = MuAdamW(target, family, roles, base_lr=.01, weight_decay=0., decay_aliases=())
    group, = [g for g in optimizer.param_groups if count_weight in g['mup_aliases']]
    assert group['lr'] == .01 and not optimizer.state
