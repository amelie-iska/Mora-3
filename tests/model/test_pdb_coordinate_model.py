"""Native PDB condition exclusion and shared width-family contracts."""
from dataclasses import replace
import hashlib
import json

import pytest
import torch

from mora3.data.pdb_masked_coordinates import (
    ATOM_FIELDS, FAMILY, SCHEMA, TASK, PDBCoordinateAxes, _project,
)
from mora3.model.mu_optimizer import MuAdamW
from test_sciplex_mup import make as sciplex


def source_row(target):
    atoms = [dict(zip(ATOM_FIELDS, ('C', atom, 'A', 'A', '1', '.'))) for atom in ('C1', 'C2', 'C3')]
    task = dict(
        schema=SCHEMA, task=TASK,
        conditions=dict(coordinate_unit='angstrom', known_atom_axis=atoms,
                        positions_angstrom=[[1., 2., 3.], None, [2., 4., 6.]]),
        targets=dict(masked_positions_angstrom=[None, target, None]),
        audit=dict(masked_target_atom_indices=[1]),
    )
    raw = json.dumps(task).encode()
    return dict(source_family=FAMILY, native_schema=SCHEMA, task_name=TASK, split='train',
                native_payload=raw, native_payload_sha256=hashlib.sha256(raw).hexdigest())


def packet_and_axes():
    packet = _project(source_row([2.5, 3.5, 4.5]), 'fixture', 0)
    labels = tuple(tuple(sorted(set(values))) for values in zip(*packet.condition.atom_labels))
    axes = PDBCoordinateAxes(labels, 3, (packet.source_coordinate,), 'a' * 64)
    return packet, axes


def model(width=8):
    from mora3.model.pdb_coordinate_model import PDBMaskedCoordinates
    _, axes = packet_and_axes()
    return PDBMaskedCoordinates(sciplex(width).double().backbone, axes, readout_mult=1.5)


def test_source_projection_excludes_withheld_values_and_rejects_invalid_mask():
    original, _ = packet_and_axes()
    altered = _project(source_row([1e4, 2e4, -3e4]), 'fixture', 0)
    assert original.condition == altered.condition
    assert original.masked_targets_angstrom != altered.masked_targets_angstrom
    with pytest.raises(ValueError, match='either observed or explicitly withheld'):
        _project(source_row(None), 'fixture', 0)


def test_prediction_translates_with_observed_atoms_and_updates_all_shared_paths():
    packet, _ = packet_and_axes()
    m = model(16)
    condition = packet.condition
    shift = (32., -21., 10.)
    shifted = replace(condition, positions_angstrom=tuple(
        tuple(a + b for a, b in zip(x, shift)) if x is not None else None
        for x in condition.positions_angstrom
    ))
    source, translated = m((condition, shifted))
    assert source.shape == (1, 3)
    torch.testing.assert_close(translated - source, source.new_tensor([shift]), rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match='target-free'):
        m((packet,))
    source.square().sum().backward()
    for parameter in (m.features.weight, m.graph.incidence.weight, m.readout.weight,
                      m.backbone.layers[0].attention.qkv.weight):
        assert parameter.grad is not None and torch.count_nonzero(parameter.grad)


def test_shared_backbone_and_native_xyz_readout_across_widths():
    from mora3.model.pdb_coordinate_model import pdb_coordinate_parameter_roles, pdb_coordinate_width_family
    models = tuple(model(w) for w in (8, 12, 16))
    family = pdb_coordinate_width_family(*models)
    target = models[-1]
    assert target.graph.layers is target.backbone.layers
    assert target.graph.incidence is not target.backbone.incidence
    roles = pdb_coordinate_parameter_roles(target)
    assert roles['readout.weight'].role == 'output'
    assert roles['readout.weight'].axes[0].kind == 'finite'
    assert family.total_numel == tuple(sum(p.numel() for p in m.parameters()) for m in models)
    optimizer = MuAdamW(target, family, roles, base_lr=.01, weight_decay=0., decay_aliases=())
    group, = [g for g in optimizer.param_groups if 'readout.weight' in g['mup_aliases']]
    assert group['lr'] == .01 and not optimizer.state
