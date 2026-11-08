"""Shared TokenGT reconstruction of withheld deposited PDB coordinates."""
import copy
import hashlib
import json
import math

import torch
from torch import nn
from torch.nn import functional as F

from mora3.data.pdb_masked_coordinates import (
    PDBCoordinateAxes, PDBCoordinateCondition, native_view_policy_sha256, pdb_coordinate_packets,
)
from .mixed_payload_projection import exact_wire
from .morphology_mup import _native_roles, _without_member_dimensions
from .mu_optimizer import _storage_signature
from .parameterization import Axis, ParameterRole, parameter_ledger, width_family
from .tokengt import TokenGT


_SHARED_TRANSFORMS = ('feature', 'role', 'graph_token', 'layers', 'final_norm')


class PDBMaskedCoordinates(nn.Module):
    """Known atom labels and observed-only centered coordinates enter the graph.

    The output predicts withheld XYZ values in the deposited frame. The
    observation centroid supplies translation equivariance. This model treats
    the deposited orientation as given. Its graph has one node per native atom.
    """

    def __init__(self, backbone, axes, *, readout_mult=1.0, max_attention_elements=32_000_000,
                 backbone_identifier_axis=('context', 'observation')):
        super().__init__()
        if type(backbone) is not TokenGT or type(axes) is not PDBCoordinateAxes:
            raise ValueError('Exact shared TokenGT and complete native PDB axes required')
        axes.binding()
        native = backbone.configuration()
        if (
            native['schema'] != 'tokengt-native-configuration-v3' or native['attention'] != 'vanilla'
            or native['feature_width'] != native['width']
            or type(readout_mult) not in (int, float) or not math.isfinite(readout_mult) or readout_mult <= 0
            or type(max_attention_elements) is not int
            or max_attention_elements < native['heads'] * native['layers'] * (axes.max_atoms + 1) ** 2
            or type(backbone_identifier_axis) is not tuple
            or len(backbone_identifier_axis) != native['id_width']
            or any(type(x) is not str or not x for x in backbone_identifier_axis)
            or len(set(backbone_identifier_axis)) != len(backbone_identifier_axis)
        ):
            raise ValueError('Exact ordinary width family and complete PDB graph resource bound required')
        self.backbone = backbone
        self._shared_reference = (backbone,)
        self.axes = axes
        self.width = native['width']
        self.base_width = native['parameterization']['base_width']
        self.readout_mult = float(readout_mult)
        self.max_attention_elements = max_attention_elements
        self.backbone_identifier_axis = backbone_identifier_axis
        self.input_width = sum(len(a) for a in axes.atom_label_axes) + 4
        self.graph = TokenGT.from_configuration(dict(native, id_width=axes.max_atoms))
        for name in _SHARED_TRANSFORMS:
            setattr(self.graph, name, getattr(backbone, name))
        self.features = nn.Linear(self.input_width, self.width, bias=False)
        self.readout = nn.Linear(self.width, 3)
        with torch.no_grad():
            scale = math.sqrt(self.width / self.base_width)
            self.readout.weight.mul_(scale)
            self.readout.bias.mul_(scale)
        reference = backbone.feature.weight
        self.to(device=reference.device, dtype=reference.dtype)
        self._configuration = self._current_configuration()

    def _current_configuration(self):
        return dict(
            schema='Mora3.PDBMaskedCoordinates/v1',
            source=self.axes.binding(),
            native_backbone=self.backbone.configuration(),
            graph=self.graph.configuration(),
            backbone_identifier_axis=self.backbone_identifier_axis,
            base_width=self.base_width,
            readout_mult=self.readout_mult,
            max_attention_elements=self.max_attention_elements,
            features='six native label axes plus observed-centered XYZ plus coordinate-availability bit',
            topology='every native atom row as one node with its exact ordinal identifier',
            output='withheld XYZ in original atom order and deposited angstrom frame',
            objective='mean squared angstrom error over withheld XYZ, then mean over packets',
        )

    def get_extra_state(self):
        if (
            set(self._modules) != {'backbone', 'graph', 'features', 'readout'}
            or self.backbone is not self._shared_reference[0] or type(self.graph) is not TokenGT
            or self.graph.incidence is self.backbone.incidence
            or any(getattr(self.graph, name) is not getattr(self.backbone, name) for name in _SHARED_TRANSFORMS)
            or type(self.features) is not nn.Linear or self.features.bias is not None
            or self.input_width != sum(len(a) for a in self.axes.atom_label_axes) + 4
            or (self.features.in_features, self.features.out_features) != (self.input_width, self.width)
            or type(self.readout) is not nn.Linear or self.readout.bias is None
            or (self.readout.in_features, self.readout.out_features) != (self.width, 3)
            or tuple(self.features.weight.shape) != (self.width, self.input_width)
            or tuple(self.readout.weight.shape) != (3, self.width) or tuple(self.readout.bias.shape) != (3,)
            or not exact_wire(self._current_configuration(), self._configuration)
        ):
            raise ValueError('PDB native source or shared operator configuration drift')
        return copy.deepcopy(self._configuration)

    def set_extra_state(self, state):
        if not exact_wire(state, self.get_extra_state()):
            raise ValueError('PDB native source checkpoint configuration mismatch')

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        for key, value in self.state_dict().items():
            if not isinstance(value, torch.Tensor) and not exact_wire(state_dict.get(prefix + key), value):
                raise ValueError('PDB native source checkpoint configuration mismatch')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def _condition_features(self, condition):
        if type(condition) is not PDBCoordinateCondition:
            raise ValueError('Exact target-free PDB condition required')
        labels = condition.atom_labels
        observed = condition.positions_angstrom
        if (not 1 <= len(labels) <= self.axes.max_atoms or len(labels) != len(observed)
                or any(type(row) is not tuple or len(row) != len(self.axes.atom_label_axes) for row in labels)):
            raise ValueError('Complete native PDB atom axis required')
        for point in observed:
            if point is not None and (
                type(point) is not tuple or len(point) != 3
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in point)
            ):
                raise ValueError('Observed source PDB coordinates must be finite XYZ triples')
        parameter = self.features.weight
        categorical = []
        for field, axis in enumerate(self.axes.atom_label_axes):
            lookup = {label: i for i, label in enumerate(axis)}
            try:
                ids = torch.tensor([lookup[row[field]] for row in labels], device=parameter.device)
            except (KeyError, TypeError) as exc:
                raise ValueError('Native PDB label outside its complete source axis') from exc
            categorical.append(F.one_hot(ids, num_classes=len(axis)).to(parameter.dtype))
        available = torch.tensor([x is not None for x in observed], device=parameter.device)
        if not bool(available.any()) or bool(available.all()):
            raise ValueError('PDB reconstruction requires observed and withheld coordinates')
        coordinates = torch.tensor([x if x is not None else (0., 0., 0.) for x in observed],
                                   dtype=torch.float64, device=parameter.device)
        centroid = coordinates[available].mean(dim=0)
        centered = torch.where(available[:, None], coordinates - centroid, 0.).to(parameter.dtype)
        if not bool(torch.isfinite(centered).all()) or not bool(torch.isfinite(centroid).all()):
            raise ValueError('Finite centered PDB coordinates required in model precision')
        values = torch.cat((*categorical, centered, available[:, None].to(parameter.dtype)), dim=1)
        return values, ~available, centroid

    def forward(self, conditions):
        self.get_extra_state()
        conditions = tuple(conditions)
        if not conditions or any(type(c) is not PDBCoordinateCondition for c in conditions):
            raise ValueError('Nonempty exact target-free PDB conditions required')
        native = self.graph.configuration()
        attention_elements = sum((len(c.atom_labels) + 1) ** 2 for c in conditions) * native['heads'] * native['layers']
        if attention_elements > self.max_attention_elements:
            raise ValueError('Complete PDB attention resource bound exceeded')
        complete = [self._condition_features(c) for c in conditions]
        outputs = []
        for values, masked, centroid in complete:
            size = len(values)
            rows = torch.arange(size, device=values.device)
            graph = dict(
                features=self.features(values),
                endpoints=torch.stack((rows, rows), dim=1),
                roles=torch.zeros_like(rows),
                identifiers=torch.eye(size, self.axes.max_atoms, dtype=values.dtype, device=values.device),
            )
            hidden = self.graph([graph])[0][0, 1:][masked]
            predicted = self.readout(hidden * (self.readout_mult / (self.width / self.base_width)))
            outputs.append(predicted.double() + centroid)
        return tuple(outputs)

    def objective(self, view, batch, *, split='train', return_predictions=False):
        if type(return_predictions) is not bool:
            raise ValueError('Explicit PDB reporting policy required')
        if native_view_policy_sha256(view) != self.axes.native_view_policy_sha256:
            raise ValueError('PDB objective requires its bound current native view')
        packets = pdb_coordinate_packets(view, batch, split=split)
        if not packets or any(p.source_coordinate not in self.axes.source_coordinates for p in packets):
            raise ValueError('PDB packets must be in the complete bound native source')
        predictions = self(tuple(p.condition for p in packets))
        losses = []
        for packet, prediction in zip(packets, predictions):
            truth = prediction.new_tensor([x for x in packet.masked_targets_angstrom if x is not None])
            losses.append((prediction - truth).square().mean())
        loss = torch.stack(losses).mean()
        return dict(loss=loss, predictions_angstrom=predictions) if return_predictions else loss


def pdb_coordinate_parameter_roles(model):
    if type(model) is not PDBMaskedCoordinates:
        raise ValueError('Exact shared PDB coordinate owner required')
    model.get_extra_state()
    native = _native_roles(model.backbone, model.backbone_identifier_axis)
    identifiers = tuple(f'PDB-native-atom-coordinate:{i}' for i in range(model.axes.max_atoms))
    graph = _native_roles(model.graph, identifiers)
    roles = {f'backbone.{name}': role for name, role in native.items()}
    roles.update({f'graph.{name}': role for name, role in graph.items()})
    hidden = Axis('width', 'hidden')
    raw = json.dumps(['PDB-native-input-v1', model.axes.binding(), 'centered-XYZ', 'observed'],
                     sort_keys=True, separators=(',', ':')).encode()
    finite_input = Axis('finite', hashlib.sha256(raw).hexdigest())
    xyz_axis = json.dumps(['PDB-deposited-frame-angstrom-v1', ['x', 'y', 'z']],
                          separators=(',', ':')).encode()
    xyz = Axis('finite', hashlib.sha256(xyz_axis).hexdigest())
    roles['features.weight'] = ParameterRole('input', (hidden, finite_input))
    roles['readout.weight'] = ParameterRole('output', (xyz, hidden))
    roles['readout.bias'] = ParameterRole('vector', (xyz,))
    _storage_signature(model)
    parameter_ledger(model, roles)
    return roles


def pdb_coordinate_width_family(base, delta, target):
    models = (base, delta, target)
    ledgers = [parameter_ledger(model, pdb_coordinate_parameter_roles(model)) for model in models]
    configuration = _without_member_dimensions(base.get_extra_state())
    if base.width != base.base_width or any(
        not exact_wire(_without_member_dimensions(model.get_extra_state()), configuration) for model in models
    ):
        raise ValueError('PDB source axes, resource policy or base width drift')
    return width_family(*ledgers)
