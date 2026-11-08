"""Bounded complete-axis synthetic geometry arithmetic, never source authority.

For one independent local slot: D_ij=||x_i-x_j||_2 and
G_ij=(x_i-mean(x)) dot (x_j-mean(x)). Both are O(3)-invariant;
chirality is deliberately not represented. Both atom axes permute with input
keys. No mixed-frame/reference features, partial XYZ, normalization, source
loading, random draws, task admission or neural invocation are provided.
"""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class SyntheticInternalFeatures:
    values: torch.Tensor
    available: torch.Tensor
    atom_keys: tuple
    reference_keys: tuple
    feature_order: tuple = ('distance', 'centered_dot')

    def require_source_use(self):
        raise ValueError('synthetic features grant no source authority')

    def require_model_use(self):
        raise ValueError('synthetic features grant no model authority')

    def require_training_use(self):
        raise ValueError('synthetic features grant no training authority')


def synthetic_internal_features(xyz, atom_keys, *, origin, max_atoms=128):
    """Return [atom, reference atom, (distance, centered dot)] and full masks.

    Caller origin is a synthetic-use declaration, not a source provenance check.
    The explicit atom budget refuses oversized inputs without slicing. Only
    float32/64 complete finite XYZ are supported. Coincident points use the
    Euclidean norm's zero subgradient. Output masks are all true: unavailable
    coordinates refuse rather than inventing zeros. This leaf is not a typed
    native/model bridge; even manually built results confer no authority.
    """
    if type(origin) is not str or origin != 'synthetic_literal':
        raise ValueError('synthetic literal origin required')
    if type(max_atoms) is not int or not 1 <= max_atoms <= 256:
        raise ValueError('explicit atom budget must be between 1 and 256')
    if (not isinstance(xyz, torch.Tensor) or xyz.ndim != 2
            or xyz.shape[1] != 3 or xyz.dtype not in (torch.float32, torch.float64)
            or not 1 <= xyz.shape[0] <= max_atoms):
        raise ValueError('bounded complete floating XYZ axis required')
    if (type(atom_keys) is not tuple or len(atom_keys) != xyz.shape[0]
            or any(type(k) is not str or not k.strip() for k in atom_keys)
            or len(set(atom_keys)) != len(atom_keys)):
        raise ValueError('complete ordered unique nonempty atom keys required')
    if not bool(torch.isfinite(xyz).all()):
        raise ValueError('complete finite XYZ required')
    centered = xyz - xyz.mean(dim=0)
    distances = torch.linalg.vector_norm(xyz[:, None, :] - xyz[None, :, :], dim=-1)
    dots = centered @ centered.T
    values = torch.stack((distances, dots), dim=-1)
    if not bool(torch.isfinite(values).all()):
        raise ValueError('nonfinite geometry feature arithmetic')
    return SyntheticInternalFeatures(values, torch.ones_like(values, dtype=torch.bool),
                                     atom_keys, atom_keys)


SYNTHETIC_SUMMARY_AXIS = (
    'mean_distance', 'mean_squared_distance', 'mean_squared_centered_dot')


@dataclass(frozen=True)
class SyntheticProjectedAtoms:
    """Lossy reference-axis moments, not a native geometry recipe/authority."""
    finite_features: torch.Tensor
    available: torch.Tensor
    projected: torch.Tensor
    atom_keys: tuple
    feature_order: tuple = SYNTHETIC_SUMMARY_AXIS

    def require_source_use(self):
        raise ValueError('synthetic composition grants no source authority')

    def require_model_use(self):
        raise ValueError('synthetic composition grants no native model authority')

    def require_training_use(self):
        raise ValueError('synthetic composition grants no training authority')


def synthetic_projected_atoms(xyz, atom_keys, sequence_features, projection, *,
                              origin, max_atoms=128):
    """Synthetic-only composition with permutation-safe complete-axis means.

    All N reference atoms, including self, contribute to the denominator N.
    Channels have arbitrary length, squared length and fourth-power length units.
    These three moments lose pair identities and chirality; no quality or source
    eligibility follows. Sequence features are caller-provided synthetic values.
    Gradients propagate through moments and the existing split affine projection.
    """
    from .invariant_atom_projection import InvariantAtomProjection
    if (type(projection) is not InvariantAtomProjection
            or projection.get_extra_state()['finite_axis'] != SYNTHETIC_SUMMARY_AXIS):
        raise ValueError('exact synthetic ordered channel projection required')
    pairs = synthetic_internal_features(xyz, atom_keys, origin=origin, max_atoms=max_atoms)
    distance, dot = pairs.values.unbind(dim=-1)
    features = torch.stack((distance.mean(1), distance.square().mean(1),
                            dot.square().mean(1)), dim=-1)
    # Existing projection checks every input before either affine operation.
    projected = projection(features, sequence_features)
    if not bool(torch.isfinite(projected).all()):
        raise ValueError('nonfinite synthetic projection arithmetic')
    return SyntheticProjectedAtoms(features, torch.ones_like(features,dtype=torch.bool),
                                   projected, atom_keys)


@dataclass(frozen=True)
class SyntheticVelocityLoss:
    """Synthetic scalar fixture only; projected scalars are not vector geometry."""
    loss: torch.Tensor
    active_cases: tuple

    def require_source_use(self):
        raise ValueError('synthetic loss grants no source authority')

    def require_model_use(self):
        raise ValueError('synthetic loss grants no native model authority')

    def require_training_use(self):
        raise ValueError('synthetic loss grants no training authority')


def synthetic_projected_velocity_loss(xyz_cases, atom_keys, sequence_cases,
                                      projection, velocities, free_atoms,
                                      required_available, *, origin, max_atoms=128):
    """Compose existing leaves for a bounded synthetic three-channel loss fixture.

    Preflight selects cases before geometry or affine work; inactive payloads
    remain opaque. Fixed atoms still supply input geometry/context but receive no
    direct velocity supervision. Targets are detached by the existing case loss.
    Three invariant output channels are assigned Cartesian slots ONLY for this
    arithmetic fixture: this is not an equivariant learned velocity decoder.
    No source, native model, training, frame or target-exclusion authority follows.
    """
    from .geometry_case_loss import active_geometry_cases, geometry_case_velocity_loss
    from .invariant_atom_projection import InvariantAtomProjection
    active = active_geometry_cases(free_atoms, required_available)
    if type(origin) is not str or origin != 'synthetic_literal':
        raise ValueError('synthetic literal origin required')
    arrays = (xyz_cases, atom_keys, sequence_cases, velocities)
    if any(type(a) not in (tuple, list) or len(a) != len(free_atoms) for a in arrays):
        raise ValueError('complete ordered synthetic case arrays required')
    if (type(projection) is not InvariantAtomProjection
            or projection.get_extra_state()['width'] != 3):
        raise ValueError('synthetic three-channel projection required')
    predictions = [None] * len(free_atoms)
    for i in active:
        predictions[i] = synthetic_projected_atoms(
            xyz_cases[i], atom_keys[i], sequence_cases[i], projection,
            origin=origin, max_atoms=max_atoms).projected
    loss = geometry_case_velocity_loss(predictions, velocities, free_atoms, required_available)
    return SyntheticVelocityLoss(loss, active)
