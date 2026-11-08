"""Bounded current-frame vector readout, not a native geometry admission path.

Design equation: v_i = free_i/N sum_j Linear(f_ij) (x_j - x_i).
Scalar f_ij are caller-declared invariant, target-free pair features. Their
provenance/invariance cannot be verified here; only synthetic literal diagnostics
are enabled. No reference coordinates, frame registration, gold geometry, path
construction, sampling or training authority is provided. This relative-vector
basis is O(3)-equivariant and cannot express chiral cross-product contributions.
"""
import torch
from torch import nn


class CurrentFrameVelocityReadout(nn.Module):
    def __init__(self, feature_axis: tuple[str, ...], *, max_atoms: int):
        super().__init__()
        if (type(feature_axis) is not tuple or not feature_axis
                or any(type(k) is not str or not k.strip() for k in feature_axis)
                or len(set(feature_axis)) != len(feature_axis)):
            raise ValueError('unique nonempty ordered scalar feature axis required')
        if type(max_atoms) is not int or max_atoms <= 0:
            raise ValueError('positive explicit atom bound required')
        self._feature_axis = feature_axis
        self._max_atoms = max_atoms
        self.coefficients = nn.Linear(len(feature_axis), 1, bias=False)

    def get_extra_state(self):
        return dict(version=1, architecture='current-relative-vector-readout-v1',
                    feature_axis=self._feature_axis, max_atoms=self._max_atoms,
                    frame_kind='current_internal', origin='synthetic_literal',
                    reduction='all-neighbor-mean-including-self',
                    coefficient_bias=False)

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError('current-frame velocity configuration mismatch')

    def require_source_use(self):
        raise ValueError('diagnostic readout grants no source authority')

    def require_model_use(self):
        raise ValueError('diagnostic readout grants no native model authority')

    def require_training_use(self):
        raise ValueError('diagnostic readout grants no training authority')

    def forward_bound_diagnostic(self, current_xyz, pair_features, *, binding,
                                 atom_axis, feature_axis):
        """Revalidate explicit current-input metadata before diagnostic arithmetic.

        Axis labels do not establish feature invariance or source provenance.
        Free masks are explicit diagnostic declarations, not region admission.
        No reference or target-coordinate argument is accepted.
        """
        from mora3.data.current_frame_request_binding import CurrentFrameRequestBinding
        if type(binding) is not CurrentFrameRequestBinding:
            raise ValueError('exact current-frame request binding required')
        binding.require_synthetic_diagnostic()
        if (type(atom_axis) is not tuple or atom_axis!=binding.native_axis
                or type(feature_axis) is not tuple or feature_axis!=self._feature_axis):
            raise ValueError('exact ordered atom and feature axes required')
        if (not isinstance(current_xyz, torch.Tensor)
                or current_xyz.shape!=(len(binding.native_axis),3)):
            raise ValueError('complete bound current XYZ shape required')
        free=torch.tensor(binding.free_atoms,dtype=torch.bool,device=self.coefficients.weight.device)
        return self(current_xyz,pair_features,free,origin='synthetic_literal',frame_kind='current_internal')

    def forward(self, current_xyz, pair_features, free_atoms, *, origin, frame_kind):
        """Return full [N,3] velocity with exact zeros on fixed atoms.

        All atoms including fixed ones supply geometry/context. Complete input
        tensors are validated before learned work; partial axes/NaNs are refused.
        Simultaneous atom permutations must permute both pair axes. No Cartesian
        component is assigned a learned scalar channel. Units of coefficient
        features/time remain caller-declared, not physically calibrated.
        """
        if type(origin) is not str or origin != 'synthetic_literal':
            raise ValueError('synthetic literal diagnostics only')
        if type(frame_kind) is not str or frame_kind != 'current_internal':
            raise ValueError('current_internal frame only; references unsupported')
        x, f, mask = current_xyz, pair_features, free_atoms
        w = self.coefficients.weight
        if (not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] != 3
                or not 1 <= x.shape[0] <= self._max_atoms
                or not isinstance(f, torch.Tensor)
                or f.shape != (x.shape[0], x.shape[0], len(self._feature_axis))
                or x.dtype != w.dtype or f.dtype != w.dtype
                or x.device != w.device or f.device != w.device
                or not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
                or mask.shape != (x.shape[0],) or mask.device != w.device
                or not bool(torch.isfinite(x).all())
                or not bool(torch.isfinite(f).all())):
            raise ValueError('complete finite aligned XYZ/pair/mask arrays matching model required')
        relative = x[None, :, :] - x[:, None, :]
        if not bool(torch.isfinite(relative).all()):
            raise ValueError('nonfinite relative geometry arithmetic')
        scalar = self.coefficients(f)
        velocity = (scalar * relative).mean(dim=1)
        if not bool(torch.isfinite(velocity).all()):
            raise ValueError('nonfinite velocity arithmetic')
        return torch.where(mask[:, None], velocity, torch.zeros_like(velocity))


def bound_current_frame_velocity_loss(xyz_cases, pair_cases, readout, bindings,
                                      atom_axes, feature_axis, velocities,
                                      free_atoms, required_available):
    """Synthetic request-bound ragged vector loss with equal active-case weight.

    Preflight all active inputs/targets before any coefficient call. Inactive
    payloads stay opaque; applicability masks must exactly match active bindings.
    Targets enter only the detached loss, never the current-frame readout.
    This composes diagnostic leaves and grants no native/training authority.
    """
    from mora3.data.current_frame_request_binding import CurrentFrameRequestBinding
    from .geometry_case_loss import active_geometry_cases, geometry_case_velocity_loss
    from .synthetic_geometry_features import SyntheticVelocityLoss
    active = active_geometry_cases(free_atoms, required_available)
    arrays = (xyz_cases, pair_cases, bindings, atom_axes, velocities)
    if any(type(a) not in (tuple, list) or len(a) != len(free_atoms) for a in arrays):
        raise ValueError('complete ordered current-frame case arrays required')
    if type(readout) is not CurrentFrameVelocityReadout:
        raise ValueError('exact current-frame diagnostic readout required')
    if type(feature_axis) is not tuple or feature_axis != readout._feature_axis:
        raise ValueError('exact ordered feature axis required')
    w = readout.coefficients.weight
    for i in active:
        b, x, f, u, mask = bindings[i], xyz_cases[i], pair_cases[i], velocities[i], free_atoms[i]
        if type(b) is not CurrentFrameRequestBinding:
            raise ValueError('exact current-frame binding required')
        b.require_synthetic_diagnostic()
        n = len(b.native_axis)
        if (type(atom_axes[i]) is not tuple or atom_axes[i] != b.native_axis
                or mask.device != w.device or mask.numel() != n
                or tuple(mask.tolist()) != b.free_atoms):
            raise ValueError('ordered bound atoms and exact free mask required')
        if (not 1 <= n <= readout._max_atoms
                or not isinstance(x, torch.Tensor) or x.shape != (n, 3)
                or not isinstance(f, torch.Tensor) or f.shape != (n, n, len(feature_axis))
                or not isinstance(u, torch.Tensor) or u.shape != (n, 3)
                or any(t.dtype != w.dtype or t.device != w.device for t in (x, f, u))
                or not bool(torch.isfinite(x).all())
                or not bool(torch.isfinite(f).all())
                or not bool(torch.isfinite(u.detach()[mask]).all())
                or not bool(torch.isfinite(x[None, :, :] - x[:, None, :]).all())):
            raise ValueError('complete finite aligned current inputs and free targets required')
    predictions = [None] * len(free_atoms)
    for i in active:
        predictions[i] = readout.forward_bound_diagnostic(
            xyz_cases[i], pair_cases[i], binding=bindings[i],
            atom_axis=atom_axes[i], feature_axis=feature_axis)
    loss = geometry_case_velocity_loss(predictions, velocities, free_atoms, required_available)
    return SyntheticVelocityLoss(loss, active)


def invariant_current_frame_velocity_loss(xyz_cases, readout, bindings, atom_axes,
                                          velocities, free_atoms,
                                          required_available, *, origin):
    """Synthetic complete-axis invariant features to request-bound vector loss.

    Pair channels are exclusively distance and centered dot from CURRENT XYZ;
    no reference/target coordinates or caller pair features are accepted. All
    active features are constructed before the bound loss preflight invokes any
    learned coefficients. Quadratic storage and nonchiral O(3) semantics remain;
    this correctness composition grants no source/model/training authority.
    """
    from .geometry_case_loss import active_geometry_cases
    from .synthetic_geometry_features import synthetic_internal_features
    active = active_geometry_cases(free_atoms, required_available)
    if type(origin) is not str or origin != 'synthetic_literal':
        raise ValueError('synthetic literal diagnostics only')
    if (type(readout) is not CurrentFrameVelocityReadout
            or readout._feature_axis != ('distance', 'centered_dot')
            or not 1 <= readout._max_atoms <= 256):
        raise ValueError('exact bounded distance/centered-dot readout required')
    arrays = (xyz_cases, bindings, atom_axes, velocities)
    if any(type(a) not in (tuple, list) or len(a) != len(free_atoms) for a in arrays):
        raise ValueError('complete ordered current-frame case arrays required')
    pairs = [None] * len(free_atoms)
    for i in active:
        pairs[i] = synthetic_internal_features(
            xyz_cases[i], tuple(str(j) for j in range(len(atom_axes[i]))),
            origin=origin, max_atoms=readout._max_atoms).values
    return bound_current_frame_velocity_loss(
        xyz_cases, pairs, readout, bindings, atom_axes,
        ('distance', 'centered_dot'), velocities, free_atoms, required_available)


class ContextualCurrentFrameVelocityReadout(nn.Module):
    """Separate synthetic-only atom-context coefficient version.

    c_ij = a*d_ij + b*g_ij + u dot h_i + v dot h_j;
    velocity_i = free_i * mean_j c_ij*(x_j-x_i).
    h is a COMPLETE caller-declared invariant, target-free scalar context axis.
    This leaf cannot authenticate that declaration. It builds distance/centered
    dot only from current XYZ, and admits no target/reference coordinate input.
    Directed endpoint coefficients need not be symmetric. O(3) equivariance,
    nonchiral relative-vector basis and quadratic pair storage are retained.
    This is an unqualified capacity adaptation, not a native geometry model.
    """
    def __init__(self, context_axis: tuple[str, ...], *, max_atoms: int):
        super().__init__()
        if (type(context_axis) is not tuple or not context_axis
                or any(type(k) is not str or not k.strip() for k in context_axis)
                or len(set(context_axis)) != len(context_axis)):
            raise ValueError('unique nonempty ordered scalar context axis required')
        if type(max_atoms) is not int or not 1 <= max_atoms <= 256:
            raise ValueError('explicit atom bound between 1 and 256 required')
        self._context_axis = context_axis
        self._max_atoms = max_atoms
        self.coefficients = nn.Linear(2 + 2 * len(context_axis), 1, bias=False)

    def parameter_declarations(self, *, width_identity):
        """Inventory packed coefficients; mixed axis explicitly forbids scaling.

        The concatenation contains two finite geometry channels and two ordered
        hidden vectors. A single width axis would misrepresent its extent. Keep
        the whole packed axis finite, role custom, until a split operator has
        separate tested scaling semantics. No checkpoint or operator is changed.
        """
        import hashlib
        import json
        from .parameterization import Axis, ParameterRole, parameter_ledger
        if type(width_identity) is not str or not width_identity.strip():
            raise ValueError('explicit context width identity required')
        if (type(self) is not ContextualCurrentFrameVelocityReadout
                or set(self._modules) != {'coefficients'}
                or self._parameters or self._buffers
                or type(self.coefficients) is not nn.Linear):
            raise ValueError('exact contextual readout ownership required')
        c = self.coefficients
        n = 2 + 2 * len(self._context_axis)
        if (c._modules or c._buffers or set(c._parameters) != {'weight', 'bias'}
                or c.bias is not None or c.in_features != n
                or c.out_features != 1 or tuple(c.weight.shape) != (1, n)):
            raise ValueError('exact bias-free packed coefficient operator required')
        def finite(value):
            raw = json.dumps(value, separators=(',', ':')).encode()
            return Axis('finite', hashlib.sha256(raw).hexdigest())
        packed = finite(['current-context-coefficients.v1', width_identity,
                         ['geometry', 'distance'], ['geometry', 'centered_dot'],
                         ['receiver', self._context_axis],
                         ['neighbor', self._context_axis]])
        roles = {'coefficients.weight': ParameterRole('custom', (
            finite(['current-vector-coefficient-scalar.v1']), packed))}
        parameter_ledger(self, roles)
        return roles

    def get_extra_state(self):
        return dict(version=1, architecture='contextual-current-relative-vector-v1',
                    context_axis=self._context_axis, max_atoms=self._max_atoms,
                    coefficient_order=('distance', 'centered_dot', 'receiver_context', 'neighbor_context'),
                    frame_kind='current_internal', origin='synthetic_literal',
                    reduction='all-neighbor-mean-including-self', coefficient_bias=False,
                    context_semantics='caller-declared-invariant-target-free-scalars')

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError('contextual current-frame configuration mismatch')

    def require_source_use(self):
        raise ValueError('synthetic contextual readout grants no source authority')

    def require_model_use(self):
        raise ValueError('synthetic contextual readout grants no native model authority')

    def require_training_use(self):
        raise ValueError('synthetic contextual readout grants no training authority')

    def _prepare(self, current_xyz, atom_context, *, binding, atom_axis, context_axis):
        """Validate and build current-only inputs with no learned operations."""
        from mora3.data.current_frame_request_binding import CurrentFrameRequestBinding
        from .synthetic_geometry_features import synthetic_internal_features
        if type(binding) is not CurrentFrameRequestBinding:
            raise ValueError('exact current-frame binding required')
        binding.require_synthetic_diagnostic()
        if (type(atom_axis) is not tuple or atom_axis != binding.native_axis
                or type(context_axis) is not tuple or context_axis != self._context_axis):
            raise ValueError('exact ordered atom and context axes required')
        x, h = current_xyz, atom_context
        w = self.coefficients.weight
        n = len(atom_axis)
        if (not isinstance(x, torch.Tensor) or x.shape != (n, 3)
                or not isinstance(h, torch.Tensor) or h.shape != (n, len(context_axis))
                or any(t.dtype != w.dtype or t.device != w.device for t in (x, h))
                or not bool(torch.isfinite(h).all())):
            raise ValueError('complete finite aligned XYZ/context matching model required')
        pairs = synthetic_internal_features(
            x, tuple(str(i) for i in range(n)), origin='synthetic_literal',
            max_atoms=self._max_atoms).values
        scalars = torch.cat((pairs, h[:, None, :].expand(n, n, -1),
                             h[None, :, :].expand(n, n, -1)), dim=-1)
        free = torch.tensor(binding.free_atoms, dtype=torch.bool, device=w.device)
        return x, scalars, free

    def _read_prepared(self, x, scalars, free):
        relative = x[None, :, :] - x[:, None, :]
        velocity = (self.coefficients(scalars) * relative).mean(dim=1)
        if not bool(torch.isfinite(velocity).all()):
            raise ValueError('nonfinite contextual velocity arithmetic')
        return torch.where(free[:, None], velocity, torch.zeros_like(velocity))

    def forward_bound_diagnostic(self, current_xyz, atom_context, *, binding,
                                 atom_axis, context_axis):
        """Bounded current-input diagnostic; no target/reference/source authority."""
        return self._read_prepared(*self._prepare(
            current_xyz, atom_context, binding=binding,
            atom_axis=atom_axis, context_axis=context_axis))


def contextual_current_frame_velocity_loss(xyz_cases, context_cases, readout,
                                          bindings, atom_axes, context_axis,
                                          velocities, free_atoms,
                                          required_available, *, origin):
    """Equal-active-case diagnostic with all-active preflight before learned work.

    All complete atom contexts include fixed atoms. Inactive arrays remain opaque;
    only free targets are checked and detached by the existing loss. Targets do
    not enter feature construction. This grants no source/model/training use.
    """
    from .geometry_case_loss import active_geometry_cases, geometry_case_velocity_loss
    from .synthetic_geometry_features import SyntheticVelocityLoss
    active = active_geometry_cases(free_atoms, required_available)
    if type(origin) is not str or origin != 'synthetic_literal':
        raise ValueError('synthetic literal diagnostics only')
    if type(readout) is not ContextualCurrentFrameVelocityReadout:
        raise ValueError('exact contextual current-frame readout required')
    arrays = (xyz_cases, context_cases, bindings, atom_axes, velocities)
    if any(type(a) not in (tuple, list) or len(a) != len(free_atoms) for a in arrays):
        raise ValueError('complete ordered contextual case arrays required')
    prepared = {}
    for i in active:
        prepared[i] = readout._prepare(
            xyz_cases[i], context_cases[i], binding=bindings[i],
            atom_axis=atom_axes[i], context_axis=context_axis)
        x, _, mask = prepared[i]
        u, supplied = velocities[i], free_atoms[i]
        if (supplied.device != mask.device or supplied.shape != mask.shape
                or not torch.equal(supplied, mask)
                or not isinstance(u, torch.Tensor) or u.shape != x.shape
                or u.dtype != x.dtype or u.device != x.device
                or not bool(torch.isfinite(u.detach()[mask]).all())):
            raise ValueError('exact bound free mask and finite free targets required')
    predictions = [None] * len(free_atoms)
    for i in active:
        predictions[i] = readout._read_prepared(*prepared[i])
    loss = geometry_case_velocity_loss(predictions, velocities, free_atoms, required_available)
    return SyntheticVelocityLoss(loss, active)


def projected_context_current_frame_velocity_loss(
        xyz_cases, sequence_cases, projection, readout, bindings, atom_axes,
        context_axis, velocities, free_atoms, required_available, *, origin,
        width_identity):
    """Synthetic current-only invariant moments -> atom context -> vector loss.

    h_i = finite(moment_i) + sequence(s_i), with ordered complete hidden axes.
    Moments are mean distance, mean squared distance and mean squared centered
    dot over ALL atoms, including self/fixed atoms. They lose pair identity and
    chirality. Sequence scalars are synthetic caller inputs, not authenticated
    native sequence conditioning. Targets enter only the detached case loss.

    All active cases preflight before either projection or coefficient call.
    This diagnostic composes existing leaves without changing their checkpoints
    or granting native source/model/training authority. Pair storage is quadratic;
    hidden width and <=256 atom limits are explicit, never truncated.
    """
    from .geometry_case_loss import active_geometry_cases, geometry_case_velocity_loss
    from .invariant_atom_projection import InvariantAtomProjection
    from .synthetic_geometry_features import (
        SYNTHETIC_SUMMARY_AXIS, SyntheticVelocityLoss, synthetic_internal_features)
    active = active_geometry_cases(free_atoms, required_available)
    if type(origin) is not str or origin != 'synthetic_literal':
        raise ValueError('synthetic literal diagnostics only')
    if (type(projection) is not InvariantAtomProjection
            or type(readout) is not ContextualCurrentFrameVelocityReadout
            or projection._finite_axis != SYNTHETIC_SUMMARY_AXIS
            or type(context_axis) is not tuple or context_axis != readout._context_axis
            or len(context_axis) != projection._width
            or type(width_identity) is not str or not width_identity
            or width_identity != projection._width_identity):
        raise ValueError('exact summary projection and ordered bound hidden axis required')
    projection.parameter_declarations()
    weight = projection.finite.weight
    if any(t.dtype != weight.dtype or t.device != weight.device
           or not bool(torch.isfinite(t).all())
           for t in tuple(projection.parameters()) + tuple(readout.parameters())):
        raise ValueError('finite aligned projection/readout parameters required')
    arrays = (xyz_cases, sequence_cases, bindings, atom_axes, velocities)
    if any(type(a) not in (tuple, list) or len(a) != len(free_atoms) for a in arrays):
        raise ValueError('complete ordered projected-context case arrays required')
    prepared = {}
    for i in active:
        x, s, u, mask = xyz_cases[i], sequence_cases[i], velocities[i], free_atoms[i]
        # No learned operation: use a zero context solely for existing binding,
        # whole-XYZ, ordered-axis and current geometry preflight.
        n = x.shape[0] if isinstance(x, torch.Tensor) and x.ndim == 2 else 0
        _, _, bound_mask = readout._prepare(
            x, weight.new_zeros((n, projection._width)), binding=bindings[i],
            atom_axis=atom_axes[i], context_axis=context_axis)
        if (not isinstance(s, torch.Tensor) or s.shape != (n, projection._width)
                or s.dtype != weight.dtype or s.device != weight.device
                or not bool(torch.isfinite(s).all())
                or mask.device != bound_mask.device or mask.shape != bound_mask.shape
                or not torch.equal(mask, bound_mask)
                or not isinstance(u, torch.Tensor) or u.shape != x.shape
                or u.dtype != weight.dtype or u.device != weight.device
                or not bool(torch.isfinite(u.detach()[mask]).all())):
            raise ValueError('complete aligned sequence, exact free mask and finite free targets required')
        pairs = synthetic_internal_features(
            x, tuple(str(j) for j in range(n)), origin=origin,
            max_atoms=readout._max_atoms).values
        d, g = pairs.unbind(dim=-1)
        moments = torch.stack((d.mean(1), d.square().mean(1),
                               g.square().mean(1)), dim=-1)
        if not bool(torch.isfinite(moments).all()):
            raise ValueError('nonfinite invariant summary arithmetic')
        prepared[i] = moments
    predictions = [None] * len(free_atoms)
    for i in active:
        h = projection(prepared[i], sequence_cases[i])
        predictions[i] = readout.forward_bound_diagnostic(
            xyz_cases[i], h, binding=bindings[i], atom_axis=atom_axes[i],
            context_axis=context_axis)
    return SyntheticVelocityLoss(geometry_case_velocity_loss(
        predictions, velocities, free_atoms, required_available), active)


class SplitContextualCoefficients(nn.Module):
    """Synthetic-only split of the packed contextual scalar equation.

    c_ij = geometry(g_ij) + receiver(h_i) + neighbor(h_j).
    Finite geometry and hidden context have separate weights; no scaling rule,
    initialization equivalence, source invariance or native integration is implied.
    Checkpoints deliberately use distinct names and configuration from v1.
    """
    def __init__(self, context_axis: tuple[str, ...], *, max_atoms: int):
        super().__init__()
        if (type(context_axis) is not tuple or not context_axis
                or any(type(k) is not str or not k.strip() for k in context_axis)
                or len(set(context_axis)) != len(context_axis)):
            raise ValueError('unique nonempty ordered context axis required')
        if type(max_atoms) is not int or not 1 <= max_atoms <= 256:
            raise ValueError('explicit atom bound between 1 and 256 required')
        self._context_axis = context_axis
        self._max_atoms = max_atoms
        self.geometry = nn.Linear(2, 1, bias=False)
        self.receiver = nn.Linear(len(context_axis), 1, bias=False)
        self.neighbor = nn.Linear(len(context_axis), 1, bias=False)

    def parameter_declarations(self, *, width_identity):
        """Strict separate finite/hidden inventory; no scaling qualification.

        Receiver and neighbor share the same ordered context semantics but remain
        independent parameter owners unless explicitly tied by the caller.
        Geometry is a fixed finite-to-finite custom operator.
        """
        import hashlib
        import json
        from .parameterization import Axis, ParameterRole, parameter_ledger
        if type(width_identity) is not str or not width_identity.strip():
            raise ValueError('explicit context width identity required')
        if (type(self) is not SplitContextualCoefficients
                or set(self._modules) != {'geometry', 'receiver', 'neighbor'}
                or self._parameters or self._buffers):
            raise ValueError('exact split coefficient ownership required')
        for name, size in (('geometry', 2), ('receiver', len(self._context_axis)),
                           ('neighbor', len(self._context_axis))):
            c = getattr(self, name)
            if (type(c) is not nn.Linear or c._modules or c._buffers
                    or set(c._parameters) != {'weight', 'bias'}
                    or c.bias is not None or c.in_features != size
                    or c.out_features != 1 or tuple(c.weight.shape) != (1, size)):
                raise ValueError('exact bias-free split coefficient operator required')
        def finite(value):
            raw = json.dumps(value, separators=(',', ':')).encode()
            return Axis('finite', hashlib.sha256(raw).hexdigest())
        scalar = finite(['current-vector-coefficient-scalar.v1'])
        geometry = finite(['split-context-geometry.v1', 'distance', 'centered_dot'])
        # Bind channel order as well as the caller's shared width identity.
        # Finite scalar identity carries that binding without inventing separate
        # receiver/neighbor width families for the same context tensor.
        context_scalar = finite(['split-context-scalar.v1', self._context_axis])
        output = ParameterRole('output', (context_scalar, Axis('width', width_identity)))
        roles = {'geometry.weight': ParameterRole('custom', (scalar, geometry)),
                 'receiver.weight': output, 'neighbor.weight': output}
        parameter_ledger(self, roles)
        return roles

    def identity_parameter_declarations(self):
        """Opt-in fixed 118-element/five-unknown-channel inventory only.

        These diagnostic categorical inputs are finite, not scalable hidden
        coordinates. Reuse the strict local ownership checks without changing
        the original hidden-context inventory, equations or checkpoint state.
        Custom roles deliberately grant no scaling or source/training authority.
        """
        import hashlib
        import json
        from .identity_noise_vector import IDENTITY_CONTEXT_AXIS
        from .parameterization import Axis, ParameterRole, parameter_ledger
        if (type(self._context_axis) is not tuple
                or self._context_axis != IDENTITY_CONTEXT_AXIS):
            raise ValueError('exact ordered native identity diagnostic channels required')
        roles = self.parameter_declarations(width_identity='identity-inventory-check-only')
        raw = json.dumps(['current-vector-atom-identity-context.v1',
                          self._context_axis], separators=(',', ':')).encode()
        context = Axis('finite', hashlib.sha256(raw).hexdigest())
        scalar = roles['geometry.weight'].axes[0]
        for name in ('receiver.weight', 'neighbor.weight'):
            roles[name] = ParameterRole('custom', (scalar, context))
        parameter_ledger(self, roles)
        return roles

    def get_extra_state(self):
        return dict(version=1, architecture='synthetic-split-context-coefficients-v1',
                    context_axis=self._context_axis, max_atoms=self._max_atoms,
                    geometry_axis=('distance', 'centered_dot'),
                    context_semantics='caller-declared-invariant-target-free',
                    origin='synthetic_literal', bias=False)

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError('split contextual coefficient configuration mismatch')

    def require_source_use(self):
        raise ValueError('synthetic split coefficients grant no source authority')

    def require_model_use(self):
        raise ValueError('synthetic split coefficients grant no native model authority')

    def require_training_use(self):
        raise ValueError('synthetic split coefficients grant no training authority')

    def forward(self, geometry, context, *, origin, context_axis):
        """Complete finite [N,N,2] and [N,W] inputs; no target argument.

        Receiver and neighbor projections precede broadcasting, avoiding packed
        hidden pair concatenation. No measured performance claim is made here.
        """
        g, h, w = geometry, context, self.geometry.weight
        if (type(origin) is not str or origin != 'synthetic_literal'
                or type(context_axis) is not tuple or context_axis != self._context_axis
                or not isinstance(h, torch.Tensor) or h.ndim != 2
                or not 1 <= h.shape[0] <= self._max_atoms
                or h.shape[1] != len(self._context_axis)
                or not isinstance(g, torch.Tensor)
                or g.shape != (h.shape[0], h.shape[0], 2)
                or any(t.dtype != w.dtype or t.device != w.device for t in (g, h))
                or not bool(torch.isfinite(g).all()) or not bool(torch.isfinite(h).all())):
            raise ValueError('complete finite aligned synthetic geometry/context required')
        c = self.geometry(g) + self.receiver(h)[:, None, :] + self.neighbor(h)[None, :, :]
        if not bool(torch.isfinite(c).all()):
            raise ValueError('nonfinite split coefficient arithmetic')
        return c


def split_context_current_frame_velocity_loss(
        xyz_cases, context_cases, coefficients, bindings, atom_axes, context_axis,
        velocities, free_atoms, required_available, *, origin, width_identity):
    """Synthetic request-bound split coefficients -> ragged relative vectors.

    Separate geometry/receiver/neighbor operators retain the existing split-v1
    checkpoint and exact equation. All active requests, complete inputs, operator
    ownership and free targets preflight before ANY learned operation. Targets
    enter only detached equal-active-case loss. Fixed atoms supply geometry and
    context; inactive payloads remain opaque. Caller contexts are not verified
    invariant native features. This function grants no native/training authority,
    width scaling, initialization migration or measured performance improvement.
    """
    from mora3.data.current_frame_request_binding import CurrentFrameRequestBinding
    from .geometry_case_loss import active_geometry_cases, geometry_case_velocity_loss
    from .synthetic_geometry_features import SyntheticVelocityLoss, synthetic_internal_features
    active = active_geometry_cases(free_atoms, required_available)
    if type(origin) is not str or origin != 'synthetic_literal':
        raise ValueError('synthetic literal diagnostics only')
    if (type(coefficients) is not SplitContextualCoefficients
            or type(context_axis) is not tuple
            or context_axis != coefficients._context_axis):
        raise ValueError('exact split coefficients and ordered context axis required')
    coefficients.parameter_declarations(width_identity=width_identity)
    weight = coefficients.geometry.weight
    if any(t.dtype != weight.dtype or t.device != weight.device
           or not bool(torch.isfinite(t).all()) for t in coefficients.parameters()):
        raise ValueError('finite aligned split parameters required')
    arrays = (xyz_cases, context_cases, bindings, atom_axes, velocities)
    if any(type(a) not in (tuple, list) or len(a) != len(free_atoms) for a in arrays):
        raise ValueError('complete ordered split-context case arrays required')
    prepared = {}
    for i in active:
        binding, axis = bindings[i], atom_axes[i]
        if type(binding) is not CurrentFrameRequestBinding:
            raise ValueError('exact current-frame request binding required')
        binding.require_synthetic_diagnostic()
        if type(axis) is not tuple or axis != binding.native_axis:
            raise ValueError('exact ordered bound atom axis required')
        n = len(axis)
        x, h, u, mask = xyz_cases[i], context_cases[i], velocities[i], free_atoms[i]
        bound_mask = torch.tensor(binding.free_atoms, dtype=torch.bool, device=weight.device)
        if (not isinstance(x, torch.Tensor) or x.shape != (n, 3)
                or not isinstance(h, torch.Tensor) or h.shape != (n, len(context_axis))
                or not isinstance(u, torch.Tensor) or u.shape != (n, 3)
                or any(t.dtype != weight.dtype or t.device != weight.device for t in (x, h, u))
                or mask.device != bound_mask.device or mask.shape != bound_mask.shape
                or not torch.equal(mask, bound_mask)
                or not bool(torch.isfinite(h).all())
                or not bool(torch.isfinite(u.detach()[mask]).all())):
            raise ValueError('complete aligned inputs, bound free mask and finite free targets required')
        pairs = synthetic_internal_features(
            x, tuple(str(j) for j in range(n)), origin=origin,
            max_atoms=coefficients._max_atoms).values
        relative = x[None, :, :] - x[:, None, :]
        if not bool(torch.isfinite(relative).all()):
            raise ValueError('nonfinite relative geometry arithmetic')
        prepared[i] = (pairs, h, relative, mask)
    predictions = [None] * len(free_atoms)
    for i in active:
        pairs, h, relative, mask = prepared[i]
        scalar = coefficients(pairs, h, origin=origin, context_axis=context_axis)
        velocity = (scalar * relative).mean(dim=1)
        if not bool(torch.isfinite(velocity).all()):
            raise ValueError('nonfinite split contextual velocity arithmetic')
        predictions[i] = torch.where(mask[:, None], velocity, torch.zeros_like(velocity))
    return SyntheticVelocityLoss(geometry_case_velocity_loss(
        predictions, velocities, free_atoms, required_available), active)


def projected_split_context_current_frame_velocity_loss(
        xyz_cases, sequence_cases, projection, coefficients, bindings, atom_axes, context_axis,
        velocities, free_atoms, required_available, *, origin, width_identity):
    """Synthetic invariant moments/projection -> split coefficients -> vector loss.

    Complete current-XYZ moments lose chirality and pair identities. Sequence
    inputs are caller-declared synthetic invariant scalars. Every active case
    and both exact parameter owners preflight before learned work; targets are
    detached loss-only. No native source/training/scaling authority is granted.
    Checkpoints/defaults of the existing leaves remain unchanged.
    """
    from mora3.data.current_frame_request_binding import CurrentFrameRequestBinding
    from .geometry_case_loss import active_geometry_cases, geometry_case_velocity_loss
    from .synthetic_geometry_features import SyntheticVelocityLoss, synthetic_internal_features
    from .invariant_atom_projection import InvariantAtomProjection
    from .synthetic_geometry_features import SYNTHETIC_SUMMARY_AXIS
    active = active_geometry_cases(free_atoms, required_available)
    if type(origin) is not str or origin != 'synthetic_literal':
        raise ValueError('synthetic literal diagnostics only')
    if (type(coefficients) is not SplitContextualCoefficients
            or type(context_axis) is not tuple
            or context_axis != coefficients._context_axis):
        raise ValueError('exact split coefficients and ordered context axis required')
    if (type(projection) is not InvariantAtomProjection
            or projection._finite_axis != SYNTHETIC_SUMMARY_AXIS
            or projection._width != len(context_axis)
            or projection._width_identity != width_identity):
        raise ValueError('exact summary projection and bound hidden width required')
    projection.parameter_declarations()
    coefficients.parameter_declarations(width_identity=width_identity)
    weight = coefficients.geometry.weight
    if any(t.dtype != weight.dtype or t.device != weight.device
           or not bool(torch.isfinite(t).all()) for t in tuple(coefficients.parameters()) + tuple(projection.parameters())):
        raise ValueError('finite aligned split parameters required')
    arrays = (xyz_cases, sequence_cases, bindings, atom_axes, velocities)
    if any(type(a) not in (tuple, list) or len(a) != len(free_atoms) for a in arrays):
        raise ValueError('complete ordered split-context case arrays required')
    prepared = {}
    for i in active:
        binding, axis = bindings[i], atom_axes[i]
        if type(binding) is not CurrentFrameRequestBinding:
            raise ValueError('exact current-frame request binding required')
        binding.require_synthetic_diagnostic()
        if type(axis) is not tuple or axis != binding.native_axis:
            raise ValueError('exact ordered bound atom axis required')
        n = len(axis)
        x, h, u, mask = xyz_cases[i], sequence_cases[i], velocities[i], free_atoms[i]
        bound_mask = torch.tensor(binding.free_atoms, dtype=torch.bool, device=weight.device)
        if (not isinstance(x, torch.Tensor) or x.shape != (n, 3)
                or not isinstance(h, torch.Tensor) or h.shape != (n, len(context_axis))
                or not isinstance(u, torch.Tensor) or u.shape != (n, 3)
                or any(t.dtype != weight.dtype or t.device != weight.device for t in (x, h, u))
                or mask.device != bound_mask.device or mask.shape != bound_mask.shape
                or not torch.equal(mask, bound_mask)
                or not bool(torch.isfinite(h).all())
                or not bool(torch.isfinite(u.detach()[mask]).all())):
            raise ValueError('complete aligned inputs, bound free mask and finite free targets required')
        pairs = synthetic_internal_features(
            x, tuple(str(j) for j in range(n)), origin=origin,
            max_atoms=coefficients._max_atoms).values
        relative = x[None, :, :] - x[:, None, :]
        if not bool(torch.isfinite(relative).all()):
            raise ValueError('nonfinite relative geometry arithmetic')
        d, g = pairs.unbind(dim=-1)
        moments = torch.stack((d.mean(1), d.square().mean(1),
                               g.square().mean(1)), dim=-1)
        if not bool(torch.isfinite(moments).all()):
            raise ValueError('nonfinite invariant summary arithmetic')
        prepared[i] = (pairs, h, moments, relative, mask)
    predictions = [None] * len(free_atoms)
    for i in active:
        pairs, sequence, moments, relative, mask = prepared[i]
        h = projection(moments, sequence)
        scalar = coefficients(pairs, h, origin=origin, context_axis=context_axis)
        velocity = (scalar * relative).mean(dim=1)
        if not bool(torch.isfinite(velocity).all()):
            raise ValueError('nonfinite split contextual velocity arithmetic')
        predictions[i] = torch.where(mask[:, None], velocity, torch.zeros_like(velocity))
    return SyntheticVelocityLoss(geometry_case_velocity_loss(
        predictions, velocities, free_atoms, required_available), active)


def _prepare_independent_noise_current_vector(*, artifact, current_xyz,
        readout, binding, atom_axis, context_sha256, free_atoms):
    """Replay-verified generated noise to a diagnostic current-frame vector.

    The native binding supplies a declared fixed known atom axis, never observed
    coordinates or authenticated biological context. Its authority is unchanged.
    Only recipe-generated XYZ drive synthetic arithmetic; no source coordinates,
    external pair features, targets or native model/training use are admitted.
    A caller replay copy permits local XYZ derivatives, but any changed value
    must refuse before learned work. This is not a sampler or trained prior.
    Existing leaf defaults and checkpoint formats remain unchanged.
    """
    from mora3.data.independent_current_noise import (
        IndependentCurrentNoise, replay_independent_current)
    from mora3.data.current_frame_request_binding import CurrentFrameRequestBinding
    from mora3.data.native_current_preflight import NativeCurrentMetadataCandidate
    from .synthetic_geometry_features import synthetic_internal_features
    if (type(artifact) is not IndependentCurrentNoise
            or type(artifact.metadata) is not NativeCurrentMetadataCandidate
            or type(binding) is not CurrentFrameRequestBinding
            or type(readout) is not CurrentFrameVelocityReadout):
        raise ValueError('exact noise artifact, binding and diagnostic readout required')
    binding.validate()
    if (binding != artifact.metadata.binding
            or type(atom_axis) is not tuple or atom_axis != binding.native_axis
            or type(context_sha256) is not str
            or context_sha256 != artifact.expected_context_sha256
            or type(free_atoms) is not tuple or free_atoms != binding.free_atoms
            or any(type(v) is not bool for v in free_atoms)):
        raise ValueError('complete ordered request/atom/context/free receipts required')
    if (readout._feature_axis != ('distance', 'centered_dot')
            or type(readout._max_atoms) is not int
            or not len(atom_axis) <= readout._max_atoms <= 256
            or set(readout._modules) != {'coefficients'}
            or readout._parameters or readout._buffers
            or type(readout.coefficients) is not nn.Linear):
        raise ValueError('exact bounded distance/centered-dot parameter owner required')
    c = readout.coefficients
    if (c._modules or c._buffers or set(c._parameters) != {'weight', 'bias'}
            or c.bias is not None or c.in_features != 2 or c.out_features != 1
            or tuple(c.weight.shape) != (1, 2)
            or c.weight.dtype not in (torch.float32, torch.float64)
            or not bool(torch.isfinite(c.weight).all())):
        raise ValueError('finite exact bias-free coefficient operator required')
    w = c.weight
    if (not isinstance(current_xyz, torch.Tensor)
            or current_xyz.shape != (len(atom_axis), 3)
            or current_xyz.dtype != w.dtype or current_xyz.device != w.device):
        raise ValueError('complete aligned replay copy required')
    replay = replay_independent_current(artifact, device=w.device, dtype=w.dtype)
    if not torch.equal(current_xyz.detach(), replay):
        raise ValueError('current XYZ differs from complete independent-noise replay')
    pairs = synthetic_internal_features(current_xyz,
        tuple(str(i) for i in range(len(atom_axis))),
        origin='synthetic_literal', max_atoms=readout._max_atoms).values
    mask = torch.tensor(free_atoms, device=w.device, dtype=torch.bool)
    return current_xyz, pairs, mask



def independent_noise_current_vector_diagnostic(*, artifact, current_xyz,
        readout, binding, atom_axis, context_sha256, free_atoms):
    """Replay-verified generated noise only; no native model/training authority."""
    prepared = _prepare_independent_noise_current_vector(
        artifact=artifact, current_xyz=current_xyz, readout=readout,
        binding=binding, atom_axis=atom_axis, context_sha256=context_sha256,
        free_atoms=free_atoms)
    return readout(*prepared, origin='synthetic_literal', frame_kind='current_internal')


def independent_noise_current_vector_loss(cases, velocities, required_available):
    """Equal active-case diagnostic loss, with all receipts checked before learning.

    Each active case is an exact dictionary of the single-vector diagnostic's
    keyword inputs. Inactive case/target payloads remain opaque. Targets are
    detached and loss-only; this neither admits source coordinates nor training.
    No checkpoint/default, mask policy or noise recipe changes are implied.
    """
    from .geometry_case_loss import active_geometry_cases, geometry_case_velocity_loss
    from .synthetic_geometry_features import SyntheticVelocityLoss
    if (type(cases) not in (list, tuple) or not cases
            or type(velocities) not in (list, tuple)
            or type(required_available) not in (list, tuple)
            or len(cases) != len(velocities) or len(cases) != len(required_available)
            or any(type(v) is not bool for v in required_available)):
        raise ValueError('complete ordered cases/targets and exact availability required')
    keys = {'artifact', 'current_xyz', 'readout', 'binding', 'atom_axis',
            'context_sha256', 'free_atoms'}
    prepared = {}
    masks = []
    signature = None
    for i, ready in enumerate(required_available):
        if not ready:
            masks.append(torch.zeros(1, dtype=torch.bool))
            continue
        case = cases[i]
        if type(case) is not dict or set(case) != keys:
            raise ValueError('exact diagnostic input fields required')
        x, pairs, mask = _prepare_independent_noise_current_vector(**case)
        u = velocities[i]
        if (not isinstance(u, torch.Tensor) or u.shape != x.shape
                or u.dtype != x.dtype or u.device != x.device
                or not bool(torch.isfinite(u.detach()[mask]).all())):
            raise ValueError('complete aligned finite free velocity targets required')
        current = (x.dtype, x.device)
        if signature is not None and signature != current:
            raise ValueError('active cases must share dtype/device')
        signature = current
        prepared[i] = (x, pairs, mask)
        masks.append(mask)
    active = active_geometry_cases(masks, required_available)
    predictions = [None] * len(cases)
    for i in active:
        predictions[i] = cases[i]['readout'](*prepared[i],
            origin='synthetic_literal', frame_kind='current_internal')
    return SyntheticVelocityLoss(geometry_case_velocity_loss(
        predictions, velocities, masks, required_available), active)


def _prepare_independent_noise_context_vector(*, artifact, current_xyz,
        coefficients, binding, atom_axis, context_sha256, free_atoms,
        context, context_axis, origin, width_identity):
    """Replay-only noise with caller-declared synthetic invariant atom context.

    Context is not authenticated native chemistry. The original unqualified
    binding is retained without promotion; no source/model/training authority,
    targets, checkpoint migration or production default is supplied here.
    """
    from mora3.data.independent_current_noise import IndependentCurrentNoise, replay_independent_current
    from mora3.data.native_current_preflight import NativeCurrentMetadataCandidate
    from mora3.data.current_frame_request_binding import CurrentFrameRequestBinding
    from .synthetic_geometry_features import synthetic_internal_features
    if (type(artifact) is not IndependentCurrentNoise
            or type(artifact.metadata) is not NativeCurrentMetadataCandidate
            or type(binding) is not CurrentFrameRequestBinding
            or type(coefficients) is not SplitContextualCoefficients):
        raise ValueError('exact artifact, binding and split coefficients required')
    binding.validate()
    if (binding != artifact.metadata.binding
            or type(atom_axis) is not tuple or atom_axis != binding.native_axis
            or type(context_sha256) is not str or context_sha256 != artifact.expected_context_sha256
            or type(free_atoms) is not tuple or free_atoms != binding.free_atoms
            or any(type(v) is not bool for v in free_atoms)
            or type(origin) is not str or origin != 'synthetic_literal'
            or type(context_axis) is not tuple or context_axis != coefficients._context_axis
            or type(coefficients._max_atoms) is not int
            or not len(atom_axis) <= coefficients._max_atoms <= 256):
        raise ValueError('complete ordered receipts and synthetic context required')
    coefficients.parameter_declarations(width_identity=width_identity)
    w = coefficients.geometry.weight
    if (w.dtype not in (torch.float32, torch.float64)
            or any(p.dtype != w.dtype or p.device != w.device
                   or not bool(torch.isfinite(p).all()) for p in coefficients.parameters())
            or not isinstance(current_xyz, torch.Tensor)
            or current_xyz.shape != (len(atom_axis), 3)
            or not isinstance(context, torch.Tensor)
            or context.shape != (len(atom_axis), len(context_axis))
            or any(t.dtype != w.dtype or t.device != w.device for t in (current_xyz, context))
            or not bool(torch.isfinite(context).all())):
        raise ValueError('finite complete aligned context, XYZ and parameters required')
    replay = replay_independent_current(artifact, device=w.device, dtype=w.dtype)
    if not torch.equal(current_xyz.detach(), replay):
        raise ValueError('current XYZ differs from independent-noise replay')
    pairs = synthetic_internal_features(current_xyz,
        tuple(str(i) for i in range(len(atom_axis))), origin=origin,
        max_atoms=coefficients._max_atoms).values
    relative = current_xyz[None, :, :] - current_xyz[:, None, :]
    if not bool(torch.isfinite(relative).all()):
        raise ValueError('nonfinite relative geometry')
    mask = torch.tensor(free_atoms, device=w.device, dtype=torch.bool)
    return pairs, relative, mask


def independent_noise_context_vector_diagnostic(**case):
    """Replay-qualified single-case synthetic contextual vector diagnostic."""
    pairs, relative, mask = _prepare_independent_noise_context_vector(**case)
    return _evaluate_noise_context_vector(case, pairs, relative, mask)


def _evaluate_noise_context_vector(case, pairs, relative, mask):
    scalar = case['coefficients'](pairs, case['context'],
        origin=case['origin'], context_axis=case['context_axis'])
    velocity = (scalar * relative).mean(dim=1)
    if not bool(torch.isfinite(velocity).all()):
        raise ValueError('nonfinite contextual vector')
    return torch.where(mask[:, None], velocity, torch.zeros_like(velocity))


def independent_noise_context_vector_loss(cases, velocities, required_available):
    """Equal-case synthetic loss; preflight every active case before learning.

    Context invariance is a caller declaration. Targets are detached loss-only
    inputs; this diagnostic grants no native source, model or training authority.
    """
    from .geometry_case_loss import active_geometry_cases, geometry_case_velocity_loss
    from .synthetic_geometry_features import SyntheticVelocityLoss
    if (type(cases) not in (list, tuple) or not cases
            or type(velocities) not in (list, tuple)
            or type(required_available) not in (list, tuple)
            or len(cases) != len(velocities) or len(cases) != len(required_available)
            or any(type(v) is not bool for v in required_available)):
        raise ValueError('complete ordered cases/targets and exact availability required')
    keys = {'artifact', 'current_xyz', 'coefficients', 'binding', 'atom_axis',
            'context_sha256', 'free_atoms', 'context', 'context_axis', 'origin',
            'width_identity'}
    prepared = {}
    masks = []
    signature = None
    for i, ready in enumerate(required_available):
        if not ready:
            masks.append(torch.zeros(1, dtype=torch.bool))
            continue
        case = cases[i]
        if type(case) is not dict or set(case) != keys:
            raise ValueError('exact contextual diagnostic input fields required')
        prepared[i] = _prepare_independent_noise_context_vector(**case)
        mask = prepared[i][2]
        x = case['current_xyz']
        u = velocities[i]
        if (not isinstance(u, torch.Tensor) or u.shape != x.shape
                or u.dtype != x.dtype or u.device != x.device
                or not bool(torch.isfinite(u.detach()[mask]).all())):
            raise ValueError('aligned finite free velocity targets required')
        current = (x.dtype, x.device)
        if signature is not None and signature != current:
            raise ValueError('active cases must share dtype/device')
        signature = current
        masks.append(mask)
    active = active_geometry_cases(masks, required_available)
    predictions = [None] * len(cases)
    for i in active:
        predictions[i] = _evaluate_noise_context_vector(cases[i], *prepared[i])
    return SyntheticVelocityLoss(geometry_case_velocity_loss(
        predictions, velocities, masks, required_available), active)
