"""Opt-in recipe for native ordered TokenGT incidence inputs only.

Pinned muP fork 19814971934ef91dd546f88e913fc963e096d11c:
init.constant_std_init_ and optim.MuAdam/MuSGD. A fixed 2R input and
scalable W output has one infinite axis. This recipe is NOT a complete model
parameterization, optimizer group, checkpoint certificate or transfer result.
Construction is read-only; callers must separately qualify any application.
"""
from dataclasses import dataclass
import math
from .parameterization import tokengt_parameter_ledger


@dataclass(frozen=True)
class NativeIncidenceScalingContract:
    widths: tuple[int, int, int]
    identifier_axis: tuple[str, ...]
    ordered_endpoint_axis: tuple[str, ...]
    aliases: tuple[str, ...]
    finite_axis_sha256: str
    fan_in: int
    dtype: str
    schema: str = 'mora3.native-incidence-input-scaling.v1'

    @property
    def init_std_multiplier(self):
        return 1.0

    @property
    def uniform_bounds(self):
        # Preserve nn.Linear's fan-in Kaiming-uniform(a=sqrt(5)) recipe.
        bound = 1 / math.sqrt(self.fan_in)
        return (-bound, bound)

    @property
    def normal_std(self):
        """Equivalent standard deviation, not a request to change distribution."""
        return 1 / math.sqrt(3 * self.fan_in)

    def lr_multiplier(self, width, *, optimizer):
        """Reference factors only; no optimizer construction or decay policy."""
        if type(width) is not int or width not in self.widths:
            raise ValueError('width must be an exact declared family member')
        if optimizer in ('adam', 'adamw'):
            return 1.0
        if optimizer == 'sgd':
            return width / self.widths[0]
        raise ValueError('explicit supported optimizer required')


def native_incidence_scaling_contract(base, delta, target, *, feature_axis,
                                      identifier_axis):
    """Bind complete fixed ordered axes and native incidence ownership.

    Exact native TokenGT owners only. External wrapper aliases are deliberately
    outside this version. Incompatible internal ties are refused by the existing
    complete native ledger, which remains custom and unusable by width_family.
    No parameter/tensor/RNG/state mutation, initialization or forward is performed.
    Axis names are caller declarations, not source authentication certificates.
    """
    ledgers = tuple(tokengt_parameter_ledger(m, feature_axis=feature_axis,
                        identifier_axis=identifier_axis) for m in (base, delta, target))
    entries = tuple(next(e for e in ledger.parameters if 'incidence.weight' in e.aliases)
                    for ledger in ledgers)
    first = entries[0]
    for entry in entries:
        if (entry.aliases != ('incidence.weight',) or entry.role != 'custom'
                or entry.axes != first.axes or entry.dtype != first.dtype
                or entry.trainable != first.trainable
                or entry.shape[1] != 2 * len(identifier_axis)):
            raise ValueError('native incidence axes, dtype or ownership drift')
    widths = tuple(e.shape[0] for e in entries)
    if not (widths[0] < widths[1] <= widths[2]):
        raise ValueError('delta must widen base; target must not shrink delta')
    return NativeIncidenceScalingContract(widths, identifier_axis,
        tuple(side + ':' + name for side in ('left', 'right') for name in identifier_axis),
        first.aliases, first.axes[1].identity, first.shape[1], first.dtype)


@dataclass(frozen=True)
class IncidenceCallsite:
    """Declared registered module path and its fixed directed input semantics."""
    path: str
    identifier_axis: tuple[str, ...]
    ordered_endpoint_axis: tuple[str, ...]
    role: str = 'ordered-endpoint-incidence'


@dataclass(frozen=True)
class SharedIncidenceScalingContract:
    native: NativeIncidenceScalingContract
    aliases: tuple[tuple[str, ...], ...]
    callsites: tuple[tuple[IncidenceCallsite, ...], ...]
    unique_numel: tuple[int, int, int]
    schema: str = 'mora3.shared-native-incidence-input-scaling.v1'


def shared_incidence_scaling_contract(base, delta, target, *, owner_paths,
                                      feature_axis, identifier_axis, callsites):
    """Bind ALL registered incidence aliases to exact native owners, read-only.

    Each root declares a native TokenGT owner path and a complete immutable set
    of registered incidence module paths with identical directed coordinates.
    Weight-only ties to other modules and bare parameter aliases refuse, even
    when their shapes match. Each family's shared weight counts once.

    This validates registration and declarations, not Python forward control
    flow or non-registering references. It does not certify an arbitrary wrapper,
    qualify custom operators, apply scaling, or authenticate biological axes.
    Rebuild after any owner/registration change; returned values are a snapshot.
    """
    from torch import nn
    roots = (base, delta, target)
    if (type(owner_paths) is not tuple or len(owner_paths) != 3
            or type(callsites) is not tuple or len(callsites) != 3
            or not all(isinstance(root, nn.Module) for root in roots)):
        raise ValueError('three roots, owner paths and immutable callsite groups required')
    owners = []
    for root, path in zip(roots, owner_paths):
        if type(path) is not str or not path:
            raise ValueError('explicit registered native owner path required')
        try:
            owners.append(root.get_submodule(path))
        except (AttributeError, TypeError) as exc:
            raise ValueError('native owner path unavailable') from exc
    native = native_incidence_scaling_contract(*owners, feature_axis=feature_axis,
                                               identifier_axis=identifier_axis)
    aliases = []
    bindings = []
    counts = []
    for root, owner, declared in zip(roots, owners, callsites):
        if (type(declared) is not tuple or not declared
                or not all(type(c) is IncidenceCallsite for c in declared)):
            raise ValueError('immutable exact callsite declarations required')
        paths = tuple(c.path for c in declared)
        if (not all(type(p) is str and p for p in paths)
                or len(set(paths)) != len(paths)):
            raise ValueError('unique registered callsite paths required')
        for c in declared:
            if (type(c.identifier_axis) is not tuple
                    or type(c.ordered_endpoint_axis) is not tuple
                    or c.identifier_axis != native.identifier_axis
                    or c.ordered_endpoint_axis != native.ordered_endpoint_axis
                    or c.role != 'ordered-endpoint-incidence'):
                raise ValueError('conflicting incidence coordinate or role declaration')
        registered = tuple(sorted(name for name, module in
                            root.named_modules(remove_duplicate=False)
                            if module is owner.incidence))
        if tuple(sorted(paths)) != registered:
            raise ValueError('missing or extra registered incidence callsites')
        actual = tuple(sorted(name for name, p in
                              root.named_parameters(remove_duplicate=False)
                              if p is owner.incidence.weight))
        expected = tuple(sorted(p + '.weight' for p in paths))
        if actual != expected:
            raise ValueError('incompatible weight-only or bare incidence alias')
        aliases.append(actual)
        bindings.append(tuple(sorted(declared, key=lambda c: c.path)))
        counts.append(owner.incidence.weight.numel())
    return SharedIncidenceScalingContract(native, tuple(aliases), tuple(bindings), tuple(counts))
