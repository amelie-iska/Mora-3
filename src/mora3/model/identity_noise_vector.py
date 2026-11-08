"""Diagnostic identity/noise composition; no native inference or admission.

The fixed known atom axis is declared, not authenticated chemistry. Atomic
identity and five explicit unknown-context indicators are constant categorical
features, never neutral charge/spin values. No target coordinates are accepted.
Existing synthetic leaf operators are evaluated as a numerical diagnostic only;
original source metadata and every source/model/training refusal are preserved.
"""
from dataclasses import asdict
import hashlib
import json
import torch
from mora3.data.native_atom_context import NativeAtomContext, encode_native_atom_identity
from mora3.data.independent_current_noise import IndependentCurrentNoise
from mora3.data.native_current_preflight import NativeCurrentMetadataCandidate
from .current_frame_velocity import independent_noise_context_vector_diagnostic

ELEMENTS = ('H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og').split()

IDENTITY_CONTEXT_AXIS = tuple(f'element:{i}' for i in range(1, 119)) + tuple(
    'missing:' + field for field in (
        'electronic_charge', 'spin_multiplicity', 'enzyme', 'cellular_assay', 'solvent'))


def _identity_noise_context_case(*, atom_context, artifact, current_xyz,
        coefficients, binding, atom_axis, free_atoms, width_identity):
    """Compose exact identity declarations and replay-qualified current noise.

    Artifact context SHA must bind canonical UTF-8 sorted compact JSON of the
    complete atom-context dataclass (version included). This is integrity binding,
    not source authentication. No caller-supplied hidden features, desired/gold
    targets, embedding lookup, checkpoint migration or production use is allowed.
    All checks precede learned coefficient calls. XYZ derivatives are local at
    the exact replay value; categorical features have no meaningful derivative.
    """
    if (type(atom_context) is not NativeAtomContext
            or type(artifact) is not IndependentCurrentNoise
            or type(artifact.metadata) is not NativeCurrentMetadataCandidate):
        raise ValueError('exact diagnostic context and replay artifact required')
    # Encoding revalidates public frozen values, including version and unknowns.
    # Do this before JSON serialization of potentially malformed declarations.
    if not isinstance(current_xyz, torch.Tensor):
        raise ValueError('complete replay current tensor required')
    encoded = encode_native_atom_identity(atom_context,
        device=current_xyz.device, dtype=current_xyz.dtype)
    from mora3.data.current_frame_request_binding import CurrentFrameRequestBinding
    if type(binding) is not CurrentFrameRequestBinding:
        raise ValueError('exact current request binding required')
    binding.validate()
    if type(atom_axis) is not tuple or atom_axis != binding.native_axis:
        raise ValueError('exact structured current atom axis required')
    digest = hashlib.sha256(json.dumps(asdict(atom_context), sort_keys=True,
        separators=(',', ':')).encode('utf-8')).hexdigest()
    if (type(atom_axis) is not tuple or atom_context.ordered_atom_ids != tuple(k.name for k in atom_axis)
            or tuple(ELEMENTS[z-1] for z in atom_context.atomic_numbers) != tuple(k.element for k in atom_axis)
            or atom_context.source_native_split != artifact.metadata.native_split
            or digest != artifact.expected_context_sha256):
        raise ValueError('exact ordered identity, native split and context receipt required')
    # Only this explicitly derived diagnostic matrix enters the synthetic leaf;
    # neither NativeAtomContext nor original binding is relabeled synthetic.
    h = torch.cat((encoded.identity, encoded.context_missing.to(encoded.identity.dtype)), dim=1)
    return dict(artifact=artifact,
        current_xyz=current_xyz, coefficients=coefficients, binding=binding,
        atom_axis=atom_axis, context_sha256=digest, free_atoms=free_atoms,
        context=h, context_axis=IDENTITY_CONTEXT_AXIS, origin='synthetic_literal',
        width_identity=width_identity)


def identity_noise_vector_diagnostic(**case):
    """Replay-qualified single-case identity/noise diagnostic; no admission."""
    return independent_noise_context_vector_diagnostic(**_identity_noise_context_case(**case))


def identity_noise_vector_loss(cases, velocities, required_available):
    """Equal-active-case diagnostic loss with detached, loss-only targets.

    Validate every active identity declaration before contextual replay/target
    preflight and learned work. Inactive payloads remain opaque. This preserves
    original unqualified bindings and supplies no native training authority.
    """
    from .current_frame_velocity import independent_noise_context_vector_loss
    if (type(cases) not in (list, tuple) or not cases
            or type(velocities) not in (list, tuple)
            or type(required_available) not in (list, tuple)
            or len(cases) != len(velocities) or len(cases) != len(required_available)
            or any(type(v) is not bool for v in required_available)):
        raise ValueError('complete ordered cases/targets and exact availability required')
    keys = {'atom_context', 'artifact', 'current_xyz', 'coefficients', 'binding',
            'atom_axis', 'free_atoms', 'width_identity'}
    contextual = []
    for case, ready in zip(cases, required_available):
        if not ready:
            contextual.append(None)
            continue
        if type(case) is not dict or set(case) != keys:
            raise ValueError('exact identity/noise diagnostic input fields required')
        contextual.append(_identity_noise_context_case(**case))
    return independent_noise_context_vector_loss(contextual, velocities, required_available)


def prepare_retained_identity_noise(*, binding, seed, manifest_path,
        payload_index, ordinal, expected_record_id, expected_reaction_group,
        expected_native_split, expected_atomic_numbers, payload_path=None):
    """Freshly authenticate retained identity and prepare a noise-only diagnostic.

    Every call rehashes the selected export. No coordinates are decoded. Request
    source identity, ordered names/elements and native split must agree before
    RNG allocation. Group/heldout hashes remain explicit unqualified declarations;
    retained reaction grouping is not corpus leakage or raw-HDF5 authentication.
    Returned kwargs grant no native task, model or training authority.
    """
    from mora3.data.transition1x_atom_identity import read_retained_atom_identity
    from mora3.data.current_frame_request_binding import CurrentFrameRequestBinding
    if type(binding) is not CurrentFrameRequestBinding:
        raise ValueError('exact current request binding required')
    binding.validate()
    retained = read_retained_atom_identity(manifest_path=manifest_path,
        payload_index=payload_index, ordinal=ordinal,
        expected_record_id=expected_record_id,
        expected_reaction_group=expected_reaction_group,
        expected_native_split=expected_native_split,
        expected_atomic_numbers=expected_atomic_numbers, payload_path=payload_path)
    return _prepare_authenticated_identity_noise(retained=retained, binding=binding,
        seed=seed, expected_native_split=expected_native_split)


def _prepare_authenticated_identity_noise(*, retained, binding, seed,
        expected_native_split):
    """Internal composition of a freshly read identity; never a public receipt API."""
    from mora3.data.current_frame_request_binding import CurrentFrameRequestBinding
    from mora3.data.independent_current_noise import construct_independent_current
    if type(binding) is not CurrentFrameRequestBinding:
        raise ValueError('exact current request binding required')
    binding.validate()
    context = retained.context
    source = binding.current.source
    axis = binding.native_axis
    if (source.source_id != context.source_record_id
            or source.original_identity_sha256 != context.source_payload_sha256
            or tuple(k.name for k in axis) != context.ordered_atom_ids
            or tuple(k.element for k in axis) != tuple(ELEMENTS[z-1] for z in context.atomic_numbers)
            or tuple(k.ordinal for k in axis) != tuple(range(len(context.atomic_numbers)))):
        raise ValueError('retained record/source/ordered atom binding mismatch')
    digest = hashlib.sha256(json.dumps(asdict(context), sort_keys=True,
        separators=(',', ':')).encode('utf-8')).hexdigest()
    artifact = construct_independent_current(binding=binding,
        context_sha256=digest, expected_context_sha256=digest, seed=seed,
        native_split=context.source_native_split,
        expected_native_split=expected_native_split)
    return dict(atom_context=context, artifact=artifact, binding=binding,
        atom_axis=axis, free_atoms=binding.free_atoms)


def retained_identity_noise_vector_loss(selections, cases, velocities,
        required_available, *, batch_identity=False):
    """Authenticate all active retained identities before noise diagnostic loss.

    Explicit batch_identity=True reuses authentication only within this call,
    for at most 64 active selections. The default single-reader path is retained.
    Each selection has the single-case preparation arguments. Numerical cases
    contain only current_xyz, coefficients and width_identity: callers cannot
    override authenticated context or source bindings. Every invocation freshly
    rehashes each selected export; no identity receipt cache is accepted.
    Inactive selections, cases and targets stay opaque. Existing ragged preflight
    validates all replay values, owners, masks and detached loss-only targets
    before any learned call. Authentication covers retained export identity only,
    never original HDF5 lineage, task admission or native training authority.
    """
    if (type(selections) not in (list, tuple) or not selections
            or type(cases) not in (list, tuple)
            or type(velocities) not in (list, tuple)
            or type(required_available) not in (list, tuple)
            or not len(selections) == len(cases) == len(velocities) == len(required_available)
            or any(type(v) is not bool for v in required_available)
            or not any(required_available)):
        raise ValueError('complete ordered retained cases/targets and exact availability required')
    if type(batch_identity) is not bool:
        raise ValueError('exact boolean batch_identity required')
    if batch_identity and sum(required_available) > 64:
        raise ValueError('batch preparation supports at most 64 active selections')
    selection_keys = {'binding', 'seed', 'manifest_path', 'payload_index',
        'ordinal', 'expected_record_id', 'expected_reaction_group',
        'expected_native_split', 'expected_atomic_numbers'}
    numerical_keys = {'current_xyz', 'coefficients', 'width_identity'}
    active = []
    for selection, case, ready in zip(selections, cases, required_available):
        if not ready:
            continue
        if (type(selection) is not dict
                or not selection_keys <= set(selection)
                or set(selection) - selection_keys - {'payload_path'}
                or type(case) is not dict or set(case) != numerical_keys):
            raise ValueError('exact retained selection and numerical diagnostic fields required')
        active.append(selection)
    identities = None
    if batch_identity:
        from mora3.data.transition1x_atom_identity import read_retained_atom_identities
        identities = iter(read_retained_atom_identities([
            {k: v for k, v in selection.items() if k not in ('binding', 'seed')}
            for selection in active]))
    prepared = []
    for selection, case, ready in zip(selections, cases, required_available):
        if not ready:
            prepared.append(None)
            continue
        if batch_identity:
            value = _prepare_authenticated_identity_noise(retained=next(identities),
                binding=selection['binding'], seed=selection['seed'],
                expected_native_split=selection['expected_native_split'])
        else:
            value = prepare_retained_identity_noise(**selection)
        prepared.append(dict(value, **case))
    return identity_noise_vector_loss(prepared, velocities, required_available)


def identity_noise_parameter_ledger(cases, required_available):
    """Opt-in finite identity inventory for active ragged diagnostic owners.

    This metadata-only view authenticates neither selections nor numerical
    payloads. It never reads XYZ, runs a forward, changes registration on owners,
    or grants source/model/training authority. Case ordinals name every alias;
    shared owners/weights count once under the existing strict alias ledger.
    All 123 categorical/missingness channels remain finite custom operators,
    so this result cannot qualify a muP width family or scaling prescription.
    """
    from torch import nn
    from .current_frame_velocity import SplitContextualCoefficients
    from .parameterization import parameter_ledger
    if (type(cases) not in (list, tuple) or not cases
            or type(required_available) not in (list, tuple)
            or len(cases) != len(required_available)
            or any(type(v) is not bool for v in required_available)
            or not 1 <= sum(required_available) <= 64):
        raise ValueError('one to 64 active cases and exact ordered availability required')
    root = nn.ModuleDict()
    declarations = {}
    keys = {'current_xyz', 'coefficients', 'width_identity'}
    for i, (case, ready) in enumerate(zip(cases, required_available)):
        if not ready:
            continue
        if (type(case) is not dict or set(case) != keys
                or type(case['coefficients']) is not SplitContextualCoefficients
                or type(case['width_identity']) is not str
                or not case['width_identity'].strip()):
            raise ValueError('exact identity diagnostic numerical fields and owner required')
        owner = case['coefficients']
        roles = owner.identity_parameter_declarations()
        prefix = f'case_{i}'
        root[prefix] = owner
        declarations.update({f'{prefix}.{name}': role for name, role in roles.items()})
    return parameter_ledger(root, declarations)
