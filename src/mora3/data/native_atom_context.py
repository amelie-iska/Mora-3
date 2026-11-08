"""Versioned metadata-only native atom identity; never a model admission receipt.

Caller supplies the source payload identity and ordered atom IDs. These declarations
bind a view, but do not authenticate source lineage or qualify neural predictors.
Coordinates/energies/forces are deliberately never inspected. Version 1 supports
unknown electronic/environment context only; richer context needs separate review.
"""
from dataclasses import dataclass
from collections.abc import Mapping


@dataclass(frozen=True)
class NativeAtomContext:
    version: str
    source_record_id: str
    source_reaction_group: str
    source_native_split: str
    source_payload_sha256: str
    ordered_atom_ids: tuple[str, ...]
    atomic_numbers: tuple[int, ...]
    electronic_charge: None = None
    spin_multiplicity: None = None
    missing_context: tuple[str, ...] = (
        'electronic_charge', 'spin_multiplicity', 'enzyme', 'cellular_assay', 'solvent')

    def require_model_use(self):
        """No instance of this metadata contract grants model/task admission."""
        raise ValueError('native atom context has no neural/task admission')


def native_atom_context(source, *, ordered_atom_ids, source_payload_sha256):
    """Copy exact ordered identity and explicit unknowns without target features.

    Unknowns are not neutral charge, singlet spin, or absent biological effects.
    Source-native split and reaction grouping are retained without reassignments.
    """
    if not isinstance(source, Mapping):
        raise ValueError('source metadata must be a mapping')
    for field in ('source_record_id', 'source_reaction_group', 'source_native_split'):
        value = source.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{field} must be explicit')
    digest = source_payload_sha256
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('source payload SHA256 must be lowercase hex')
    if not isinstance(ordered_atom_ids, (list, tuple)):
        raise ValueError('ordered atom IDs must be a complete sequence')
    ids = tuple(ordered_atom_ids)
    if not ids or any(not isinstance(x, str) or not x.strip() for x in ids) or len(set(ids)) != len(ids):
        raise ValueError('ordered atom IDs must be nonempty unique strings')
    numbers = source.get('atomic_numbers')
    if not isinstance(numbers, (list, tuple)) or len(numbers) != len(ids):
        raise ValueError('complete native atomic number axis required')
    if any(type(z) is not int or not 1 <= z <= 118 for z in numbers):
        raise ValueError('native atomic numbers must be integers in [1,118]')
    for field in ('electronic_charge', 'spin_multiplicity'):
        if field not in source or source[field] is not None or source.get(field + '_known') is not False:
            raise ValueError(f'{field} must be explicitly unknown in version 1')
    if source.get('task_admitted') is not False:
        raise ValueError('version 1 requires explicit closed task admission')
    return NativeAtomContext('native-atom-context-v1', source['source_record_id'],
        source['source_reaction_group'], source['source_native_split'], digest, ids, tuple(numbers))


@dataclass(frozen=True)
class NativeAtomIdentityEncoding:
    """Diagnostic categorical features with separate explicit missingness.

    Tensors are caller-owned fresh storage; metadata remains immutable. This is
    not an authenticated chemistry context or a qualified neural input.
    """
    context: NativeAtomContext
    identity: object
    context_missing: object
    context_fields: tuple[str, ...]

    def require_model_use(self):
        raise ValueError('native atom identity encoding has no neural/task admission')


def encode_native_atom_identity(context, *, device='cpu', dtype=None):
    """Encode Z as e_(Z-1) on the complete 118-element axis, without targets.

    Missing electronic/environment values are boolean masks, never inferred
    neutral/singlet values. Only diagnostic FP32/64 encodings are qualified.
    No model defaults, embeddings, parameters or admission policies are changed.
    """
    import torch
    if type(context) is not NativeAtomContext:
        raise ValueError('exact native atom context contract required')
    # Revalidate public frozen values too: direct construction/replace is possible.
    replay = native_atom_context(dict(
        source_record_id=context.source_record_id,
        source_reaction_group=context.source_reaction_group,
        source_native_split=context.source_native_split,
        atomic_numbers=context.atomic_numbers,
        electronic_charge=context.electronic_charge,
        spin_multiplicity=context.spin_multiplicity,
        electronic_charge_known=False, spin_multiplicity_known=False,
        task_admitted=False), ordered_atom_ids=context.ordered_atom_ids,
        source_payload_sha256=context.source_payload_sha256)
    if replay != context:
        raise ValueError('unsupported native atom context version or missingness')
    if dtype is None:
        dtype = torch.float32
    if dtype not in (torch.float32, torch.float64):
        raise ValueError('diagnostic identity encoding requires FP32 or FP64')
    indices = torch.tensor(context.atomic_numbers, device=device, dtype=torch.long) - 1
    identity = torch.nn.functional.one_hot(indices, num_classes=118).to(dtype=dtype)
    missing = torch.ones((len(context.atomic_numbers), len(context.missing_context)),
                         device=device, dtype=torch.bool)
    return NativeAtomIdentityEncoding(context, identity, missing, context.missing_context)
