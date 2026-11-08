"""Bounded identity-only reader for one pinned retained Transition1x export.

Authentication stops at the retained Parquet export: the original HDF5 is not
opened or independently authenticated. Export hashes, splits and reaction groups
are not corpus admission, canonical chemical identity or biological context.
No coordinates, forces, energies or target-derived features are decoded.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .native_atom_context import NativeAtomContext, native_atom_context

MANIFEST_SHA256 = '2722e93efd62a04e829bdbff9f08240bcffe0d0bf960e2671a12251b598a914c'
_COLUMNS = ('source_record_id', 'source_reaction_group', 'source_native_split',
            'atomic_numbers', 'electronic_charge', 'spin_multiplicity',
            'electronic_charge_known', 'spin_multiplicity_known', 'task_admitted',
            'conditional_TS_inference_admitted', 'phaseII_dynamics_admitted')


@dataclass(frozen=True)
class RetainedAtomIdentity:
    """Source-local ordinal identity, with no transferable authority token."""
    context: NativeAtomContext
    payload_index: int
    ordinal: int
    manifest_sha256: str
    version: str = 'transition1x-retained-atom-identity-v1'

    def require_model_use(self):
        raise ValueError('retained export identity grants no model admission')

    def require_training_use(self):
        raise ValueError('retained export identity grants no training admission')

    def require_task_use(self):
        raise ValueError('retained export identity grants no task admission')


def read_retained_atom_identity(*, manifest_path, payload_index, ordinal,
        expected_record_id, expected_reaction_group, expected_native_split,
        expected_atomic_numbers, payload_path=None):
    """Freshly hash a pinned manifest and selected payload, then project identity.

    Ordinal is zero-based within the explicitly selected payload, never a filtered
    table. The caller must bind record/group/split and the complete ordered Z axis.
    Atom IDs are source-local ordinal names, not chemistry mapping across records.
    Optional payload relocation must retain every original byte. No digest cache,
    external source access, HDF5 runtime, source writes or neural work occurs.
    """
    return _read_identity(manifest_path=manifest_path, payload_index=payload_index,
        ordinal=ordinal, expected_record_id=expected_record_id,
        expected_reaction_group=expected_reaction_group,
        expected_native_split=expected_native_split,
        expected_atomic_numbers=expected_atomic_numbers, payload_path=payload_path,
        cache={})


def read_retained_atom_identities(selections):
    """Authenticate a bounded ordered batch with reuse confined to this call.

    Each unique manifest/payload is read and hashed afresh on the next call.
    Only identity columns are decoded, once per selected payload row group.
    This grants no additional source, task, model or training authority.
    """
    if type(selections) not in (list, tuple) or not 1 <= len(selections) <= 64:
        raise ValueError('bounded nonempty batch of at most 64 selections required')
    required = {'manifest_path', 'payload_index', 'ordinal', 'expected_record_id',
                'expected_reaction_group', 'expected_native_split', 'expected_atomic_numbers'}
    if any(type(row) is not dict or not required <= row.keys()
           or row.keys() - required - {'payload_path'} for row in selections):
        raise ValueError('exact identity selection fields required')
    cache = {}
    return tuple(_read_identity(**row, cache=cache) for row in selections)


def _read_identity(*, manifest_path, payload_index, ordinal, expected_record_id,
                   expected_reaction_group, expected_native_split,
                   expected_atomic_numbers, payload_path=None, cache):
    import pyarrow as pa
    import pyarrow.parquet as pq

    if type(payload_index) is not int or not 0 <= payload_index < 11:
        raise ValueError('bounded exact payload index required')
    if type(ordinal) is not int or ordinal < 0:
        raise ValueError('exact nonnegative source ordinal required')
    if (not isinstance(expected_atomic_numbers, (tuple, list))
            or not expected_atomic_numbers
            or any(type(z) is not int or not 1 <= z <= 118 for z in expected_atomic_numbers)):
        raise ValueError('complete exact atomic number axis required')
    for value in (expected_record_id, expected_reaction_group):
        if type(value) is not str or not value.strip():
            raise ValueError('explicit source identity binding required')
    if type(expected_native_split) is not str or expected_native_split not in ('train', 'val', 'test'):
        raise ValueError('explicit source split required')
    manifest_key = ('manifest', Path(manifest_path).resolve())
    if manifest_key not in cache:
        cache[manifest_key] = Path(manifest_path).read_bytes()
    manifest_bytes = cache[manifest_key]
    parsed_key = ('parsed', manifest_key)
    if parsed_key not in cache:
        if hashlib.sha256(manifest_bytes).hexdigest() != MANIFEST_SHA256:
            raise ValueError('pinned retained manifest hash mismatch')
        cache[parsed_key] = json.loads(manifest_bytes)
    manifest = cache[parsed_key]
    item = manifest['payloads'][payload_index]
    if ordinal >= item['rows']:
        raise ValueError('ordinal outside native payload')
    # Parse the same authenticated bytes, avoiding a hash-then-reopen window.
    path = Path(item['path'] if payload_path is None else payload_path)
    if path.stat().st_size != item['bytes']:
        raise ValueError('native payload byte count mismatch')
    payload_key = ('payload', path.resolve())
    if payload_key not in cache:
        cache[payload_key] = path.read_bytes()
    data = cache[payload_key]
    parquet_key = ('parquet', path.resolve(), item['sha256'], item['rows'], item['bytes'])
    if parquet_key not in cache:
        if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError('native payload hash mismatch')
        parquet = pq.ParquetFile(pa.BufferReader(data))
        if parquet.metadata.num_rows != item['rows']:
            raise ValueError('native payload row count mismatch')
        cache[parquet_key] = parquet
    parquet = cache[parquet_key]
    offset = ordinal
    for group in range(parquet.metadata.num_row_groups):
        size = parquet.metadata.row_group(group).num_rows
        if offset < size:
            group_key = ('group', parquet_key, group)
            if group_key not in cache:
                cache[group_key] = parquet.read_row_group(group, columns=list(_COLUMNS))
            row = cache[group_key].slice(offset, 1).to_pylist()[0]
            break
        offset -= size
    else:
        raise ValueError('source ordinal not found')
    if (row['source_record_id'] != expected_record_id
            or row['source_reaction_group'] != expected_reaction_group
            or row['source_native_split'] != expected_native_split
            or tuple(row['atomic_numbers']) != tuple(expected_atomic_numbers)):
        raise ValueError('source ordinal/record/group/split/ordered axis binding mismatch')
    if any(row[field] is not False for field in
           ('task_admitted', 'conditional_TS_inference_admitted', 'phaseII_dynamics_admitted')):
        raise ValueError('retained source requires closed admission')
    ids = tuple(f'{row["source_record_id"]}:atom:{i}' for i in range(len(row['atomic_numbers'])))
    context = native_atom_context(row, ordered_atom_ids=ids, source_payload_sha256=item['sha256'])
    return RetainedAtomIdentity(context, payload_index, ordinal, MANIFEST_SHA256)
