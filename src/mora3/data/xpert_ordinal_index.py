"""Finalized Norman original-offset binding and bounded stored-row gathering.

Observation/context and full sparse-profile composition remain separate gates.
Historical fixture/full-store APIs do not acquire row access through this module.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .xpert_mmap import NativeMmapArray, transport_descriptor, _canonical, _json
from .xpert_full_store import XPertFullNumericStore


def original_gene_coordinates(indptr, offsets, width):
    """Locate original storage ordinals in CSC intervals, retaining empty columns."""
    if (type(width) is not int or width < 1
            or type(indptr) is not np.ndarray or indptr.dtype.str != '<i4'
            or indptr.shape != (width + 1,)
            or type(offsets) is not np.ndarray or offsets.dtype.str != '<u4'
            or offsets.ndim != 1 or len(offsets) > width
            or indptr[0] != 0 or np.any(indptr[1:] < indptr[:-1])
            or (len(offsets) and (offsets[-1] >= indptr[-1]
                or np.any(offsets[1:] <= offsets[:-1])))):
        raise ValueError('exact original CSC pointers/ordered bounded ordinals required')
    genes = np.searchsorted(indptr, offsets, side='right') - 1
    if np.any(genes[1:] <= genes[:-1]):
        raise ValueError('pinned unique original row/gene coordinates required; no coalescing')
    return genes.astype('<i4')


@dataclass(frozen=True)
class NormanRowOrdinals:
    native_row_index: int
    original_offsets_bytes: bytes
    gene_coordinates_bytes: bytes
    provenance_bytes: bytes

    def offsets(self):
        return np.frombuffer(self.original_offsets_bytes, dtype='<u4')

    def gene_coordinates(self):
        return np.frombuffer(self.gene_coordinates_bytes, dtype='<i4')

    def audit(self):
        return _json(self.provenance_bytes)

    def require_model_use(self, purpose):
        raise ValueError('ordinal metadata is not source-context/task admission')


class XPertNormanOrdinalIndex:
    """Fresh pinned proof/header/stat/chunk replay for up to three source rows."""
    HANDOFF = dict(
        path='outputs/data-sources/20261101-XPert-source-v1/handoff-full-Norman-original-ordinal-row-access-index-v2.json',
        sha256='f274acd5c95ac2de380473e4307437147e21151d2f7308911df26774be27b7f5')
    MANIFEST = dict(
        path='outputs/data-sources/20261101-XPert-source-v1/curated/Norman-original-ordinal-row-access-index-v2/numeric-store-manifest.json',
        bytes=6594345, sha256='4315b32d793d62a3f8003f4fddcde86552963bba5e2f130e1a233d348fd2b55e')

    def __init__(self, *, root):
        self.store = XPertFullNumericStore(root=root)
        self.root = self.store.root
        self._snapshot()

    def _read(self, ref):
        ref = dict(ref)
        path = Path(ref['path'])
        if path.is_absolute():
            ref['path'] = str(path.relative_to(Path(__file__).resolve().parents[3]))
        return self.store._read(ref)

    def _snapshot(self):
        handoff = self._read(self.HANDOFF)
        manifest = self._read(self.MANIFEST)
        original = self.store._snapshot()
        proof = self._read(handoff['independent_completion_proof'])
        if (handoff['index_manifest']['sha256'] != self.MANIFEST['sha256']
                or manifest['source_store_ref']['sha256'] != self.store.MANIFEST['sha256']
                or proof['index_manifest']['sha256'] != self.MANIFEST['sha256']
                or handoff['all_original_offsets_bijective_and_row_gene_mapping_checked'] is not True
                or manifest['complete_original_ordinal_occurrence_bijection'] is not True
                or manifest['source_values_copied_or_rewritten'] is not False
                or manifest['source_task_admitted'] != 0 or manifest['model_use'] is not False
                or proof['whole_semantic_audit'] != manifest['semantic_audit']):
            raise ValueError('exact finalized source/index/independent semantic proof required')
        dataset = next(d for d in original['datasets'] if d['source_article_ID'] == 32221275)
        if manifest['original_source_dataset'] != dataset:
            raise ValueError('index must retain exact original source dataset')
        for ref in handoff['source_rights_and_scientific_preparation_gates'].values():
            self._read(ref)
        if set(manifest['array_transports']) != {'row_indptr', 'row_source_element_ordinal'}:
            raise ValueError('only finalized original-ordinal access metadata allowed')
        return manifest, original, dataset

    def read_row_offsets(self, rows):
        from .xpert_mmap import _selectors
        manifest, original, dataset = self._snapshot()
        _selectors(rows, dataset['logical_shape'][0])
        if len(rows) > 3:
            raise ValueError('at most three bounded original rows per selection')
        arrays = {key: NativeMmapArray(root=self.root,
            descriptor_bytes=transport_descriptor(record))
            for key, record in manifest['array_transports'].items()}
        rowptr = arrays['row_indptr']
        offsets = arrays['row_source_element_ordinal']
        total = dataset['native_stored_matrix_values']
        width = dataset['logical_shape'][1]
        if (rowptr.descriptor()['native_dtype'] != '<u8'
                or rowptr.descriptor()['native_shape'] != [dataset['logical_shape'][0]+1]
                or offsets.descriptor()['native_dtype'] != '<u4'
                or offsets.descriptor()['native_shape'] != [total]
                or int(rowptr.read_elements(0, 1).native_values()[0]) != 0
                or int(rowptr.read_elements(dataset['logical_shape'][0],
                    dataset['logical_shape'][0]+1).native_values()[0]) != total):
            raise ValueError('complete native index extent/terminal pointers required')
        ptr = self.store.read_storage(dataset['arrays']['indptr'], 0, width+1)
        if int(ptr.native_values()[-1]) != total:
            raise ValueError('original CSC terminal pointer differs from index')
        out = []
        for row in rows:
            pair = rowptr.read_elements(row, row+2)
            start, stop = map(int, pair.native_values())
            if not 0 <= start <= stop <= total or stop-start > width:
                raise ValueError('bounded exact original row index interval required')
            span = offsets.read_elements(start, stop)
            genes = original_gene_coordinates(ptr.native_values(), span.native_values(), width)
            provenance = dict(
                binding_version='mora3.xpert.Norman-original-ordinal-index.v2.reader.v1',
                index_handoff_sha256=self.HANDOFF['sha256'],
                index_manifest_sha256=self.MANIFEST['sha256'],
                full_store_manifest_sha256=self.store.MANIFEST['sha256'],
                native_row_index=row, complete_gene_axis_length=width,
                original_stored_occurrences=stop-start,
                row_pointer_proof=pair.audit(), ordinal_proof=span.audit(),
                original_column_pointer_proof=ptr.audit(),
                whole_occurrence_bijection_inherited=True,
                selected_original_row_indices_rechecked=False,
                expression_values_read=False, context_bound=False,
                implicit_zero_is_missing=False, model_use_allowed=False,
                task_admitted=False, official_split=None, full_dataset_ready=False)
            out.append(NormanRowOrdinals(row, span.values_bytes, genes.tobytes(), _canonical(provenance)))
        return tuple(out)

    def require_model_use(self, purpose):
        raise ValueError('original index access is not source-context/task admission')

    def read_stored_rows(self, rows, *, max_payload_bytes):
        """Gather <=3 original sparse rows; full profiles/context remain separate.

        Caller explicitly budgets union numeric/validity I/O across BOTH source
        arrays. No full matrix or dense row is assembled, and no context inferred.
        """
        from .xpert_native_gather import (plan_original_ordinals,
            gather_original_ordinals, MAX_GATHER_PAYLOAD_BYTES)
        if (type(max_payload_bytes) is not int
                or not 1 <= max_payload_bytes <= MAX_GATHER_PAYLOAD_BYTES):
            raise ValueError('explicit bounded original gather IO budget required')
        ordinals = self.read_row_offsets(rows)
        # Fresh complete source/index proof immediately precedes source selection.
        manifest, original, dataset = self._snapshot()
        arrays = {key: self.store._array(original, dataset['arrays'][key])
            for key in ('data','indices')}
        if (arrays['data'].descriptor()['native_dtype'] != '<f4'
                or arrays['indices'].descriptor()['native_dtype'] != '<i4'
                or any(a.descriptor()['native_shape'] != [dataset['native_stored_matrix_values']]
                    for a in arrays.values())):
            raise ValueError('exact native float32/int32 source storage required')
        offsets = tuple(row.offsets() for row in ordinals)
        planned = sum(plan_original_ordinals(a,offsets)[1] for a in arrays.values())
        if planned > max_payload_bytes:
            raise ValueError('combined original value/index chunks exceed gather IO budget')
        gathered = {}; audits = {}
        for key,array in arrays.items():
            gathered[key],audits[key] = gather_original_ordinals(
                array,offsets,max_payload_bytes=max_payload_bytes)
        result = []
        for i,row in enumerate(ordinals):
            raw,finite = gathered['data'][i]
            indices,index_finite = gathered['indices'][i]
            native_indices = np.frombuffer(indices,dtype='<i4')
            if (np.any(native_indices != row.native_row_index)
                    or index_finite != b'\1'*len(native_indices)):
                raise ValueError('selected original CSC row indices differ from requested row')
            provenance = dict(row.audit(),
                binding_version='mora3.xpert.Norman-original-stored-row.v2.reader.v1')
            provenance.update(selected_original_row_indices_rechecked=True,
                expression_values_read=True,original_native_row_indices_bytes=indices.hex(),
                numeric_array_proofs=audits,combined_payload_bytes=planned,
                max_payload_bytes=max_payload_bytes,context_bound=False,
                complete_sparse_profile_composed=False)
            result.append(NormanStoredRow(row,raw,finite,_canonical(provenance)))
        return tuple(result)


    def read_candidate_profiles(self, rows, *, max_payload_bytes):
        """Bind <=3 native sparse targets, complete gene axis and exact context.

        No matrix densification, normalization, task admission or split assignment.
        Original numeric union I/O remains explicitly budgeted by the caller.
        """
        from .xpert_full_store import XPertFullCandidateProfiles
        stored = self.read_stored_rows(rows, max_payload_bytes=max_payload_bytes)
        contexts = self.store.read_contexts(32221275, rows)
        observations = self.store.read_observations(32221275, rows)
        axis = self.store.read_gene_axis(32221275)
        if (len(axis.native_records) != 33694
                or tuple(r.native_row_index for r in stored) != rows
                or len(contexts.contexts) != len(rows)
                or len(observations.native_records) != len(rows)
                or any(c.native_fields() != observations.record(i)['native_fields']
                       for i,c in enumerate(contexts.contexts))):
            raise ValueError('exact native sparse profile/observation/context/axis binding required')
        profiles = tuple(NormanSparseProfile(r, len(axis.native_records)) for r in stored)
        return XPertFullCandidateProfiles(profiles, observations, contexts, axis,
            _canonical(dict(contexts.audit(),
                binding_version='mora3.xpert.Norman-full-sparse-candidate.v1',
                native_row_indices=list(rows), numeric_bound=True,
                gene_axis_sha256=axis.audit()['gene_axis_sha256'],
                gene_axis_length=len(axis.native_records),
                index_manifest_sha256=self.MANIFEST['sha256'],
                numeric_proofs=[r.audit() for r in stored],
                complete_sparse_profile_composed=True, implicit_zero_is_missing=False,
                source_identity_verified=True, task_admitted=False,
                model_use_allowed=False, official_split=None)))


@dataclass(frozen=True)
class NormanStoredRow:
    ordinals: NormanRowOrdinals
    values_bytes: bytes
    finite_mask: bytes
    provenance_bytes: bytes

    def __post_init__(self):
        if (type(self.values_bytes) is not bytes or type(self.finite_mask) is not bytes
                or len(self.values_bytes) != 4*len(self.ordinals.offsets())
                or self.finite_mask != np.isfinite(self.native_values()).tobytes()):
            raise ValueError('exact native float32 stored occurrences/validity required')
        _json(self.provenance_bytes)

    @property
    def native_row_index(self):
        return self.ordinals.native_row_index

    def native_values(self):
        return np.frombuffer(self.values_bytes,dtype='<f4')

    def audit(self):
        return _json(self.provenance_bytes)

    def require_model_use(self, purpose):
        raise ValueError('stored row gathering is not source-context/task admission')


@dataclass(frozen=True)
class NormanSparseProfile:
    """Native sparse occurrences plus logical axis; never a dense value vector.

    An absent CSC occurrence is an observed finite zero, whereas a stored
    nonfinite occurrence remains nonfinite. Explicit stored signed zeros retain
    their original bits in ``stored``.
    """
    stored: NormanStoredRow
    axis_length: int

    def __post_init__(self):
        genes = self.stored.ordinals.gene_coordinates()
        if (type(self.axis_length) is not int or not 1 <= self.axis_length <= 33694
                or len(genes) != len(self.stored.native_values())
                or np.any(genes < 0) or np.any(genes >= self.axis_length)
                or np.any(genes[1:] <= genes[:-1])):
            raise ValueError('bounded complete sparse gene axis with unique ordered coordinates required')

    @property
    def native_row_index(self):
        return self.stored.native_row_index

    def axis_masks(self):
        genes = self.stored.ordinals.gene_coordinates()
        present = np.zeros(self.axis_length, dtype=bool)
        present[genes] = True
        finite = np.ones(self.axis_length, dtype=bool)
        finite[genes] = np.frombuffer(self.stored.finite_mask, dtype=bool)
        return dict(stored=present, implicit_zero=~present, finite=finite)

    def require_model_use(self, purpose):
        raise ValueError('sparse profile composition is not task/model admission')
