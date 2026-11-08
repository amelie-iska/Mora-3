"""Pinned full X-Pert numeric transport v1, without semantic task admission.

This deliberately exposes storage spans only. Observation/context/axis replay and
source-qualified dense/CSC interfaces are separate acceptance gates. It does not
extend the historical fixture reader or supply neural inputs.
"""
from pathlib import Path
import json
import copy
from .xpert_mmap import NativeMmapArray, NativeNumericSlice, transport_descriptor, _canonical, _sha


class XPertFullNumericStore:
    MANIFEST = dict(
        path='outputs/data-sources/20261101-XPert-source-v1/curated/full-native-numeric-store-v1/numeric-store-manifest.json',
        bytes=792391, sha256='917906149da22c2722e93d01b5f6cc3e5399cce91dba5776fac3d4dcd03dbb84')
    HANDOFF = dict(
        path='outputs/data-sources/20261101-XPert-source-v1/handoff-full-XPert-native-numeric-store-v1.json',
        sha256='78b7b22282c4656ca9da3eab3be01b35560f20b46c3e9ce809f05f160f9bc0df')

    def __init__(self, *, root):
        self.root = Path(root).resolve()
        self._snapshot()

    def _read(self, ref):
        path = (self.root / ref['path']).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('full-store proof must remain within project')
        raw = path.read_bytes()
        if _sha(raw) != ref['sha256'] or ('bytes' in ref and len(raw) != ref['bytes']):
            raise ValueError('exact full-store immutable proof mismatch')
        return json.loads(raw)

    def _snapshot(self):
        # Revalidate immutable bindings each selection; no future-store fallback.
        handoff = self._read(self.HANDOFF)
        manifest = self._read(self.MANIFEST)
        if (handoff['numeric_store_manifest']['sha256'] != self.MANIFEST['sha256']
                or manifest['fixture_only'] is not False
                or manifest['full_source_export'] is not True
                or manifest['model_use_allowed'] is not False
                or type(manifest['task_admitted_records']) is not int
                or manifest['task_admitted_records'] != 0):
            raise ValueError('candidate-only complete transport required')
        return manifest

    def read_storage(self, array_id, start, stop):
        manifest = self._snapshot()
        if type(array_id) is not str or array_id not in manifest['array_transports']:
            raise ValueError('array absent from pinned full store')
        record = copy.deepcopy(manifest['array_transports'][array_id])
        # The full export adds source finite counts to expected chunks. Preserve
        # the pinned receipt; project only partition coordinates for the v1 core.
        for chunk in record['expected_native']['chunks']:
            count = chunk.pop('finite_values')
            if type(count) is not int or not 0 <= count <= chunk['native_elements']:
                raise ValueError('exact full-export finite-count proof required')
        array = NativeMmapArray(root=self.root,
            descriptor_bytes=transport_descriptor(record))
        value = array.read_elements(start, stop)
        provenance = dict(value.audit(), full_store_manifest_sha256=self.MANIFEST['sha256'],
            full_store_handoff_sha256=self.HANDOFF['sha256'], array_id=array_id,
            binding_version='mora3.xpert.full-numeric-transport.v1', context_bound=False)
        return NativeNumericSlice(value.values_bytes, value.finite_mask, value.native_dtype,
            value.start, value.stop, _canonical(provenance))

    def _array(self, manifest, key):
        # Keep constructor identity aligned with the current strict CSC API.
        from .xpert_mmap import NativeMmapArray, transport_descriptor
        record = copy.deepcopy(manifest['array_transports'][key])
        for chunk in record['expected_native']['chunks']:
            count = chunk.pop('finite_values')
            if type(count) is not int or not 0 <= count <= chunk['native_elements']:
                raise ValueError('exact full-export finite-count proof required')
        return NativeMmapArray(root=self.root, descriptor_bytes=transport_descriptor(record))

    def read_dense_rows(self, rows):
        """Complete native numeric rows only; no observation/context qualification."""
        from .xpert_mmap import DenseNativeRow, _selectors
        manifest = self._snapshot()
        dataset = next(d for d in manifest['datasets'] if d['matrix_encoding'] == 'dense')
        _selectors(rows, dataset['logical_shape'][0])
        width = dataset['logical_shape'][1]
        return tuple(DenseNativeRow(i, self.read_storage(dataset['arrays']['X'],
            i*width, (i+1)*width), width) for i in rows)

    def read_csc_columns(self, columns):
        """Original full Norman columns; never densify, coalesce or sort."""
        from dataclasses import replace
        from .xpert_mmap import CSCMmapMatrix, _selectors
        manifest = self._snapshot()
        dataset = next(d for d in manifest['datasets'] if d['matrix_encoding'] == 'csc_matrix')
        _selectors(columns, dataset['logical_shape'][1])
        # Absolute source-owner receipts are projected relative to the original
        # project root. Read the proof under this instance's root, never outside.
        ref = dict(dataset['csc_full_original_coordinate_proof'])
        original_root = Path(__file__).resolve().parents[3]
        ref['path'] = str(Path(ref['path']).relative_to(original_root))
        proof = self._read(ref)['Norman_CSC_coordinate_audit']
        indices_key = dataset['arrays']['indices']
        expected_indices = manifest['array_transports'][indices_key]['transport']['native_value_sha256']
        if (proof['native_shape'] != dataset['logical_shape']
                or proof['indices_SHA256'] != expected_indices
                or proof['native_stored_values'] != dataset['native_stored_matrix_values']
                or proof['all_native_indices_whole_SHA_match'] is not True
                or proof['native_CSC_coordinate_unique'] is not True
                or type(proof['columns_not_strict_ascending']) is not int
                or proof['columns_not_strict_ascending'] != 0
                or type(proof['duplicate_coordinate_occurrences']) is not int
                or proof['duplicate_coordinate_occurrences'] != 0
                or proof['no_coalesce_or_rewrite'] is not True):
            raise ValueError('exact full native CSC coordinate proof required')
        matrix = CSCMmapMatrix(*(self._array(manifest, dataset['arrays'][k])
            for k in ('data','indices','indptr')), logical_shape=tuple(dataset['logical_shape']),
            coordinate_unique=True, strictly_ascending=True)
        provenance = dict(full_store_manifest_sha256=self.MANIFEST['sha256'],
            full_store_handoff_sha256=self.HANDOFF['sha256'],
            coordinate_proof_sha256=ref['sha256'],
            binding_version='mora3.xpert.full-native-coordinate-access.v1',
            context_bound=False, source_identity_verified=False,
            task_admitted=False, model_use_allowed=False, official_split=None,
            source_evidence_role='source-native numeric column only; axes/context unqualified')
        return tuple(replace(c, provenance_bytes=_canonical(dict(provenance,
            native_column_index=c.native_column_index))) for c in matrix.read_columns(columns))

    def read_observations(self, source_article_id, rows):
        """Bounded native obs replay, without interpreting context as conditions.

        Verify the complete compressed asset freshly; decode only through the
        greatest requested ordinal. No derived index or matrix access is used.
        """
        import gzip
        from io import BytesIO
        from .xpert_mmap import _selectors
        manifest = self._snapshot()
        if type(source_article_id) is not int:
            raise ValueError('explicit native source article integer required')
        datasets = [d for d in manifest['datasets']
                    if d['source_article_ID'] == source_article_id]
        if len(datasets) != 1:
            raise ValueError('source absent from exact full store')
        dataset = datasets[0]
        _selectors(rows, dataset['logical_shape'][0])
        ref = dataset['obs_axis']
        if ref['native_rows'] != dataset['logical_shape'][0]:
            raise ValueError('native observation count mismatch')
        original_root = Path(__file__).resolve().parents[3]
        path = (self.root / Path(ref['path']).relative_to(original_root)).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('observation proof must remain within project')
        raw = path.read_bytes()
        if len(raw) != ref['bytes'] or _sha(raw) != ref['sha256']:
            raise ValueError('exact full-store immutable proof mismatch')
        selected = {}
        wanted = set(rows)
        fields = [ref['index_column'], *ref['native_column_order']]
        with gzip.GzipFile(fileobj=BytesIO(raw)) as stream:
            for ordinal in range(max(rows) + 1):
                line = stream.readline()
                if not line:
                    raise ValueError('native observation axis incomplete')
                if ordinal not in wanted:
                    continue
                record = json.loads(line)
                if (record['native_axis'] != 'obs'
                        or type(record['native_row_index']) is not int
                        or record['native_row_index'] != ordinal
                        or list(record['native_fields']) != fields
                        or record['task_admitted'] is not False
                        or record['adapter_support'] is not False):
                    raise ValueError('exact native observation order/fields/roles required')
                selected[ordinal] = line
        return XPertNativeObservations(tuple(selected[i] for i in rows), _canonical(dict(
            binding_version='mora3.xpert.full-observation-replay.v1',
            source_article_id=source_article_id, native_row_indices=list(rows),
            observation_axis_sha256=ref['sha256'],
            source_native_observation_unit=dataset['source_native_observation_unit'],
            full_store_manifest_sha256=self.MANIFEST['sha256'],
            full_store_handoff_sha256=self.HANDOFF['sha256'],
            context_bound=False, task_admitted=False, model_use_allowed=False,
            official_split=None, source_evidence_role='source-native obs metadata only')))

    def read_contexts(self, source_article_id, rows):
        """Fresh source-proof-bound target-free context; no numeric/task admission."""
        from .xpert_native import FULL_REF, SOURCES, XPertContext, _canonical as native_canonical
        manifest = self._snapshot()
        # Observation selection performs strict article/ordinal validation first.
        observations = self.read_observations(source_article_id, rows)
        dataset = next(d for d in manifest['datasets']
                       if d['source_article_ID'] == source_article_id)
        full = self._read(FULL_REF)
        refs = dict(rights=full['source_release_rights'],
                    QC=dataset['source_QC'],
                    preparation=dataset['source_preparation_gates'],
                    roles=dataset['source_role_masks'])
        original_root = Path(__file__).resolve().parents[3]
        for ref in refs.values():
            relative = dict(ref)
            path = Path(relative['path'])
            if path.is_absolute():
                relative['path'] = str(path.relative_to(original_root))
            self._read(relative)
        if (dataset['source_original_H5AD']['sha256'] != SOURCES[source_article_id]['source_sha256']
                or dataset['source_task_admitted'] != 0
                or dataset['model_use_allowed'] is not False):
            raise ValueError('exact source identity and closed task gates required')
        contexts = tuple(XPertContext(source_article_id,
            native_canonical(
                observations.record(i)['native_fields'])) for i in range(len(rows)))
        return XPertFullContexts(contexts, _canonical(dict(
            observations.audit(), binding_version='mora3.xpert.full-context-proof.v1',
            proof_refs=refs, rights_verified_as_evidence_only=True,
            numeric_bound=False, context_bound=True,
            source_value_role='native_expression_target_only',
            biological_samecell_link=None, guide_efficacy=None)))

    def read_candidate_profiles(self, source_article_id, rows):
        """Compose complete dense targets with source context, without admission.

        Norman CSC row assembly requires a separately reviewed ordinal index.
        This version only binds the pinned L1000 dense source, never sparse rows.
        """
        manifest = self._snapshot()
        if type(source_article_id) is not int or source_article_id != 32221302:
            raise ValueError('only explicit pinned L1000 dense profile composition supported')
        dataset = next(d for d in manifest['datasets']
                       if d['source_article_ID'] == source_article_id)
        if dataset['matrix_encoding'] != 'dense':
            raise ValueError('dense native profile source required')
        from .xpert_mmap import _selectors
        _selectors(rows, dataset['logical_shape'][0])
        contexts = self.read_contexts(source_article_id, rows)
        observations = self.read_observations(source_article_id, rows)
        axis = self.read_gene_axis(source_article_id)
        profiles = self.read_dense_rows(rows)
        if (tuple(p.native_row_index for p in profiles) != rows
                or any(p.axis_length != len(axis.native_records) for p in profiles)
                or any(context.native_fields() != observations.record(i)['native_fields']
                       for i, context in enumerate(contexts.contexts))):
            raise ValueError('exact native profile/observation/axis composition required')
        return XPertFullCandidateProfiles(profiles, observations, contexts, axis,
            _canonical(dict(contexts.audit(),
                binding_version='mora3.xpert.full-dense-candidate-composition.v1',
                numeric_bound=True, gene_axis_sha256=axis.audit()['gene_axis_sha256'],
                gene_axis_length=len(axis.native_records),
                numeric_proofs=[p.values.audit() for p in profiles],
                source_identity_verified=True, task_admitted=False,
                model_use_allowed=False, official_split=None)))

    def read_candidate_columns(self, source_article_id, columns, context_rows):
        """Bind Norman columns and selected contexts without assembling sparse rows.

        Complete numeric columns retain original coordinates. Contexts cover only
        explicitly requested observation ordinals, never every logical row.
        """
        from .xpert_mmap import _selectors
        manifest = self._snapshot()
        if type(source_article_id) is not int or source_article_id != 32221275:
            raise ValueError('only explicit pinned Norman CSC source supported')
        dataset = next(d for d in manifest['datasets']
                       if d['source_article_ID'] == source_article_id)
        if dataset['matrix_encoding'] != 'csc_matrix':
            raise ValueError('native CSC source required')
        _selectors(columns, dataset['logical_shape'][1])
        _selectors(context_rows, dataset['logical_shape'][0])
        contexts = self.read_contexts(source_article_id, context_rows)
        observations = self.read_observations(source_article_id, context_rows)
        axis = self.read_gene_axis(source_article_id)
        numeric = self.read_csc_columns(columns)
        if (len(axis.native_records) != dataset['logical_shape'][1]
                or tuple(c.native_column_index for c in numeric) != columns
                or any(c.logical_rows != dataset['logical_shape'][0] for c in numeric)
                or any(c.native_fields() != observations.record(i)['native_fields']
                       for i, c in enumerate(contexts.contexts))):
            raise ValueError('exact native column/axis/selected-context binding required')
        return XPertFullCandidateColumns(numeric, observations, contexts, axis,
            _canonical(dict(contexts.audit(),
                binding_version='mora3.xpert.full-csc-selected-context-candidate.v1',
                native_column_indices=list(columns),
                context_native_row_indices=list(context_rows),
                all_observations_context_bound=False,
                numeric_bound=True, source_identity_verified=True,
                gene_axis_sha256=axis.audit()['gene_axis_sha256'],
                gene_axis_length=len(axis.native_records),
                coordinate_proofs=[c.audit() for c in numeric],
                task_admitted=False, model_use_allowed=False, official_split=None)))

    def read_gene_axis(self, source_article_id):
        """Replay the complete source var axis; no observation/task qualification.

        Compressed source bytes are freshly verified on every call. Retain exact
        JSONL records (including native numeric encodings), order and missingness.
        This reads metadata only and never changes or densifies the matrix.
        """
        import gzip
        from io import BytesIO
        manifest = self._snapshot()
        if type(source_article_id) is not int:
            raise ValueError('explicit native source article integer required')
        datasets = [d for d in manifest['datasets']
                    if d['source_article_ID'] == source_article_id]
        if len(datasets) != 1:
            raise ValueError('source absent from exact full store')
        dataset = datasets[0]
        ref = dataset['var_axis']
        original_root = Path(__file__).resolve().parents[3]
        relative = Path(ref['path']).relative_to(original_root)
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('gene-axis proof must remain within project')
        raw = path.read_bytes()
        if len(raw) != ref['bytes'] or _sha(raw) != ref['sha256']:
            raise ValueError('exact full-store immutable proof mismatch')
        with gzip.GzipFile(fileobj=BytesIO(raw)) as stream:
            records = tuple(stream)
        if len(records) != ref['native_rows'] or len(records) != dataset['logical_shape'][1]:
            raise ValueError('complete native gene axis required')
        expected_fields = [ref['index_column'], *ref['native_column_order']]
        for ordinal, raw_record in enumerate(records):
            record = json.loads(raw_record)
            if (record['native_axis'] != 'var'
                    or type(record['native_row_index']) is not int
                    or record['native_row_index'] != ordinal
                    or list(record['native_fields']) != expected_fields
                    or record['task_admitted'] is not False
                    or record['adapter_support'] is not False):
                raise ValueError('exact native gene order/fields/roles required')
        return XPertNativeGeneAxis(records, _canonical(dict(
            binding_version='mora3.xpert.full-gene-axis.v1',
            source_article_id=source_article_id,
            full_store_manifest_sha256=self.MANIFEST['sha256'],
            full_store_handoff_sha256=self.HANDOFF['sha256'],
            gene_axis_sha256=ref['sha256'], gene_axis_length=len(records),
            context_bound=False, task_admitted=False, model_use_allowed=False,
            official_split=None, source_evidence_role='source-native var metadata only')))


from dataclasses import dataclass


@dataclass(frozen=True)
class XPertNativeGeneAxis:
    """Immutable native JSONL replay; records decode into fresh caller copies."""
    native_records: tuple[bytes, ...]
    provenance_bytes: bytes

    def record(self, ordinal):
        if type(ordinal) is not int or not 0 <= ordinal < len(self.native_records):
            raise ValueError('native gene ordinal out of bounds')
        return json.loads(self.native_records[ordinal])

    def audit(self):
        return json.loads(self.provenance_bytes)

    def require_model_use(self, purpose):
        raise ValueError('gene-axis replay is not observation/context/task admission')


@dataclass(frozen=True)
class XPertNativeObservations(XPertNativeGeneAxis):
    """Selected original observation records; all task-use gates stay closed."""

    def require_model_use(self, purpose):
        raise ValueError('observation replay is not context/rights/task admission')


@dataclass(frozen=True)
class XPertFullContexts:
    """Immutable context/proof selection; numeric composition remains separate."""
    contexts: tuple
    provenance_bytes: bytes

    def audit(self):
        return json.loads(self.provenance_bytes)

    def require_model_use(self, purpose):
        from .xpert_native import require_XPert_model_use
        return require_XPert_model_use(purpose)


@dataclass(frozen=True)
class XPertFullCandidateProfiles:
    """Source-coordinate-bound targets; not qualified neural inputs or splits."""
    profiles: tuple
    observations: XPertNativeObservations
    contexts: XPertFullContexts
    gene_axis: XPertNativeGeneAxis
    provenance_bytes: bytes

    def audit(self):
        return json.loads(self.provenance_bytes)

    def require_model_use(self, purpose):
        from .xpert_native import require_XPert_model_use
        return require_XPert_model_use(purpose)


@dataclass(frozen=True)
class XPertFullCandidateColumns:
    """Complete native columns plus explicitly partial observation context."""
    columns: tuple
    observations: XPertNativeObservations
    contexts: XPertFullContexts
    gene_axis: XPertNativeGeneAxis
    provenance_bytes: bytes

    def audit(self):
        return json.loads(self.provenance_bytes)

    def require_model_use(self, purpose):
        from .xpert_native import require_XPert_model_use
        return require_XPert_model_use(purpose)
