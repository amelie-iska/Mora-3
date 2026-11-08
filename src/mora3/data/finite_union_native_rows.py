"""Read-only filtered view of original native source ordinals.

The view is a data selection, never a scientific/model/optimizer authorization.
Caller chooses explicit original columns. Gold/native payloads are not renamed
or implicitly converted into model conditions. Arrow is imported only by the
actual native iterator, keeping format validation independently testable.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


ROLES = {'train': 0, 'validation': 1, 'test': 2}
IDENTITY_FIELDS = ('st_dev', 'st_ino', 'st_mode', 'st_nlink', 'st_uid',
    'st_gid', 'st_size', 'st_mtime_ns', 'st_ctime_ns')


def full9(path):
    stat = Path(path).stat()
    return tuple(getattr(stat, field) for field in IDENTITY_FIELDS)


def read_exact(reference, *, root):
    path = Path(root) / reference['path']
    before = full9(path)
    raw = path.read_bytes()
    if (full9(path) != before or len(raw) != reference['bytes']
            or hashlib.sha256(raw).hexdigest() != reference['sha256']):
        raise ValueError('immutable artifact identity/bytes differ')
    return raw


def decode_disposition(descriptor, *, root):
    """Return exact original-row codes, rejecting implicit/truncated masks."""
    rows = descriptor['source_shard']['rows']
    value = descriptor['disposition']
    if type(rows) is not int or rows < 0 or type(value.get('rows')) is not int or value['rows'] != rows:
        raise ValueError('exact original ordinal count required')
    if value.get('kind') == 'homogeneous-original-row-disposition':
        code = value.get('code')
        if type(code) is not int or not 0 <= code <= 10:
            raise ValueError('unsupported disposition code')
        raw = bytes([code]) * rows
    elif value.get('kind') == 'original-row-ordinal-uint8':
        raw = read_exact(value['reference'], root=root)
        if len(raw) != rows or any(code > 10 for code in raw):
            raise ValueError('unsupported or incomplete ordinal disposition')
    else:
        raise ValueError('explicit original-row disposition representation required')
    if sum(code <= 2 for code in raw) != descriptor['primary_retained_rows']:
        raise ValueError('retained ordinal count differs')
    return raw


def selected_ordinals(codes, *, split, row_start, row_stop):
    if split not in ROLES:
        raise ValueError('literal original source role required')
    if (type(row_start) is not int or type(row_stop) is not int
            or not 0 <= row_start <= row_stop <= len(codes)):
        raise ValueError('exact absolute source-row interval required')
    return tuple(i for i in range(row_start, row_stop) if codes[i] == ROLES[split])


@dataclass(frozen=True)
class SelectedNativeSourceBatch:
    source_shard: dict
    source_row_ordinals: tuple
    original_split: str
    native_rows: object


class FiniteUnionNativeView:
    def __init__(self, manifest_path, expected_sha256, *, root):
        self.root = Path(root)
        path = Path(manifest_path)
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError('trusted finite-view manifest SHA differs')
        self.manifest = json.loads(raw)
        if (self.manifest.get('schema') != 'Mora3.ImmutableFiniteEligibleNativeUnion/v1'
                or self.manifest.get('full_project_training_ready') is not False
                or self.manifest.get('canonical_master_pointer_unchanged') is not True):
            raise ValueError('explicit immutable source-data view required')
        descriptor_raw = read_exact(self.manifest['original_immutable_source_views'], root=self.root)
        self.descriptors = tuple(json.loads(line) for line in descriptor_raw.splitlines())
        if (sum(row['source_shard']['rows'] for row in self.descriptors) != self.manifest['source_taskviews']
                or sum(row['primary_retained_rows'] for row in self.descriptors) != self.manifest['primary_retained_taskviews']):
            raise ValueError('view source and retained ordinal conservation differs')
        paths = [row['source_shard']['path'] for row in self.descriptors]
        if len(paths) != len(set(paths)):
            raise ValueError('duplicate original source shard')
        self._sealed_descriptors = tuple(json.dumps(row, sort_keys=True, separators=(',', ':'),
            allow_nan=False).encode() for row in self.descriptors)
        self._ordinal_exclusions = frozenset()
        self._sealed_by_path = {row['source_shard']['path']: sealed
            for row, sealed in zip(self.descriptors, self._sealed_descriptors)}


    @classmethod
    def from_effective_filter(cls, descriptor_path, expected_sha256, *, root):
        """Layer a reviewed exact ordinal exclusion over an immutable base view.

        Original uint8 codes, sources and master remain unchanged. This is a
        source-data selection and conveys no scientific/training permission.
        """
        import pyarrow.parquet as pq
        raw = Path(descriptor_path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError('trusted additive filter descriptor SHA differs')
        filt = json.loads(raw)
        if (filt.get('schema') != 'Mora3.AdditiveFiniteNativeOrdinalFilter/v1'
                or filt.get('reason') != 'heldout-MCSA-direct-source-annotation-evidence'
                or filt.get('source_role') != 'train' or filt.get('source_and_parent_masks_unchanged') is not True
                or filt.get('full_project_training_ready') is not False):
            raise ValueError('exact source-only additive ordinal filter required')
        parent = filt['parent_native_view']
        read_exact(parent, root=root)
        view = cls(Path(root)/parent['path'], parent['sha256'], root=root)
        evidence = json.loads(read_exact(filt['source_task_consumption_manifest'], root=root))
        check = json.loads(read_exact(filt['independent_source_ordinal_verification'], root=root))
        if (check.get('producer_manifest') != filt['source_task_consumption_manifest']
                or evidence.get('parent_finite_native_view') != parent
                or check.get('exact_source_task_coordinate_dispositions_both_directions') is not True
                or check.get('false_positives') != 0 or check.get('false_negatives') != 0):
            raise ValueError('same-snapshot exact independent task-ordinal proof required')
        exclusions = [json.loads(line) for line in read_exact(evidence['proposed_ordinal_overlay'], root=root).splitlines()]
        coordinates = set()
        by_source = {}
        for row in exclusions:
            shard = row['source_shard']; path = shard['path']; ordinal = row['absolute_original_row_ordinal']
            if (path not in view._sealed_by_path or type(ordinal) is not int
                    or row['original_split'] != 'train' or row['parent_view_disposition_code'] != 0
                    or type(row.get('direct_source_annotation_paper_assertions_consumed')) is not int
                    or row['direct_source_annotation_paper_assertions_consumed'] < 1
                    or row.get('same_experiment_or_physical_identity_proven') is not False):
                raise ValueError('only exact proven currently retained train task ordinals can be excluded')
            native = json.loads(view._sealed_by_path[path])
            if native['source_shard'] != shard or not 0 <= ordinal < shard['rows']:
                raise ValueError('source shard/ordinal differs from immutable base')
            coordinate = path, ordinal
            if coordinate in coordinates:
                raise ValueError('duplicate additive exclusion coordinate')
            coordinates.add(coordinate)
            by_source.setdefault(path, []).append(row)
        if (not coordinates or type(filt.get('excluded_taskviews')) is not int
                or filt['excluded_taskviews'] != len(coordinates)
                or check['counts']['exact_exclusion_taskviews'] != len(coordinates)):
            raise ValueError('exact positive additive ordinal count required')
        for path, rows in by_source.items():
            native = json.loads(view._sealed_by_path[path]); source = view.root/path
            identity = tuple(native['source_full9'])
            if full9(source) != identity:
                raise ValueError('bound original source identity differs')
            read_exact(native['original_group_sidecar'], root=root)
            codes = decode_disposition(native, root=root)
            table = pq.read_table(source, columns=['task_id','native_payload_sha256','task_name','source_family','split'])
            for row in rows:
                ordinal = row['absolute_original_row_ordinal']; actual = table.slice(ordinal,1).to_pylist()[0]
                if (codes[ordinal] != 0 or any(actual[key] != row[key] for key in
                        ('task_id','native_payload_sha256','task_name','source_family')) or actual['split'] != 'train'):
                    raise ValueError('actual original task/role no longer matches exact retained exclusion')
            if full9(source) != identity:
                raise ValueError('original source changed during filter binding')
        expected_counts = dict(view.manifest['primary_retained_split_counts'])
        expected_counts['train'] -= len(coordinates)
        if (expected_counts != filt['effective_retained_split_counts']
                or sum(expected_counts.values()) != filt['effective_retained_taskviews']):
            raise ValueError('base and effective role/count conservation differs')
        view._ordinal_exclusions = frozenset(coordinates)
        view.effective_filter = filt
        return view

    def require_native_rows(self, batch, *, split):
        """Re-read exact selected native records; this is source/data validation.

        No model objective, global project gate or optimizer is authorized by
        this method. Adapters derive their conditions/targets from these bound
        records and their separately reviewed native extractor semantics.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq
        if (type(batch) is not SelectedNativeSourceBatch or split not in ROLES
                or batch.original_split != split or type(batch.source_shard) is not dict
                or type(batch.source_row_ordinals) is not tuple or not batch.source_row_ordinals
                or any(type(index) is not int for index in batch.source_row_ordinals)
                or tuple(sorted(set(batch.source_row_ordinals))) != batch.source_row_ordinals
                or not isinstance(batch.native_rows, pa.RecordBatch)):
            raise ValueError('exact typed native batch/role and sorted unique absolute source ordinals required')
        sealed = self._sealed_by_path.get(batch.source_shard.get('path'))
        if sealed is None:
            raise ValueError('native source shard absent from this immutable view')
        descriptor = json.loads(sealed)
        if descriptor['source_shard'] != batch.source_shard:
            raise ValueError('original source shard reference differs')
        codes = decode_disposition(descriptor, root=self.root)
        if any((batch.source_shard['path'], index) in self._ordinal_exclusions for index in batch.source_row_ordinals):
            raise ValueError('native source row excluded by exact additive task filter')
        if any(not 0 <= index < len(codes) or codes[index] != ROLES[split] for index in batch.source_row_ordinals):
            raise ValueError('row is excluded/quarantined or belongs to another original role')
        columns = batch.native_rows.schema.names
        if (len(columns) != len(set(columns)) or not {'task_id', 'source_family', 'split'} <= set(columns)
                or batch.native_rows.num_rows != len(batch.source_row_ordinals)):
            raise ValueError('complete requested native columns and original row count required')
        source = self.root / descriptor['source_shard']['path']
        expected_identity = tuple(descriptor['source_full9'])
        if full9(source) != expected_identity:
            raise ValueError('bound original source identity changed')
        read_exact(descriptor['original_group_sidecar'], root=self.root)
        parquet = pq.ParquetFile(source)
        if set(columns) != set(parquet.schema_arrow.names):
            raise ValueError('model source binding requires every original native field; use iter_native_task_batches')
        source_start = 0
        native_parts = []
        for group in range(parquet.metadata.num_row_groups):
            stop = source_start + parquet.metadata.row_group(group).num_rows
            local = [index-source_start for index in batch.source_row_ordinals if source_start <= index < stop]
            if local:
                native = parquet.read_row_group(group, columns=columns, use_threads=False)
                native_parts.append(native.take(pa.array(local, type=pa.int64())))
            source_start = stop
        if source_start != len(codes) or full9(source) != expected_identity:
            raise ValueError('original native source rows/identity differ')
        expected = pa.concat_tables(native_parts)
        actual = pa.Table.from_batches([batch.native_rows])
        if (not expected.equals(actual, check_metadata=True)
                or any(value != split for value in expected['split'].to_pylist())):
            raise ValueError('native source field/value/type/order or original role was altered')
        return None

    def iter_native_task_batches(self, *, split, batch_size=1024, source_paths=None):
        """Explicitly request complete original task records, including audit.

        Conditions/targets remain inside their original native payload. This
        never designates every source field as a model predictor.
        """
        return self.iter_native_batches(split=split, columns=None, batch_size=batch_size, source_paths=source_paths)

    def iter_native_batches(self, *, split, columns, batch_size=1024, source_paths=None):
        """Read exact selected original rows for source-adapter consumption.

        This method does not activate a model objective or training run. Each
        selected row retains absolute source coordinates and original split.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq
        if (split not in ROLES or type(batch_size) is not int or batch_size <= 0
                or columns is not None and (type(columns) not in (list, tuple) or not columns
                    or any(type(column) is not str for column in columns)
                    or len(columns) != len(set(columns)))):
            raise ValueError('explicit role/columns and positive exact batch size required')
        required = ('task_id', 'source_family', 'split')
        projected = None if columns is None else list(dict.fromkeys((*required, *columns)))
        if source_paths is None:
            descriptors = self._sealed_descriptors
        else:
            if (type(source_paths) not in (tuple, list) or not source_paths
                    or any(type(path) is not str or not path for path in source_paths)
                    or len(set(source_paths)) != len(source_paths)
                    or any(path not in self._sealed_by_path for path in source_paths)):
                raise ValueError('exact nonempty unique bound source paths required')
            descriptors = tuple(self._sealed_by_path[path] for path in source_paths)
        for sealed in descriptors:
            descriptor = json.loads(sealed)
            codes = decode_disposition(descriptor, root=self.root)
            if ROLES[split] not in codes:
                continue
            path = self.root / descriptor['source_shard']['path']
            expected_identity = tuple(descriptor['source_full9'])
            if full9(path) != expected_identity:
                raise ValueError('bound native source identity changed')
            # Exact task IDs/ordinals were joined to the SHA-pinned authoritative
            # sidecar by the producer; pin that unchanged artifact at consumption.
            read_exact(descriptor['original_group_sidecar'], root=self.root)
            start = 0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=projected):
                stop = start + batch.num_rows
                ordinals = selected_ordinals(codes, split=split, row_start=start, row_stop=stop)
                ordinals = tuple(index for index in ordinals if (descriptor['source_shard']['path'], index) not in self._ordinal_exclusions)
                if full9(path) != expected_identity:
                    raise ValueError('native source changed during batch read')
                if ordinals:
                    selected = batch.take(pa.array([i - start for i in ordinals], type=pa.int64()))
                    if (any(value != split for value in selected.column(selected.schema.get_field_index('split')).to_pylist())
                            or selected.column(selected.schema.get_field_index('task_id')).null_count
                            or selected.column(selected.schema.get_field_index('source_family')).null_count):
                        raise ValueError('selected original task identity/role mismatch')
                    yield SelectedNativeSourceBatch(descriptor['source_shard'], ordinals, split, selected)
                    if full9(path) != expected_identity:
                        raise ValueError('native source changed while consumer held batch')
                start = stop
            if start != len(codes) or full9(path) != expected_identity:
                raise ValueError('native original row count/identity differs')
