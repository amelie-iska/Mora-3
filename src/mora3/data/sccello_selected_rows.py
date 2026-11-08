"""Row-group selection for audited native scCello sources.

Selected batches carry explicit original row ordinals. They never pretend that
a noncontiguous subset is a contiguous source range. This reader does not select
scientific splits or confer task eligibility; the caller supplies audited rows.
The original contiguous reader remains a separate, unchanged source interface.
"""
from bisect import bisect_left
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .sccello_native_rows import (
    AUDIT_FIELDS, Refusal, SourceBinding, TokenVocabulary,
    _immutable_source, validate_gene_axes,
)


def _prepare_coordinates(values, total_rows):
    """Pure exact integer domain: no bool/float casts, sorting, or deduplication."""
    if type(total_rows) is not int or total_rows < 0:
        raise Refusal("nonnegative exact source row count required")
    if type(values) not in (list, tuple) or len(values) > total_rows:
        raise Refusal("bounded explicit native row coordinate sequence required")
    result = []
    previous = -1
    for value in values:
        if type(value) is not int or not 0 <= value < total_rows or value <= previous:
            raise Refusal("strictly increasing unique in-range native integer row coordinates required")
        result.append(value)
        previous = value
    return tuple(result)


def _planned_row_groups(coordinates, row_group_lengths, total_rows):
    """Pure group-coordinate planning; all source footer row counts conserved."""
    if any(type(n) is not int or n < 0 for n in row_group_lengths):
        raise Refusal("exact nonnegative source row-group lengths required")
    if sum(row_group_lengths) != total_rows:
        raise Refusal("source footer row-group conservation failed")
    plans = []
    start = 0
    for ordinal, count in enumerate(row_group_lengths):
        stop = start + count
        left, right = bisect_left(coordinates, start), bisect_left(coordinates, stop)
        if left != right:
            plans.append((ordinal, start, stop, left, right))
        start = stop
    return tuple(plans)


def _selection_values(value, total_rows):
    if isinstance(value, pa.ChunkedArray):
        if not pa.types.is_integer(value.type) or value.null_count:
            raise Refusal("non-null native integer Arrow coordinates required")
        value = value.to_pylist()
    elif isinstance(value, pa.Array):
        if not pa.types.is_integer(value.type) or value.null_count:
            raise Refusal("non-null native integer Arrow coordinates required")
        value = value.to_pylist()
    elif isinstance(value, np.ndarray):
        if value.ndim != 1 or value.dtype.kind not in ("i", "u"):
            raise Refusal("one-dimensional native integer NumPy coordinates required")
        value = value.tolist()
    elif type(value) not in (list, tuple):
        raise Refusal("explicit integer list/tuple, NumPy or Arrow row coordinates required")
    return _prepare_coordinates(value, total_rows)


@dataclass(frozen=True)
class SelectedNativeGeneBatch:
    binding: SourceBinding
    source_row_ordinals: pa.Array
    gene_token_ids: pa.Array
    gene_expression_nums: pa.Array
    audit_metadata: dict

    def __post_init__(self):
        if (self.source_row_ordinals.type != pa.uint64() or self.source_row_ordinals.null_count or
                len(self.source_row_ordinals) != len(self.gene_token_ids) or
                len(self.source_row_ordinals) != len(self.gene_expression_nums) or
                any(len(v) != len(self.source_row_ordinals) for v in self.audit_metadata.values())):
            raise Refusal("exact explicit selected source coordinates and paired rows required")

    @property
    def num_rows(self):
        return len(self.source_row_ordinals)

    def gene_key_lists(self, vocabulary):
        if not isinstance(vocabulary, TokenVocabulary):
            raise Refusal("exact passive source token vocabulary required")
        flat = pc.list_flatten(self.gene_token_ids)
        keys = pc.take(pa.array(vocabulary.gene_keys + (None,), type=pa.string()), flat)
        lengths = pc.list_value_length(self.gene_token_ids)
        offsets = pa.concat_arrays([pa.array([0], type=pa.int32()), pc.cumulative_sum(lengths)])
        return pa.ListArray.from_arrays(offsets, keys)


def iter_selected_native_gene_batches(binding, source_row_ordinals, *,
                                     batch_size=128, audit_fields=(),
                                     max_positions=8_000_000):
    """Read only intersecting row groups and emit exact selected coordinates.

    Coordinates must be strictly increasing and unique. Any intended stochastic
    reordering or repeated sampling is a separate explicit caller operation.
    Empty coordinates yield no payload batches and claim no source consumption.
    Gold labels remain opt-in audit metadata, never automatic predictor inputs.
    """
    if not isinstance(binding, SourceBinding):
        raise Refusal("audited immutable source binding required")
    if type(batch_size) is not int or not 1 <= batch_size <= 1024:
        raise Refusal("batch_size must be an explicit bounded positive integer")
    if (type(audit_fields) not in (tuple, list) or any(type(x) is not str for x in audit_fields) or
            len(set(audit_fields)) != len(audit_fields) or set(audit_fields) - AUDIT_FIELDS):
        raise Refusal("explicit unique native audit field names required")
    coordinates = _selection_values(source_row_ordinals, binding.rows)
    if not coordinates:
        return
    with _immutable_source(binding) as (stream, check):
        parquet = pq.ParquetFile(stream, memory_map=False, pre_buffer=False,
                                 arrow_extensions_enabled=False)
        try:
            if parquet.metadata.num_rows != binding.rows:
                raise Refusal("audited source row count changed")
            fields = [binding.gene_field, "gene_expression_nums", *audit_fields]
            if any(name not in parquet.schema_arrow.names for name in fields):
                raise Refusal("requested native field absent; no fabricated null column")
            lengths = [parquet.metadata.row_group(i).num_rows
                       for i in range(parquet.metadata.num_row_groups)]
            plans = _planned_row_groups(coordinates, lengths, binding.rows)
            emitted = 0
            for group, group_start, group_stop, _, _ in plans:
                decoded = 0
                for raw in parquet.iter_batches(batch_size=batch_size, columns=fields,
                                                row_groups=[group], use_threads=False):
                    start, stop = group_start + decoded, group_start + decoded + raw.num_rows
                    left, right = bisect_left(coordinates, start), bisect_left(coordinates, stop)
                    if left != right:
                        original = coordinates[left:right]
                        indices = pa.array([row-start for row in original], type=pa.int64())
                        ids = pc.take(raw.column(binding.gene_field), indices)
                        expression = pc.take(raw.column("gene_expression_nums"), indices)
                        native_lengths = validate_gene_axes(ids, expression, max_positions=max_positions)
                        audit = {name: pc.take(raw.column(name), indices) for name in audit_fields}
                        if "length" in audit:
                            declared = audit["length"]
                            if (declared.type != pa.int64() or declared.null_count or
                                    declared.to_pylist() != [int(n) for n in native_lengths]):
                                raise Refusal("selected native length field disagrees with gene axis")
                        check()
                        yield SelectedNativeGeneBatch(binding, pa.array(original, type=pa.uint64()),
                                                      ids, expression, audit)
                        check()
                        emitted += len(original)
                    decoded += raw.num_rows
                if group_start + decoded != group_stop:
                    raise Refusal("selected source row-group conservation failed")
            if emitted != len(coordinates):
                raise Refusal("all requested original source rows must be emitted exactly once")
        finally:
            parquet.close()
