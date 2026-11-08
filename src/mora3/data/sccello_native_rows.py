"""Bounded passive scCello source batches with immutable provenance.

This reader binds an earlier whole-byte/native audit to the current file's full
stat identity. It does not repeat a whole-file hash for every training batch,
assign splits, admit a dataset, remove median-selected truncation, or reinterpret
provider-transformed expression as counts. Metadata is an audit sidecar, never
automatically an unknown-label predictor input.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

TOKEN_COUNT = 25426
SOURCE_CLS_ID = 25426
IDENTITY_FIELDS = ("dev", "ino", "mode", "uid", "gid", "nlink", "size", "mtime_ns", "ctime_ns")
AUDIT_FIELDS = frozenset(("cell_dataset_id", "cell_donor_local_ids", "cell_ct_ontology",
                         "cell_type", "cell_tissue", "cell_tissue_ontology", "cell_disease",
                         "cell_assay_ids", "cell_dev", "cell_counts", "length"))


class Refusal(ValueError):
    """The exact source or native representation contract was not established."""


def file_identity(value):
    return dict(dev=value.st_dev, ino=value.st_ino, mode=value.st_mode,
                uid=value.st_uid, gid=value.st_gid, nlink=value.st_nlink,
                size=value.st_size, mtime_ns=value.st_mtime_ns, ctime_ns=value.st_ctime_ns)


def _sha(value):
    return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None


def _integer(value):
    return type(value) is int and value >= 0


def _open_regular(path):
    """Open every absolute path component without following a symlink."""
    path = os.fspath(path)
    if not os.path.isabs(path) or any(p in (".", "..") for p in path.split("/")):
        raise Refusal("canonical absolute source path required")
    handles = []
    try:
        fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
        handles.append(fd)
        for component in Path(path).parts[1:-1]:
            fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            handles.append(fd)
        fd = os.open(Path(path).name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        handles.append(fd)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise Refusal("regular immutable source file required")
        return handles
    except BaseException:
        for handle in reversed(handles):
            os.close(handle)
        raise


@dataclass(frozen=True)
class SourceBinding:
    path: str
    source_sha256: str
    receipt_sha256: str
    source_ordinal: int
    release_index: int
    repository: str
    revision: str
    source_role: str
    stored_HF_split: str | None
    gene_field: str
    rows: int
    identity_items: tuple

    @property
    def expected_identity(self):
        return dict(self.identity_items)


def _receipt_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise Refusal("duplicate source receipt JSON key")
        result[key] = value
    return result


def bind_audited_source(receipt_path, *, receipt_sha256):
    """Pin one completed audit; source-role strings do not confer admission."""
    if not _sha(receipt_sha256):
        raise Refusal("explicit audit receipt SHA256 required")
    handles = _open_regular(receipt_path)
    try:
        before = file_identity(os.fstat(handles[-1]))
        if before["size"] > 4 * 1024**2:
            raise Refusal("bounded source audit receipt required")
        raw = os.pread(handles[-1], before["size"] + 1, 0)
        if file_identity(os.fstat(handles[-1])) != before or hashlib.sha256(raw).hexdigest() != receipt_sha256:
            raise Refusal("audit receipt integrity changed")
    finally:
        for handle in reversed(handles):
            os.close(handle)
    try:
        receipt = json.loads(raw, object_pairs_hook=_receipt_pairs,
                             parse_constant=lambda value: (_ for _ in ()).throw(Refusal("nonfinite receipt JSON")))
        source = receipt["source"]
        identity = receipt["closing_full9"]
        if (receipt["schema"] != "Mora3.scCello.NativeSourceFileAudit.v1" or
                receipt["status"] != "COMPLETE_NATIVE_SCAN" or
                receipt["opening_full9"] != identity or
                set(identity) != set(IDENTITY_FIELDS) or
                any(not _integer(v) for v in identity.values()) or
                not stat.S_ISREG(identity["mode"]) or
                receipt["source_rows"] != receipt["footer_rows"] or
                not _integer(receipt["source_rows"]) or
                source["bytes"] != identity["size"] or not _sha(source["sha256"]) or
                any(source["file_identity"].get(k) != identity[k]
                    for k in ("dev", "ino", "size", "mtime_ns", "ctime_ns")) or
                any(not _integer(source[k]) for k in ("ordinal", "release_index")) or
                any(type(source[k]) is not str or not source[k]
                    for k in ("path", "repository", "revision", "source_role")) or
                receipt["gene_field"] not in ("gene_token_ids", "gene_ensembl_ids")):
            raise Refusal("complete exact native audit binding required")
    except (KeyError, TypeError) as error:
        raise Refusal("source audit binding fields missing") from error
    return SourceBinding(source["path"], source["sha256"], receipt_sha256,
                         source["ordinal"], source["release_index"], source["repository"],
                         source["revision"], source["source_role"], source.get("stored_HF_split"),
                         receipt["gene_field"], receipt["source_rows"], tuple(sorted(identity.items())))


@dataclass(frozen=True)
class TokenVocabulary:
    gene_keys: tuple
    source_sha256: str

    @classmethod
    def from_passive_mapping(cls, decoded):
        """Consume the existing passive numeric decoder, never unpickle."""
        try:
            pairs = decoded["pairs"]
            if not _sha(decoded["source_sha256"]) or len(pairs) != TOKEN_COUNT:
                raise Refusal("exact immutable source token vocabulary required")
            keys = [None] * TOKEN_COUNT
            seen = set()
            for ordinal, pair in enumerate(pairs):
                value, key = pair["value"]["value"], pair["key"]
                if (pair["ordinal"] != ordinal or type(value) is not int or
                        not 0 <= value < TOKEN_COUNT or keys[value] is not None or
                        type(key) is not str or not key or key in seen):
                    raise Refusal("unique exact native token IDs and gene keys required")
                keys[value] = key
                seen.add(key)
            if keys[:2] != ["<pad>", "<mask>"]:
                raise Refusal("native reserved token roles changed")
        except (KeyError, TypeError) as error:
            raise Refusal("passive numeric vocabulary fields missing") from error
        return cls(tuple(keys), decoded["source_sha256"])


@dataclass(frozen=True)
class NativeGeneBatch:
    """Physical source rows, never an assertion of globally unique cells."""
    binding: SourceBinding
    source_row_start: int
    gene_token_ids: pa.Array
    gene_expression_nums: pa.Array
    audit_metadata: dict

    @property
    def num_rows(self):
        return len(self.gene_token_ids)

    @property
    def source_row_stop(self):
        return self.source_row_start + self.num_rows

    def gene_key_lists(self, vocabulary):
        """CLS maps to null as a separate source role; never modify vocabulary."""
        flat = pc.list_flatten(self.gene_token_ids)
        keys = pc.take(pa.array(vocabulary.gene_keys + (None,), type=pa.string()), flat)
        lengths = pc.list_value_length(self.gene_token_ids).to_numpy()
        offsets = np.concatenate((np.array([0], dtype=np.int32), np.cumsum(lengths, dtype=np.int32)))
        return pa.ListArray.from_arrays(pa.array(offsets), keys)


def validate_gene_axes(ids, expression, *, max_positions=8_000_000):
    """Vectorized paired source axes, preserving their original rank/order."""
    if (not pa.types.is_list(ids.type) or ids.type.value_type != pa.int64() or
            not pa.types.is_list(expression.type) or expression.type.value_type != pa.float64()):
        raise Refusal("native list<int64>/list<double> schema required")
    if ids.null_count or expression.null_count:
        raise Refusal("null native paired lists")
    gl = pc.list_value_length(ids).to_numpy()
    el = pc.list_value_length(expression).to_numpy()
    if not np.array_equal(gl, el) or np.any(gl == 0):
        raise Refusal("nonempty exact paired gene/expression axes required")
    if not _integer(max_positions) or int(gl.sum()) > max_positions:
        raise Refusal("bounded batch positions required; source is never truncated")
    tokens = pc.list_flatten(ids)
    values = pc.list_flatten(expression)
    if tokens.null_count or values.null_count:
        raise Refusal("null native gene/expression leaf values")
    iv, ev = tokens.to_numpy(), values.to_numpy()
    if (np.any(iv < 2) or np.any(iv > SOURCE_CLS_ID) or
            not np.all(np.isfinite(ev)) or np.any(ev < 0)):
        raise Refusal("unknown/reserved tokens or nonfinite/negative native expression")
    starts = np.cumsum(gl) - gl
    if np.any(iv[starts] != SOURCE_CLS_ID) or np.any(ev[starts] != 0):
        raise Refusal("exact separate source CLS prefix/expression required")
    rows = np.repeat(np.arange(len(gl), dtype=np.int64), gl)
    codes = np.sort(rows * (SOURCE_CLS_ID + 1) + iv)
    if np.any(codes[1:] == codes[:-1]):
        raise Refusal("duplicate token identity within source row")
    return gl


@contextmanager
def _immutable_source(binding):
    handles = _open_regular(binding.path)
    source = None
    check = None
    try:
        if file_identity(os.fstat(handles[-1])) != binding.expected_identity:
            raise Refusal("audited source identity changed")
        parents = [file_identity(os.fstat(fd)) for fd in handles[:-1]]
        source = os.fdopen(os.dup(handles[-1]), "rb", buffering=0)

        def check():
            if file_identity(os.fstat(handles[-1])) != binding.expected_identity:
                raise Refusal("source changed during native consumption")
            for fd, before in zip(handles[:-1], parents):
                after = file_identity(os.fstat(fd))
                if any(after[k] != before[k] for k in ("dev", "ino", "mode", "uid", "gid")):
                    raise Refusal("source ancestor security identity changed")

        yield source, check
    finally:
        try:
            if check is not None:
                check()
        finally:
            if source is not None:
                source.close()
            for handle in reversed(handles):
                os.close(handle)


def iter_native_gene_batches(binding, *, batch_size=128, audit_fields=(), max_positions=8_000_000):
    """Stream exact source rows; caller must separately establish task eligibility."""
    if not isinstance(binding, SourceBinding):
        raise Refusal("audited immutable source binding required")
    if type(batch_size) is not int or not 1 <= batch_size <= 1024:
        raise Refusal("batch_size must be an explicit bounded positive integer")
    if len(set(audit_fields)) != len(audit_fields) or set(audit_fields) - AUDIT_FIELDS:
        raise Refusal("explicit unique native audit field names required")
    with _immutable_source(binding) as (stream, check):
        parquet = pq.ParquetFile(stream, memory_map=False, pre_buffer=False,
                                 arrow_extensions_enabled=False)
        try:
            if parquet.metadata.num_rows != binding.rows:
                raise Refusal("audited source row count changed")
            fields = [binding.gene_field, "gene_expression_nums", *audit_fields]
            if any(name not in parquet.schema_arrow.names for name in fields):
                raise Refusal("requested native field absent; no fabricated null column")
            row_start = 0
            for batch in parquet.iter_batches(batch_size=batch_size, columns=fields, use_threads=False):
                ids, expression = batch.column(binding.gene_field), batch.column("gene_expression_nums")
                lengths = validate_gene_axes(ids, expression, max_positions=max_positions)
                if "length" in audit_fields:
                    source_length = batch.column("length")
                    if source_length.type != pa.int64() or source_length.null_count or not np.array_equal(source_length.to_numpy(), lengths):
                        raise Refusal("source native length field disagrees with gene axis")
                check()
                yield NativeGeneBatch(binding, row_start, ids, expression,
                                      {name: batch.column(name) for name in audit_fields})
                check()
                row_start += batch.num_rows
            if row_start != binding.rows:
                raise Refusal("source row conservation failed")
        finally:
            parquet.close()
