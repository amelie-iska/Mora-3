"""Native axis, provenance, immutability, leakage, and closed-stream regression checks."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mora3.data import sccello_native_rows as native

def binding(tmp_path, *, ids=None, expression=None, gene_field="gene_token_ids", extra=None):
    ids = [[25426, 2, 3], [25426, 4]] if ids is None else ids
    expression = [[0.] + [float(i+1) for i in range(len(row)-1)] for row in ids] if expression is None else expression
    fields = {gene_field: pa.array(ids, type=pa.list_(pa.int64())),
              "gene_expression_nums": pa.array(expression, type=pa.list_(pa.float64())),
              "length": pa.array([len(row) for row in ids], type=pa.int64())}
    if extra:
        fields.update(extra)
    path = tmp_path / "source.parquet"
    pq.write_table(pa.table(fields), path, row_group_size=1)
    identity = native.file_identity(path.stat())
    source = dict(path=str(path), bytes=path.stat().st_size,
                  sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  ordinal=41, release_index=3, repository="synthetic-only/scCello",
                  revision="synthetic-fixed-revision", source_role="official_OOD_test",
                  stored_HF_split="train",
                  file_identity={k: identity[k] for k in ("dev", "ino", "size", "mtime_ns", "ctime_ns")})
    receipt = dict(schema="Mora3.scCello.NativeSourceFileAudit.v1", status="COMPLETE_NATIVE_SCAN",
                   source=source, opening_full9=identity, closing_full9=identity,
                   source_rows=len(ids), footer_rows=len(ids), gene_field=gene_field)
    rp = tmp_path / "receipt.json"
    rp.write_text(json.dumps(receipt))
    pin = hashlib.sha256(rp.read_bytes()).hexdigest()
    return native.bind_audited_source(rp, receipt_sha256=pin), path, rp, pin


def test_preserves_source_row_axis_and_rank_order(tmp_path):
    b, _, _, _ = binding(tmp_path)
    batches = list(native.iter_native_gene_batches(b, batch_size=1))
    assert [(x.source_row_start, x.source_row_stop) for x in batches] == [(0, 1), (1, 2)]
    assert batches[0].gene_token_ids.to_pylist() == [[25426, 2, 3]]
    assert batches[0].gene_expression_nums.to_pylist() == [[0., 1., 2.]]
    assert batches[0].binding.source_role == "official_OOD_test"
    assert batches[0].binding.stored_HF_split == "train"


def test_default_never_loads_gold_label_as_predictor_metadata(tmp_path):
    b, _, _, _ = binding(tmp_path, extra={"cell_type": pa.array([12, 42], type=pa.int64())})
    assert all(x.audit_metadata == {} for x in native.iter_native_gene_batches(b))


def test_explicit_audit_metadata_keeps_type_null_and_values(tmp_path):
    b, _, _, _ = binding(tmp_path, extra={"cell_donor_local_ids": pa.array([1, None], type=pa.int64())})
    batches = list(native.iter_native_gene_batches(b, audit_fields=("cell_donor_local_ids", "length")))
    assert batches[0].audit_metadata["cell_donor_local_ids"].type == pa.int64()
    assert batches[0].audit_metadata["cell_donor_local_ids"].to_pylist() == [1, None]


def test_alternate_gene_field_does_not_change_integer_token_semantics(tmp_path):
    b, _, _, _ = binding(tmp_path, gene_field="gene_ensembl_ids")
    assert next(native.iter_native_gene_batches(b)).gene_token_ids.to_pylist()[0] == [25426, 2, 3]


def test_no_implicit_2048_truncation(tmp_path):
    ids = [[25426] + list(range(2, 2052))]
    b, _, _, _ = binding(tmp_path, ids=ids)
    assert len(next(native.iter_native_gene_batches(b)).gene_token_ids[0]) == 2051


def test_paired_axis_length_mismatch_refuses(tmp_path):
    b, _, _, _ = binding(tmp_path, expression=[[0., 1.], [0., 1.]])
    with pytest.raises(native.Refusal, match="paired"):
        list(native.iter_native_gene_batches(b))


@pytest.mark.parametrize("tokens", [[25426, 2, 2], [25426, 1, 3], [25426, 25427], [2, 3]])
def test_duplicates_reserved_unknown_or_missing_cls_refuse(tmp_path, tokens):
    b, _, _, _ = binding(tmp_path, ids=[tokens])
    with pytest.raises(native.Refusal):
        list(native.iter_native_gene_batches(b))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.])
def test_invalid_expression_refuses(tmp_path, value):
    b, _, _, _ = binding(tmp_path, expression=[[0., value, 2.], [0., 1.]])
    with pytest.raises(native.Refusal, match="expression"):
        list(native.iter_native_gene_batches(b))


def test_nonzero_cls_expression_refuses(tmp_path):
    b, _, _, _ = binding(tmp_path, expression=[[1., 1., 2.], [0., 1.]])
    with pytest.raises(native.Refusal, match="CLS"):
        list(native.iter_native_gene_batches(b))


def test_source_changed_after_binding_refuses(tmp_path):
    b, path, _, _ = binding(tmp_path)
    with path.open("ab") as f:
        f.write(b"changed-only-synthetic-fixture")
    with pytest.raises(native.Refusal, match="identity changed"):
        list(native.iter_native_gene_batches(b))


def test_generator_close_checks_source_even_before_next_batch(tmp_path):
    b, path, _, _ = binding(tmp_path)
    stream = native.iter_native_gene_batches(b, batch_size=1)
    next(stream)
    with path.open("ab") as f:
        f.write(b"changed-only-synthetic-fixture")
    with pytest.raises(native.Refusal, match="changed during"):
        stream.close()


def test_symlinked_source_or_ancestor_is_refused(tmp_path):
    b, path, _, _ = binding(tmp_path)
    link = tmp_path / "link.parquet"
    link.symlink_to(path)
    altered = native.SourceBinding(**{**b.__dict__, "path": str(link)})
    with pytest.raises(OSError):
        list(native.iter_native_gene_batches(altered))


def test_wrong_receipt_pin_refuses(tmp_path):
    _, _, receipt, _ = binding(tmp_path)
    with pytest.raises(native.Refusal, match="receipt integrity"):
        native.bind_audited_source(receipt, receipt_sha256="0"*64)


def test_duplicate_receipt_field_refuses(tmp_path):
    _, _, receipt, _ = binding(tmp_path)
    raw = receipt.read_text()
    duplicate = tmp_path / "duplicate-receipt.json"
    duplicate.write_text(raw[:-1] + ', "schema":"Mora3.scCello.NativeSourceFileAudit.v1"}')
    pin = hashlib.sha256(duplicate.read_bytes()).hexdigest()
    with pytest.raises(native.Refusal, match="duplicate"):
        native.bind_audited_source(duplicate, receipt_sha256=pin)


def test_missing_requested_native_metadata_not_fabricated(tmp_path):
    b, _, _, _ = binding(tmp_path)
    with pytest.raises(native.Refusal, match="field absent"):
        list(native.iter_native_gene_batches(b, audit_fields=("cell_disease",)))


def test_batch_bound_refuses_without_truncation(tmp_path):
    b, _, _, _ = binding(tmp_path)
    with pytest.raises(native.Refusal, match="bounded batch positions"):
        list(native.iter_native_gene_batches(b, max_positions=2))


def test_sliced_arrow_offsets_conserve_paired_axes():
    ids = pa.array([[25426, 9, 8], [25426, 2]], type=pa.list_(pa.int64())).slice(1, 1)
    ex = pa.array([[0., 8., 9.], [0., 3.]], type=pa.list_(pa.float64())).slice(1, 1)
    assert native.validate_gene_axes(ids, ex).tolist() == [2]


def vocabulary():
    pairs = [{"ordinal": i, "key": "<pad>" if i == 0 else "<mask>" if i == 1 else "EXACT_SYNTHETIC_KEY_"+str(i),
              "value": {"value": i}} for i in range(native.TOKEN_COUNT)]
    return dict(pairs=pairs, source_sha256="a"*64)


def test_vocab_and_cls_roles_remain_separate(tmp_path):
    b, _, _, _ = binding(tmp_path)
    v = native.TokenVocabulary.from_passive_mapping(vocabulary())
    first = next(native.iter_native_gene_batches(b))
    assert first.gene_key_lists(v).to_pylist()[0] == [None, "EXACT_SYNTHETIC_KEY_2", "EXACT_SYNTHETIC_KEY_3"]
    assert len(v.gene_keys) == 25426 and v.gene_keys[:2] == ("<pad>", "<mask>")


def test_float_token_dictionary_values_refuse():
    d = vocabulary()
    d["pairs"][2]["value"]["value"] = 2.0
    with pytest.raises(native.Refusal, match="native token IDs"):
        native.TokenVocabulary.from_passive_mapping(d)


def test_duplicate_vocabulary_coordinate_refuses():
    d = vocabulary()
    d["pairs"][2]["value"]["value"] = 3
    with pytest.raises(native.Refusal, match="native token IDs"):
        native.TokenVocabulary.from_passive_mapping(d)
