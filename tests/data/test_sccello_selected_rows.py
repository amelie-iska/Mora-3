"""Exact selected coordinates, row-group pushdown, and target-sidecar isolation."""
import hashlib
import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mora3.data import sccello_native_rows as base
from mora3.data import sccello_selected_rows as selected


def source_binding(tmp_path, *, long_first=False):
    ids = [[25426, i+2] for i in range(8)]
    if long_first:
        ids[0] = [25426] + list(range(2, 2052))
    expression = [[0.] + [float(i+.5)] * (len(row)-1) for i, row in enumerate(ids)]
    donor = [None if i % 2 == 0 else i for i in range(8)]
    fields = {
        "gene_token_ids": pa.array(ids, type=pa.list_(pa.int64())),
        "gene_expression_nums": pa.array(expression, type=pa.list_(pa.float64())),
        "length": pa.array([len(x) for x in ids], type=pa.int64()),
        "cell_donor_local_ids": pa.array(donor, type=pa.int64()),
        "cell_ct_ontology": pa.array(["SYNTHETIC_ONLY_CL_"+str(i) for i in range(8)]),
    }
    path = tmp_path / "source.parquet"
    pq.write_table(pa.table(fields), path, row_group_size=2)
    identity = base.file_identity(path.stat())
    source = dict(path=str(path), bytes=path.stat().st_size,
                  sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  ordinal=41, release_index=3, repository="synthetic-only/scCello",
                  revision="synthetic-fixed-revision", source_role="official_OOD_test",
                  stored_HF_split="train",
                  file_identity={k: identity[k] for k in ("dev","ino","size","mtime_ns","ctime_ns")})
    receipt = dict(schema="Mora3.scCello.NativeSourceFileAudit.v1", status="COMPLETE_NATIVE_SCAN",
                   source=source, opening_full9=identity, closing_full9=identity,
                   source_rows=8, footer_rows=8, gene_field="gene_token_ids")
    rp = tmp_path / "receipt.json"
    rp.write_text(json.dumps(receipt))
    binding = base.bind_audited_source(rp, receipt_sha256=hashlib.sha256(rp.read_bytes()).hexdigest())
    return binding, path, fields


def flatten(batches, field):
    return [item for batch in batches for item in getattr(batch, field).to_pylist()]


def test_exact_noncontiguous_coordinates_and_paired_source_values(tmp_path):
    binding, _, fields = source_binding(tmp_path)
    batches = list(selected.iter_selected_native_gene_batches(binding, [1,4,7], batch_size=2))
    assert flatten(batches, "source_row_ordinals") == [1,4,7]
    assert flatten(batches, "gene_token_ids") == fields["gene_token_ids"].take(pa.array([1,4,7])).to_pylist()
    assert flatten(batches, "gene_expression_nums") == fields["gene_expression_nums"].take(pa.array([1,4,7])).to_pylist()
    assert all(batch.binding.source_role == "official_OOD_test" for batch in batches)
    assert all(not hasattr(batch, "source_row_start") and not hasattr(batch, "source_row_stop") for batch in batches)


def test_only_intersecting_row_groups_requested(tmp_path, monkeypatch):
    binding, _, _ = source_binding(tmp_path)
    real = selected.pq.ParquetFile
    observed = []
    class TrackingParquet:
        def __init__(self, *args, **kwargs):
            assert kwargs["arrow_extensions_enabled"] is False
            assert kwargs["memory_map"] is False
            self.inner = real(*args, **kwargs)
        def __getattr__(self, key):
            return getattr(self.inner, key)
        def iter_batches(self, *args, **kwargs):
            observed.append(kwargs["row_groups"])
            return self.inner.iter_batches(*args, **kwargs)
        def close(self):
            self.inner.close()
    monkeypatch.setattr(selected.pq, "ParquetFile", TrackingParquet)
    batches = list(selected.iter_selected_native_gene_batches(binding, [1,6]))
    assert observed == [[0],[3]]
    assert flatten(batches, "source_row_ordinals") == [1,6]


def test_default_labels_not_loaded_or_reclassified(tmp_path):
    binding, _, _ = source_binding(tmp_path)
    batches = list(selected.iter_selected_native_gene_batches(binding, [0,2]))
    assert all(batch.audit_metadata == {} for batch in batches)


def test_explicit_metadata_retains_native_null_types_and_gold_role(tmp_path):
    binding, _, fields = source_binding(tmp_path)
    batches = list(selected.iter_selected_native_gene_batches(
        binding, [0,3,6], audit_fields=("cell_donor_local_ids","cell_ct_ontology","length")))
    donor = [v for batch in batches for v in batch.audit_metadata["cell_donor_local_ids"].to_pylist()]
    assert donor == [None,3,None]
    assert all(batch.audit_metadata["cell_donor_local_ids"].type == pa.int64() for batch in batches)
    labels = [v for batch in batches for v in batch.audit_metadata["cell_ct_ontology"].to_pylist()]
    assert labels == ["SYNTHETIC_ONLY_CL_0","SYNTHETIC_ONLY_CL_3","SYNTHETIC_ONLY_CL_6"]


@pytest.mark.parametrize("coordinates", [
    [True], [1.0], [None], [-1], [8], [2,1], [1,1], ["1"],
    np.array([1.]), np.array([True]), np.array([[1]]),
    pa.array([1,None], type=pa.uint64()), pa.array([1.], type=pa.float64()),
])
def test_invalid_coordinate_domains_refuse_without_coercion(tmp_path, coordinates):
    binding, _, _ = source_binding(tmp_path)
    with pytest.raises(base.Refusal):
        list(selected.iter_selected_native_gene_batches(binding, coordinates))


@pytest.mark.parametrize("coordinates", [
    [0,3,7], (0,3,7), np.array([0,3,7],dtype=np.uint64),
    pa.array([0,3,7],type=pa.uint64()),
    pa.chunked_array([[0,3],[7]],type=pa.uint64()),
])
def test_supported_integer_coordinate_vectors_roundtrip(tmp_path, coordinates):
    binding, _, _ = source_binding(tmp_path)
    batches = list(selected.iter_selected_native_gene_batches(binding, coordinates))
    assert flatten(batches, "source_row_ordinals") == [0,3,7]
    assert all(batch.source_row_ordinals.type == pa.uint64() for batch in batches)


def test_coordinate_snapshot_cannot_be_changed_mid_stream(tmp_path):
    binding, _, _ = source_binding(tmp_path)
    coordinates = [0,3,7]
    stream = selected.iter_selected_native_gene_batches(binding, coordinates, batch_size=1)
    first = next(stream)
    coordinates[:] = [1]
    rest = list(stream)
    assert flatten([first,*rest], "source_row_ordinals") == [0,3,7]


def test_selected_axes_bounded_without_truncating_or_vetting_unconsumed_rows(tmp_path):
    binding, _, _ = source_binding(tmp_path, long_first=True)
    batch = next(selected.iter_selected_native_gene_batches(binding, [1], max_positions=2))
    assert batch.gene_token_ids.to_pylist() == [[25426,3]]
    with pytest.raises(base.Refusal, match="bounded batch positions"):
        list(selected.iter_selected_native_gene_batches(binding, [0], max_positions=2))


def test_selected_generator_close_rechecks_immutable_source(tmp_path):
    binding, path, _ = source_binding(tmp_path)
    stream = selected.iter_selected_native_gene_batches(binding, [0,7], batch_size=1)
    next(stream)
    with path.open("ab") as f:
        f.write(b"synthetic-fixture-only-change")
    with pytest.raises(base.Refusal, match="changed during"):
        stream.close()


def test_requested_absent_audit_field_not_fabricated(tmp_path):
    binding, _, _ = source_binding(tmp_path)
    with pytest.raises(base.Refusal, match="field absent"):
        list(selected.iter_selected_native_gene_batches(binding, [0], audit_fields=("cell_disease",)))


def test_empty_selection_yields_no_source_payload(tmp_path):
    binding, _, _ = source_binding(tmp_path)
    assert list(selected.iter_selected_native_gene_batches(binding, [])) == []


def test_selected_gene_key_mapping_keeps_cls_separate(tmp_path):
    binding, _, _ = source_binding(tmp_path)
    pairs = [{"ordinal":i,"key":"<pad>" if i==0 else "<mask>" if i==1 else "EXACT_SYNTHETIC_KEY_"+str(i),
              "value":{"value":i}} for i in range(base.TOKEN_COUNT)]
    vocab = base.TokenVocabulary.from_passive_mapping(dict(pairs=pairs,source_sha256="a"*64))
    batches = list(selected.iter_selected_native_gene_batches(binding, [1,6]))
    keys = [row for batch in batches for row in batch.gene_key_lists(vocab).to_pylist()]
    assert keys == [[None,"EXACT_SYNTHETIC_KEY_3"],[None,"EXACT_SYNTHETIC_KEY_8"]]
    assert len(vocab.gene_keys) == 25426
