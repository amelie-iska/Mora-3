"""Full original first STRING shard, native b1/b64 and independent immutable-source replay.

Source qualification only; b64 is a *diagnostic* batching equivalence, not the
frozen campaign b1 route and never a training-budget replacement.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json
import os

from mora3.data.admet import canonical
from mora3.data.finite_union_native_rows import decode_disposition, full9
from mora3.data.string_score_current import FAMILY, TASK, SCHEMA, SCORE_AXIS, UNIT
from mora3.research.native_annotation_registry import load_original_annotation_assets, VIEW
from mora3.research.native_annotation_campaign_routes import bind_string
from mora3.research.campaign26_string_original_common import (
    COMMON, audit_string_pair, _one_record, verified_score_packet,
)
from mora3.research.primary_native_routes import finite_view_policy_sha256


def h(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _sig(path, ordinal, raw, scores):
    return (path, ordinal, h(raw), h(canonical(scores)))


def _native_first_shard(binding, source, path, expected_count, *, unit_rows: bool):
    """Process one actual shard and refuse cursor discontinuities or target drift."""
    signatures = []
    cursor_mid = None
    checked_resume = False
    expected_next = None
    native_windows = 0
    initial_cursor = binding.route.initial_cursor
    last_cursor = initial_cursor
    it = binding.route.iter_from(initial_cursor)
    try:
        for window in it:
            batch = window.batch
            if batch.source_shard["path"] != path:
                raise ValueError("STRING original first-shard scan crossed its authorized scope")
            native_windows += 1
            source._view.require_native_rows(batch, split="train")
            if unit_rows:
                record = _one_record(window, source, binding)
                assert record["native_target_positions"] == 4
                signatures.append((record["source_shard_path"], record["source_row_ordinal"],
                                   record["native_payload_sha256"],
                                   record["native_integer_score_sha256"]))
            else:
                packets = source.packets(batch, split="train")
                rows = batch.native_rows.to_pylist()
                if len(packets) != len(rows):
                    raise ValueError("STRING grouped native packet count mismatch")
                prep = binding.prepare(window)
                if (prep.observed_positions != 4 * len(rows)
                        or prep.unavailable_reason is not None or not callable(prep.evaluate)):
                    raise ValueError("STRING batched native target counts/availability drift")
                for ordinal, row, packet in zip(batch.source_row_ordinals, rows, packets, strict=True):
                    raw = row["native_payload"]
                    if (type(raw) is not bytes or h(raw) != row["native_payload_sha256"]
                            or row["source_family"] != FAMILY or row["task_name"] != TASK
                            or row["native_schema"] != SCHEMA or row["split"] != "train"
                            or packet.source_coordinate != (path, ordinal, h(raw))):
                        raise ValueError("STRING batched native source coordinate/identity drift")
                    literal = json.loads(raw)
                    if canonical(literal) != raw:
                        raise ValueError("STRING batched original canonical payload drift")
                    target = literal["targets"]
                    if any(target[x]["unit"] != UNIT for x in SCORE_AXIS):
                        raise ValueError("STRING original integer score unit drift")
                    verified_score_packet({x: target[x]["value"] for x in SCORE_AXIS}, packet.scores)
                    signatures.append(_sig(path, ordinal, raw, packet.scores))
            if unit_rows and expected_next is not None and len(signatures) == 754:
                if (window.cursor_after != expected_next.cursor_after
                        or not window.batch.native_rows.equals(
                            expected_next.batch.native_rows, check_metadata=True)):
                    raise ValueError("STRING mid-shard same-run resume differs from original native row")
                checked_resume = True
            if unit_rows and len(signatures) == 753:
                cursor_mid = window.cursor_after
                restarted = binding.route.iter_from(cursor_mid)
                try:
                    expected_next = next(restarted)
                finally:
                    restarted.close()
                if expected_next.batch.source_shard["path"] != path:
                    raise ValueError("STRING source cursor resumed outside original first shard")
            last_cursor = window.cursor_after
            if len(signatures) >= expected_count:
                break
    finally:
        it.close()
    if (len(signatures) != expected_count or signatures[0][1] != 0
            or any(a[1] >= b[1] for a, b in zip(signatures, signatures[1:]))):
        raise ValueError("STRING native shard did not exhaust the exact eligible source positions")
    if unit_rows and not checked_resume:
        raise ValueError("STRING full original shard has insufficient rows for cursor recovery")
    return signatures, native_windows, h(initial_cursor), h(last_cursor), (
        h(cursor_mid) if cursor_mid is not None else None)


def _direct_first_shard(root, view, path):
    """Independent read of the original Parquet and sealed disposition sidecar."""
    import pyarrow.parquet as pq
    descriptor = json.loads(view._sealed_by_path[path])
    original = root / path
    full = tuple(descriptor["source_full9"])
    if full9(original) != full:
        raise ValueError("Original STRING Parquet source changed before verification")
    codes = decode_disposition(descriptor, root=root)
    signature = []
    total = 0
    for group in pq.ParquetFile(original).iter_batches(batch_size=64, use_threads=False):
        for row in group.to_pylist():
            if (codes[total] == 0 and row["source_family"] == FAMILY
                    and row["task_name"] == TASK and row["native_schema"] == SCHEMA
                    and row["split"] == "train"):
                raw = row["native_payload"]
                if (type(raw) is not bytes or h(raw) != row["native_payload_sha256"]
                        or row["evidence_class"] !=
                            "source_integrated_computed_association_confidence_prior_not_experimental_binding"):
                    raise ValueError("Original direct STRING row lacks authenticated computed evidence")
                literal = json.loads(raw)
                if canonical(literal) != raw or literal["task"] != TASK or literal["schema"] != SCHEMA:
                    raise ValueError("Original direct STRING task canonical payload drift")
                targets = literal["targets"]
                scores = tuple(targets[x]["value"] for x in SCORE_AXIS)
                if any(targets[x]["unit"] != UNIT for x in SCORE_AXIS):
                    raise ValueError("Original direct STRING native score units drift")
                verified_score_packet({x: targets[x]["value"] for x in SCORE_AXIS}, scores)
                signature.append(_sig(path, total, raw, scores))
            total += 1
    if total != len(codes) or full9(original) != full:
        raise ValueError("Original STRING Parquet mutated or did not exhaust its row sidecar")
    expected = [c for c in descriptor["source_task_schema_counts"]
                if c["eligible_view_disposition"] == 0 and c["source_family"] == FAMILY
                and c["task_name"] == TASK and c["split"] == "train"]
    if len(expected) != 1 or expected[0]["rows"] != len(signature):
        raise ValueError("Original STRING sealed train census differs from complete first shard")
    return signature, total


def audit_string_first_shard(root):
    root = Path(root).resolve()
    # Validates both immutable two-window frozen routes, snapshots, adapter hashes and policy.
    pair = audit_string_pair(root)
    view, sources, availability, _, _ = load_original_annotation_assets(root)
    if (finite_view_policy_sha256(view) != VIEW
            or availability["string_score_profile"]["train_rows"] != 474656):
        raise ValueError("Exact STRING first-shard source family/source policy required")
    source = sources["string"]
    candidate = bind_string(source, view, snapshot_sha256=pair["candidate_snapshot"], batch_size=1)
    common = bind_string(source, view, snapshot_sha256=COMMON, batch_size=1)
    bulk = bind_string(source, view, snapshot_sha256=COMMON, batch_size=64)
    path = common.route._paths[0]
    if path != candidate.route._paths[0] or path != bulk.route._paths[0]:
        raise ValueError("STRING frozen/common diagnostic source shard mismatch")
    descriptor = json.loads(view._sealed_by_path[path])
    eligible = [row for row in descriptor["source_task_schema_counts"]
                if row["eligible_view_disposition"] == 0 and row["source_family"] == FAMILY
                and row["task_name"] == TASK and row["split"] == "train"]
    if len(eligible) != 1 or eligible[0]["rows"] <= 753:
        raise ValueError("Exact original first STRING source shard census unavailable")
    expected_count = eligible[0]["rows"]
    frozen, windows, initial, after, midpoint = _native_first_shard(
        common, source, path, expected_count, unit_rows=True)
    batched, grouped_windows, grouped_initial, grouped_after, _ = _native_first_shard(
        bulk, source, path, expected_count, unit_rows=False)
    direct, parquet_rows = _direct_first_shard(root, view, path)
    if frozen != direct or frozen != batched or frozen[:2] != [
        (item["source_shard_path"], item["source_row_ordinal"],
         item["native_payload_sha256"], item["native_integer_score_sha256"])
        for item in pair["original_source_records"]
    ]:
        raise ValueError("STRING first-shard frozen/native-batched/direct original records differ")
    if len(frozen) <= 2 or grouped_windows >= windows:
        raise ValueError("STRING full first-shard batch equivalence not actually exercised")
    return {
        "schema": "Mora3.Campaign26StringFirstFullShard/v1",
        "first_original_shard": path, "original_parquet_rows_including_excluded": parquet_rows,
        "original_train_rows": len(frozen), "independent_original_train_rows": len(direct),
        "original_validation_rows": 0, "original_test_rows": 0,
        "native_first_shard_exhausted": True,
        "native_b1_windows": windows, "native_b64_diagnostic_windows": grouped_windows,
        "frozen_b1_initial_cursor_sha256": initial, "frozen_b1_after_last_eligible_cursor_sha256": after,
        "native_b1_midshard_cursor_sha256": midpoint,
        "diagnostic_b64_initial_cursor_sha256": grouped_initial,
        "diagnostic_b64_after_last_eligible_cursor_sha256": grouped_after,
        "native_batch1_midshard_cursor_recovered": True,
        "native_batch1_equal_batch64_equal_parquet": True,
        "frozen_candidate_common_first_two_equal": pair["candidate_common_native_equal"],
        "original_train_signature_sha256": h(canonical(frozen)),
        "original_native_score_target_positions": 4 * len(frozen),
        "computed_association_not_experimental_binding": True,
        "native_batch64_not_frozen_campaign_route": True,
        "common_source_snapshot_sha256": COMMON, "source_view_policy_sha256": VIEW,
        "total_original_string_train_census": 474656,
        "complete_string_corpus_qualified": False, "global_split_admitted": False,
        "training_admission": False, "optimizer_updates": 0, "counted_proxy_trials": 0,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3 or os.uname().nodename != "iska":
        raise SystemExit("Usage: python -m mora3.research.campaign26_string_first_full_shard ACTIVE_ROOT NEW_JSON")
    dest = Path(sys.argv[2])
    if dest.exists():
        raise FileExistsError("Existing immutable original-source audit must not be overwritten")
    result = audit_string_first_shard(sys.argv[1])
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({"sha256": h(dest.read_bytes()), "rows": result["original_train_rows"],
                      "native_windows": result["native_b1_windows"]}, sort_keys=True))
