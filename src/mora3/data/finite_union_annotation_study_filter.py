"""Layer independently verified task-study exclusions on the original view."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from .finite_union_native_rows import FiniteUnionNativeView, decode_disposition, full9, read_exact


ADDITIONAL_REASON = "heldout-MCSA-direct-native-target-or-label-association-study-evidence"
SCHEMA = "Mora3.AdditionalFiniteNativeStudyOrdinalFilter/v1"
FAMILIES = frozenset({
    "ReactZyme", "UniProt-function-annotations", "UniProt-GOA-assertion-packets", "UniProt-kinetics",
})


def _exact_filter(path, expected_sha256, root):
    descriptor = root / path
    reference = {"path": str(descriptor), "bytes": descriptor.stat().st_size, "sha256": expected_sha256}
    data = json.loads(read_exact(reference, root=root))
    if (data.get("schema") != SCHEMA or data.get("reason") != ADDITIONAL_REASON
            or data.get("source_role") != "train"
            or data.get("source_and_all_parent_masks_unchanged") is not True
            or data.get("full_project_training_ready") is not False
            or data.get("optimizer_launch_authorized") is not False):
        raise ValueError("exact train-only additive source study filter required")
    return data


def _read_evidence(data, root):
    producer_ref = data["source_task_consumption_manifest"]
    proof_ref = data["independent_source_ordinal_verification"]
    producer = json.loads(read_exact(producer_ref, root=root))
    proof = json.loads(read_exact(proof_ref, root=root))
    previous = data["parent_effective_filter"]
    if (producer.get("schema") != "Mora3.AdditionalExactNativeAnnotationStudyRoutes/v1"
            or proof.get("schema") != "Mora3.IndependentAdditionalExactNativeAnnotationStudyRoutes/v1"
            or proof.get("producer_manifest") != producer_ref
            or proof.get("effective_source_filter") != previous
            or producer.get("effective_source_filter") != previous
            or proof.get("complete_original_ordinal_and_evidence_multisets") is not True
            or proof.get("false_positive_exclusions") != 0 or proof.get("false_negative_exclusions") != 0
            or proof.get("unresolved_ReactZyme_upstream_label_associations") != data["unresolved_ReactZyme_upstream_label_associations"]
            or proof.get("mask_or_source_or_master_modified") is not False
            or proof.get("optimizer_authorized") is not False):
        raise ValueError("same-snapshot independent route study proof required")
    rows = tuple(json.loads(line) for line in read_exact(producer["ordinal_overlay"], root=root).splitlines())
    family_counts = Counter(row["source_family"] for row in rows)
    expected_counts = {family: stat.get("direct_study_exclusions", 0)
                       for family, stat in proof["complete_family_counts"].items()}
    if (set(expected_counts) != FAMILIES or family_counts != Counter(expected_counts)
            or data["excluded_taskviews_by_family"] != expected_counts or not rows
            or type(data.get("additional_excluded_taskviews")) is not int
            or data["additional_excluded_taskviews"] != len(rows)
            or producer["proposed_additional_current_train_exclusion_taskviews"] != len(rows)
            or proof["verified_current_train_exclusions"] != len(rows)):
        raise ValueError("exact additional source ordinal and family counts differ")
    return rows


def _bind_exclusions(view, rows, root):
    import pyarrow.parquet as pq

    coordinates = set()
    by_source = defaultdict(list)
    for row in rows:
        shard = row["source_shard"]
        path = shard["path"]
        ordinal = row["absolute_original_row_ordinal"]
        coordinate = path, ordinal
        if (path not in view._sealed_by_path or type(ordinal) is not int
                or not 0 <= ordinal < shard["rows"]
                or row["original_split"] != "train" or row["parent_view_disposition_code"] != 0
                or row["proposed_additive_reason"] != ADDITIONAL_REASON
                or row["scientific_causal_or_same_experiment_identity_proven"] is not False
                or type(row["actual_consumed_source_paper_assertions"]) is not list
                or not row["actual_consumed_source_paper_assertions"]
                or coordinate in coordinates or coordinate in view._ordinal_exclusions):
            raise ValueError("additional study filter requires new retained train ordinals")
        coordinates.add(coordinate)
        by_source[path].append(row)

    columns = ["task_id", "source_family", "task_name", "native_schema", "source_revision", "native_payload_sha256", "split"]
    for path, rows in by_source.items():
        descriptor = json.loads(view._sealed_by_path[path])
        source = root / path
        identity = tuple(descriptor["source_full9"])
        if full9(source) != identity:
            raise ValueError("immutable native source shard identity changed")
        read_exact(descriptor["original_group_sidecar"], root=root)
        codes = decode_disposition(descriptor, root=root)
        table = pq.read_table(source, columns=columns)
        for row in rows:
            ordinal = row["absolute_original_row_ordinal"]
            actual = table.slice(ordinal, 1).to_pylist()[0]
            if (row["source_shard"] != descriptor["source_shard"]
                    or row["original_group_sidecar"] != descriptor["original_group_sidecar"]
                    or codes[ordinal] != 0 or actual["split"] != "train"
                    or any(actual[field] != row[field] for field in columns[:-1])):
                raise ValueError("actual original task, source coordinate or original role differs")
        if full9(source) != identity:
            raise ValueError("immutable native source shard changed during binding")
    return frozenset(coordinates)


def load_study_filtered_native_view(descriptor_path, expected_sha256, *, root):
    """Return the same native-reader ABI for a new immutable data-only scope.

    The earlier constructor, descriptors and source objects keep their prior
    selection semantics. New callers explicitly bind this descriptor.
    """
    root = Path(root)
    data = _exact_filter(descriptor_path, expected_sha256, root)
    rows = _read_evidence(data, root)
    parent = data["parent_effective_filter"]
    read_exact(parent, root=root)
    view = FiniteUnionNativeView.from_effective_filter(root / parent["path"], parent["sha256"], root=root)
    added = _bind_exclusions(view, rows, root)
    counts = dict(view.effective_filter["effective_retained_split_counts"])
    counts["train"] -= len(added)
    if (counts != data["effective_retained_split_counts"]
            or sum(counts.values()) != data["effective_retained_taskviews"]
            or data["previous_excluded_taskviews"] != len(view._ordinal_exclusions)
            or data["combined_excluded_taskviews"] != len(added) + len(view._ordinal_exclusions)):
        raise ValueError("complete previous/new source role count conservation differs")
    view._ordinal_exclusions = view._ordinal_exclusions.union(added)
    view.effective_filter = data
    return view
