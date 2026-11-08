"""Apply a separately verified source-only exclusion for one inverse predictor/target overlap."""
import gzip
from hashlib import sha256
import json
from pathlib import Path

from .finite_union_annotation_study_filter import load_study_filtered_native_view
from .finite_union_native_rows import full9, read_exact


SCHEMA = "Mora3.FiniteNativeInverseLiteralTargetOrdinalFilter/v1"
FAMILY = "UniProt-full-record-annotations"
TASK = "source_UniProt_annotation_conditioned_canonical_sequence"
REASON = "literal_target_fragment_exposed_by_original_annotation_kinetic_substrate_condition"
CONDITION_PREFIX = "/conditions/source_kinetic_conditions/"
TARGET_POINTER = "/targets/canonical_sequence"


def _digest(raw):
    return sha256(raw).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _pointer(value, path):
    if type(path) is not str or not path.startswith("/"):
        raise ValueError("absolute native JSON pointer required")
    for part in path.split("/")[1:]:
        part = part.replace("~1", "/").replace("~0", "~")
        if type(value) is list:
            if not part.isascii() or not part.isdecimal() or str(int(part)) != part or int(part) >= len(value):
                raise ValueError("exact original native array coordinate required")
            value = value[int(part)]
        elif type(value) is dict and part in value:
            value = value[part]
        else:
            raise ValueError("exact native source field required")
    return value


def _require_literal_witness(payload, original_page, evidence):
    condition_ptr = evidence["condition_field_JSON_pointer"]
    original_ptr = evidence["exact_original_kinetic_substrate_pointer"]
    target_ptr = evidence["target_field_JSON_pointer"]
    entry_index = evidence["original_source_entry_row_ordinal"]
    if (
        payload.get("schema") != "mora3.CorpusTask/v1" or payload.get("task") != TASK
        or type(entry_index) is not int or entry_index < 0
        or type(condition_ptr) is not str or not condition_ptr.startswith(CONDITION_PREFIX)
        or not condition_ptr.endswith("/substrate_label") or target_ptr != TARGET_POINTER
        or type(original_ptr) is not str or not original_ptr.startswith(f"/results/{entry_index}/comments/")
        or not original_ptr.endswith("/substrate")
    ):
        raise ValueError("only the exact declared kinetic condition and canonical target are in this guard")
    condition = _pointer(payload, condition_ptr)
    target = _pointer(payload, target_ptr)
    original_entry = _pointer(original_page, f"/results/{entry_index}")
    native_substrate = _pointer(original_page, original_ptr)
    if (
        type(condition) is not str or type(target) is not str or not condition or not target
        or native_substrate != condition or original_entry["sequence"]["value"] != target
        or original_entry["primaryAccession"] != evidence["source_record_id"]
        or _digest(_canonical(payload["conditions"])) != evidence["canonical_conditions_sha256"]
        or _digest(_canonical(payload["targets"])) != evidence["canonical_targets_sha256"]
        or _digest(condition.encode("utf-8")) != evidence["condition_UTF8_value_sha256"]
        or _digest(target.encode("utf-8")) != evidence["target_UTF8_value_sha256"]
    ):
        raise ValueError("original source and canonical predictor/target bytes differ from the exact witness")
    audit = original_entry["entryAudit"]
    revision = f'UniProt2026_03;entryVersion{audit["entryVersion"]};sequenceVersion{audit["sequenceVersion"]}'
    if revision != evidence["source_revision"]:
        raise ValueError("original native entry or sequence version differs")
    offset = evidence["token_condition_character_offset0"]
    residue = evidence["target_canonical_residue_offset0"]
    length = evidence["exact_literal_uppercase_token_length"]
    if (
        type(offset) is not int or type(residue) is not int or type(length) is not int
        or length < 8 or not 0 <= offset <= len(condition) - length or not 0 <= residue <= len(target) - length
    ):
        raise ValueError("literal token offsets and complete original target are required")
    token = condition[offset:offset + length]
    adjacent = condition[max(offset - 1, 0):offset] + condition[offset + length:offset + length + 1]
    if (
        any(not "A" <= letter <= "Z" for letter in token)
        or any(letter.isalnum() or letter == "_" for letter in adjacent)
        or target[residue:residue + length] != token
        or _digest(token.encode("ascii")) != evidence["exact_literal_uppercase_token_SHA256"]
    ):
        raise ValueError("the standalone literal target fragment does not match original condition and target")


def _bind_original_row(view, evidence, root):
    shard = evidence["original_shard"]
    path, ordinal = shard["path"], evidence["absolute_original_row_ordinal"]
    coordinate = path, ordinal
    sealed = view._sealed_by_path.get(path)
    if (
        sealed is None or type(ordinal) is not int or not 0 <= ordinal < shard["rows"]
        or coordinate in view._ordinal_exclusions or evidence["original_split"] != "train"
        or evidence["parent_effective_disposition"] != "retained_train"
        or evidence["source_family"] != FAMILY or evidence["task_name"] != TASK
        or evidence["guard_reason"] != REASON or evidence["exact_positive_literal_witness"] is not True
        or evidence["role_reassigned"] is not False or evidence["original_masks_mutated"] is not False
        or evidence["global_leakage_free_claim"] is not False
    ):
        raise ValueError("one exact retained native inverse train coordinate required")
    descriptor = json.loads(sealed)
    if (
        descriptor["source_shard"] != shard
        or descriptor["original_group_sidecar"] != evidence["original_group_sidecar"]
        or descriptor["source_full9"] != evidence["original_source_full9"]
        or list(full9(root / path)) != evidence["original_source_full9"]
    ):
        raise ValueError("immutable original inverse source shard or group metadata changed")
    read_exact(shard, root=root)
    matches = []
    for batch in view.iter_native_task_batches(split="train", source_paths=(path,), batch_size=64):
        if ordinal not in batch.source_row_ordinals:
            continue
        view.require_native_rows(batch, split="train")
        index = batch.source_row_ordinals.index(ordinal)
        matches.append(batch.native_rows.slice(index, 1).to_pylist()[0])
    if len(matches) != 1:
        raise ValueError("exact original retained source coordinate was not uniquely issued")
    row, = matches
    raw = row["native_payload"]
    if (
        type(raw) is not bytes or row["native_schema"] != "mora3.CorpusTask/v1"
        or row["record_id"] != evidence["source_record_id"] or row["split"] != "train"
        or row["source_sha256"] != evidence["original_raw_page"]["sha256"]
        or any(row[key] != evidence[key] for key in ("source_family", "task_name", "source_revision", "task_id", "native_payload_sha256"))
        or _digest(raw) != evidence["native_payload_sha256"]
    ):
        raise ValueError("native inverse payload and exact original source coordinates differ")
    original = read_exact(evidence["original_raw_page"], root=root)
    if original.startswith(b"\x1f\x8b"):
        original = gzip.decompress(original)
    _require_literal_witness(json.loads(raw), json.loads(original), evidence)
    return coordinate


def load_inverse_literal_filtered_native_view(descriptor_path, expected_sha256, *, root):
    """Return the unchanged native-reader ABI with one explicit, source-only extra omission."""
    root = Path(root)
    path = root / descriptor_path
    descriptor = json.loads(read_exact({
        "path": str(path), "bytes": path.stat().st_size, "sha256": expected_sha256,
    }, root=root))
    if (
        descriptor.get("schema") != SCHEMA or descriptor.get("reason") != REASON
        or descriptor.get("source_role") != "train"
        or type(descriptor.get("additional_excluded_taskviews")) is not int
        or descriptor["additional_excluded_taskviews"] != 1
        or descriptor.get("source_roles_shards_and_original_masks_unchanged") is not True
        or descriptor.get("full_project_training_ready") is not False
        or descriptor.get("optimizer_launch_authorized") is not False
    ):
        raise ValueError("exact source-only inverse literal target filter required")
    source_ref = descriptor["source_inverse_literal_guard_proposal"]
    source = json.loads(read_exact(source_ref, root=root))
    proof = json.loads(read_exact(descriptor["independent_native_ordinal_and_literal_proof"], root=root))
    parent = descriptor["parent_effective_filter"]
    if (
        source.get("schema") != "Mora3.UniProtInverseExactNativeLiteralTargetExclusionProposal/v1"
        or source["parent_effective_filter"] != parent
        or proof.get("schema") != "Mora3.UniProtIndependentCurrentNativeAnnotationAndLiteralTargetVerification/v1"
        or proof["prospective_one_row_guard"] != source_ref
        or proof["source_audit"] != source["source_audit"]
        or proof["exact_original_raw_substrate_target_collision_count"] != 1
        or proof["exact_native_condition_target_digest_scope"] is not True
        or proof["existing_ordinal_codes_and_exclusions_read_directly"] is not True
        or proof["original_data_sha_checked_for_all_selected_source_shards"] is not True
        or proof["optimizer_training_authorized"] is not False
        or source["optimizer_training_authorized"] is not False
        or type(source["physical_source_rows_rewritten"]) is not int
        or source["physical_source_rows_rewritten"] != 0 or source["underlying_masks_mutated"] is not False
        or source["new_source_role_assignment"] is not False or len(source["candidates"]) != 1
        or source["proposed_inverse_only_effective_rows"] != source["original_source_retained_inverse_rows"] - 1
        or descriptor["exact_inverse_before"] != source["original_source_retained_inverse_rows"]
        or descriptor["exact_inverse_after"] != source["proposed_inverse_only_effective_rows"]
        or descriptor["exact_forward_unchanged"] != source["original_source_retained_forward_rows"]
    ):
        raise ValueError("same-snapshot independent inverse native target fragment proof required")
    read_exact(parent, root=root)
    read_exact(source["source_audit"], root=root)
    view = load_study_filtered_native_view(root / parent["path"], parent["sha256"], root=root)
    coordinate = _bind_original_row(view, source["candidates"][0], root)
    counts = dict(view.effective_filter["effective_retained_split_counts"])
    counts["train"] -= 1
    before = len(view._ordinal_exclusions)
    if (
        counts != descriptor["effective_retained_split_counts"]
        or sum(counts.values()) != descriptor["effective_retained_taskviews"]
        or before != descriptor["previous_excluded_taskviews"]
        or before + 1 != descriptor["combined_excluded_taskviews"]
    ):
        raise ValueError("source role, task and ordinal count conservation differs")
    view._ordinal_exclusions = view._ordinal_exclusions.union({coordinate})
    view.effective_filter = descriptor
    return view
