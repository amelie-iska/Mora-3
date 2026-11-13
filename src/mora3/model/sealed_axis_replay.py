"""Reconstruct historical constructor axes from pinned proofs without reading row payloads."""

from pathlib import Path

from mora3.data.finite_union_native_rows import ROLES, decode_disposition, full9
from mora3.data.pdb_masked_coordinates import (
    FAMILY, SCHEMA, TASK, PDBCoordinateAxes, native_view_policy_sha256,
)


def retained_roles(view, expected, root):
    paths = {path for path, _ in expected}
    descriptors = {row["source_shard"]["path"]: row for row in view.descriptors if row["source_shard"]["path"] in paths}
    if descriptors.keys() != paths:
        raise ValueError("Frozen source coordinates are absent from the exact current view")
    inverse = {code: role for role, code in ROLES.items()}
    actual = {}
    for path, descriptor in descriptors.items():
        source = Path(root) / path
        if source.resolve() != source or not source.is_relative_to(root) \
                or full9(source) != tuple(descriptor["source_full9"]):
            raise ValueError("Frozen source envelope original shard identity changed")
        for ordinal, code in enumerate(decode_disposition(descriptor, root=root)):
            key = path, ordinal
            if code in inverse and key not in view._ordinal_exclusions:
                actual[key] = inverse[code]
    if actual != expected:
        raise ValueError("Frozen source envelope does not cover exact retained original coordinates and roles")
    return descriptors


def frozen_pdb_axes(view, proof, root):
    if (proof["complete_original_labels_observations_and_targets_preserved"] is not True
            or proof["validation_or_test_packets"] != 0 or proof["global_training_authorization"] is not False):
        raise ValueError("Exact independent PDB axis proof required")
    historical = proof["source_axes"]
    coordinates = tuple(tuple(row) for row in historical["source_coordinates"])
    expected = {(path, ordinal): "train" for path, ordinal, _ in coordinates}
    if len(coordinates) != len(expected) or len(coordinates) != proof["train_packets"]:
        raise ValueError("Complete unique frozen PDB train coordinate proof required")
    descriptors = retained_roles(view, expected, root)
    retained_pdb = {}
    for descriptor in view.descriptors:
        entries = [entry for entry in descriptor["source_task_schema_counts"]
                   if entry["source_family"] == FAMILY and entry["eligible_view_disposition"] == 0]
        if entries:
            if any((row["task_name"], row["native_schema"], row["split"]) != (TASK, SCHEMA, "train")
                   for row in entries):
                raise ValueError("Frozen PDB source route or original role differs")
            retained_pdb[descriptor["source_shard"]["path"]] = sum(row["rows"] for row in entries)
    if set(retained_pdb) != set(descriptors) or sum(retained_pdb.values()) != len(coordinates):
        raise ValueError("Frozen PDB source route extent differs")
    return PDBCoordinateAxes(tuple(tuple(axis) for axis in historical["atom_label_axes"]),
                             historical["max_atoms"], coordinates, native_view_policy_sha256(view))


def verify_frozen_grayscale(view, image, root):
    rows = image["rows"]
    expected = {(r["source_shard"]["path"], r["absolute_original_row_ordinal"]): r["original_split"] for r in rows}
    if len(expected) != len(rows) or len(rows) != image["complete_retained_rows"]:
        raise ValueError("Complete unique frozen grayscale coordinate proof required")
    descriptors = retained_roles(view, expected, root)
    if any(descriptors[r["source_shard"]["path"]]["source_shard"] != r["source_shard"] for r in rows):
        raise ValueError("Frozen grayscale original source shard references changed")
