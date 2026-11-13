"""Complete original BioGRID chemical train-source replay under frozen common owner.

This is an authenticated source-only finite replay. The two-window candidate
comparison does not extend either campaign allocation or training permission.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json
import os

from mora3.data.admet import canonical
from mora3.data.biogrid_chemical_current import (
    FAMILY, REVISION, TASK, SCHEMA, KNOWN_FIELDS, TARGET_FIELDS,
)
from mora3.data.finite_union_native_rows import decode_disposition, full9
from mora3.research.native_annotation_registry import load_original_annotation_assets, VIEW
from mora3.research.native_annotation_campaign_routes import bind_biogrid_chemical
from mora3.research.primary_campaign_source_rebind import load_inactive_common_source_rebind
from mora3.research.primary_source_factory_preflight import document
from mora3.research.primary_native_routes import finite_view_policy_sha256
from mora3.research.mup_campaign import executable_adapter_sha256


KEY = "biogrid_chemical_annotation"
CANDIDATE = "1091e3f27d57cb70203ab9fce1136836d0cd96eef955b6682a3e6264abadc733"
COMMON = "79d328f542d1158581875ef325e56f213b407992f95db6ee0c7352a3459669d4"
EVIDENCE = "source_curated_publication_supported_annotation"


def h(raw):
    return sha256(raw).hexdigest()


def verify_biogrid_native_target(doc, packet, prepared):
    """Refuse target-to-condition leakage or changed literal joint action/type labels."""
    if (type(doc) is not dict or set(doc["conditions"]) != set(KNOWN_FIELDS)
            or set(doc["targets"]) != set(TARGET_FIELDS)
            or set(doc["conditions"]) & set(TARGET_FIELDS)
            or packet.condition.inputs() != doc["conditions"]
            or packet.target != tuple(doc["targets"][field] for field in TARGET_FIELDS)
            or prepared.observed_positions != 1
            or prepared.unavailable_reason is not None
            or not callable(prepared.evaluate)):
        raise ValueError("BioGRID literal joint target/input isolation or native target count differs")
    return 1


def _original(window, source, binding):
    batch = window.batch
    if (window.original_source_rows != 1 or batch.original_split != "train"
            or batch.native_rows.num_rows != 1):
        raise ValueError("One original BioGRID native train occurrence required")
    source._view.require_native_rows(batch, split="train")
    row, = batch.native_rows.to_pylist()
    raw = row["native_payload"]
    if (type(raw) is not bytes or h(raw) != row["native_payload_sha256"]
            or (row["source_family"], row["source_revision"], row["task_name"],
                row["native_schema"], row["split"], row["evidence_class"])
            != (FAMILY, REVISION, TASK, SCHEMA, "train", EVIDENCE)):
        raise ValueError("Original BioGRID raw source hash, release, role or evidence differs")
    doc = json.loads(raw)
    if (canonical(doc) != raw or doc.get("schema") != SCHEMA or doc.get("task") != TASK
            or doc["audit"]["evidence"] != "original BIOGRID curated"):
        raise ValueError("Exact canonical original BioGRID source audit required")
    packet, = source.packets(batch, split="train")
    if packet.source_coordinate != (
            batch.source_shard["path"], batch.source_row_ordinals[0], h(raw)):
        raise ValueError("Original BioGRID source coordinate differs")
    count = verify_biogrid_native_target(doc, packet, binding.prepare(window))
    return (batch.source_shard["path"], batch.source_row_ordinals[0],
            h(canonical([row["task_id"], row["native_payload_sha256"]])),
            h(canonical(packet.target)), count)


def frozen_biogrid_prefix(binding, source):
    it = binding.route.iter_from(binding.route.initial_cursor)
    try:
        first, second = next(it), next(it)
    finally:
        it.close()
    restart = binding.route.iter_from(first.cursor_after)
    try:
        resumed = next(restart)
    finally:
        restart.close()
    if (second.cursor_after != resumed.cursor_after
            or not second.batch.native_rows.equals(resumed.batch.native_rows, check_metadata=True)):
        raise ValueError("BioGRID original cursor recovery changed")
    return (first, second), [_original(w, source, binding) for w in (first, second)]


def _direct_parquet(root, view, paths, axes):
    """Independently scan original files without invoking campaign route or adapter."""
    import pyarrow.parquet as pq

    signatures = []
    target_axis = set(axes.target_pairs)
    classes = {}
    for path in paths:
        descriptor = json.loads(view._sealed_by_path[path])
        original = root / path
        if full9(original) != tuple(descriptor["source_full9"]):
            raise ValueError("Original BioGRID source shard changed before independent read")
        dispositions = decode_disposition(descriptor, root=root)
        ordinal = 0
        for batch in pq.ParquetFile(original).iter_batches(batch_size=64, use_threads=False):
            for row in batch.to_pylist():
                if (dispositions[ordinal] == 0
                        and (row["source_family"], row["source_revision"], row["task_name"],
                             row["native_schema"], row["split"])
                        == (FAMILY, REVISION, TASK, SCHEMA, "train")):
                    raw = row["native_payload"]
                    if (type(raw) is not bytes or h(raw) != row["native_payload_sha256"]
                            or row["evidence_class"] != EVIDENCE):
                        raise ValueError("Original BioGRID direct source evidence changed")
                    doc = json.loads(raw)
                    if (canonical(doc) != raw or doc.get("schema") != SCHEMA or doc.get("task") != TASK
                            or doc["audit"]["evidence"] != "original BIOGRID curated"
                            or set(doc["conditions"]) != set(KNOWN_FIELDS)
                            or set(doc["targets"]) != set(TARGET_FIELDS)
                            or set(doc["conditions"]) & set(TARGET_FIELDS)):
                        raise ValueError("BioGRID direct raw target/condition provenance differs")
                    original_row = doc["audit"]["source_row"]
                    expected = {
                        "chemical_identifier": original_row["Chemical Source"].upper() + ":" + original_row["Chemical Source ID"],
                        "chemical_name": original_row["Chemical Name"],
                        "gene_identifier": original_row["Entrez Gene ID"],
                        "organism_taxon": original_row["Organism ID"],
                    }
                    target = (original_row["Action"], original_row["Interaction Type"])
                    if (expected != doc["conditions"] or target != tuple(
                            doc["targets"][field] for field in TARGET_FIELDS) or target not in target_axis):
                        raise ValueError("Original BioGRID chemical/gene source audit differs")
                    classes[target] = classes.get(target, 0) + 1
                    signatures.append((path, ordinal,
                                       h(canonical([row["task_id"], row["native_payload_sha256"]]))))
                ordinal += 1
        if ordinal != len(dispositions) or full9(original) != tuple(descriptor["source_full9"]):
            raise ValueError("Original BioGRID source changed during direct read")
    return signatures, classes


def audit_biogrid_complete(root):
    root = Path(root).resolve()
    context = load_inactive_common_source_rebind(root)
    frozen = document(root, context.evidence)
    found = [r for r in frozen["route_bindings"] if r["task_key"] == KEY]
    if (len(found) != 1 or len(frozen["route_bindings"]) != 62
            or frozen["training_authorized"] is not False
            or frozen["global_split_accepted"] is not False):
        raise ValueError("Inactive exact 62-route frozen source required")
    record, = found
    if (frozen["inactive_candidate_snapshot"]["sha256"] != CANDIDATE
            or frozen["common_source_snapshot"]["sha256"] != COMMON
            or record["common_view_policy_sha256"] != VIEW
            or record["candidate"]["max_windows"] != 2
            or record["common"]["max_windows"] != 2):
        raise ValueError("Frozen BioGRID source family or budget differs")
    view, sources, availability, _, _ = load_original_annotation_assets(root)
    source = sources["biogrid_chemical"]
    if (finite_view_policy_sha256(view) != VIEW
            or availability[KEY]["train_rows"] != 2706
            or availability[KEY]["validation_rows"] != 0
            or availability[KEY]["test_rows"] != 0):
        raise ValueError("Exact original BioGRID native train-only view differs")
    code_root = Path(__file__).resolve().parents[3]
    bound = {}
    for label, snap in (("candidate", CANDIDATE), ("common", COMMON)):
        b = bind_biogrid_chemical(source, view, snapshot_sha256=snap, batch_size=1)
        observed = {
            "task_key": KEY, "max_windows": 2,
            "initial_cursor_sha256": h(b.route.initial_cursor),
            "adapter_sha256": executable_adapter_sha256(b.prepare, root=code_root),
        }
        if observed != record[label] or b.route.source_snapshot_sha256 != snap:
            raise ValueError("Exact frozen BioGRID executable/cursor drift: " + label)
        bound[label] = b

    candidate_windows, candidate_prefix = frozen_biogrid_prefix(bound["candidate"], source)
    common_windows, common_prefix = frozen_biogrid_prefix(bound["common"], source)
    if (candidate_prefix != common_prefix
            or any(not a.batch.native_rows.equals(b.batch.native_rows, check_metadata=True)
                   for a, b in zip(candidate_windows, common_windows, strict=True))):
        raise ValueError("Original BioGRID frozen candidate/common row prefix differs")

    common = bound["common"]
    signatures = []
    target_counts = {}
    cursor_recoveries = 0
    pending = None
    it = common.route.iter_from(common.route.initial_cursor)
    try:
        for window in it:
            item = _original(window, source, common)
            signatures.append(item[:3])
            target_counts[item[3]] = target_counts.get(item[3], 0) + 1
            n = len(signatures)
            if pending is not None and n in (3, 1354, 2706):
                old_cursor, old_row = pending
                if (window.cursor_after != old_cursor
                        or not window.batch.native_rows.equals(old_row, check_metadata=True)):
                    raise ValueError("Native BioGRID same-run cursor resumed at different original row")
                cursor_recoveries += 1
                pending = None
            if n in (2, 1353, 2705):
                restart = common.route.iter_from(window.cursor_after)
                try:
                    successor = next(restart)
                except StopIteration as error:
                    raise ValueError("BioGRID native route prematurely exhausted") from error
                finally:
                    restart.close()
                pending = (successor.cursor_after, successor.batch.native_rows)
    finally:
        it.close()
    if (len(signatures) != 2706 or len(set(signatures)) != 2706
            or cursor_recoveries != 3):
        raise ValueError("BioGRID complete original native route count/cursor differs")
    direct, direct_classes = _direct_parquet(root, view, tuple(common.route._paths), source.axes)
    if direct != signatures or candidate_prefix[:2] != common_prefix:
        raise ValueError("BioGRID original direct Parquet and native source signatures differ")
    if len(direct_classes) != len(source.axes.target_pairs) or sum(direct_classes.values()) != 2706:
        raise ValueError("BioGRID original full literal joint target class axis incomplete")
    return {
        "schema": "Mora3.Campaign26BioGRIDCompleteCommonOriginal/v1",
        "source_release": REVISION, "source_evidence_class": EVIDENCE,
        "source_only": True, "frozen_62_route_metadata_only": True,
        "source_snapshot_sha256": COMMON, "source_view_policy_sha256": VIEW,
        "frozen_candidate_common": record, "candidate_common_two_window_equal": True,
        "common_original_train_rows_exhausted": len(signatures),
        "original_joint_target_positions": sum(target_counts.values()),
        "original_joint_target_classes": len(direct_classes),
        "original_signature_chain_sha256": h(canonical(signatures)),
        "independent_original_parquet_equal": True,
        "original_cursor_recoveries": cursor_recoveries,
        "literal_joint_target_excluded_from_conditions": True,
        "candidate_common_full_replay": False,
        "global_biological_split_admitted": False, "complete_62_source_family_admitted": False,
        "training_admission": False, "optimizer_updates": 0, "counted_proxy_trials": 0,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python -m mora3.research.campaign26_biogrid_complete_common ACTIVE_ROOT NEW_JSON")
    if os.uname().nodename != "iska":
        raise ValueError("Only authorized Iska machine allowed")
    dst = Path(sys.argv[2])
    if dst.exists():
        raise FileExistsError("Immutable source evidence must not be replaced")
    result = audit_biogrid_complete(sys.argv[1])
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("x") as output:
        json.dump(result, output, sort_keys=True, indent=2)
        output.write("\n")
    print(json.dumps({"audit_sha256": h(dst.read_bytes()),
                      "native_original_rows": result["common_original_train_rows_exhausted"],
                      "original_joint_target_classes": result["original_joint_target_classes"]},
                     sort_keys=True))
