"""Target-inaccessible original NCI-AIDS chemical exposure against four prior families.

Reads only source-group sidecars from the immutable native owner. A literal
source-native molecule or scaffold hash is a grouping *signal*, never proof
of standardized chemical identity or a globally accepted evaluation split.
Does not open any native numerical/label/condition source payload.
"""
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import json
import re

from mora3.data.admet import canonical
from mora3.data.finite_union_native_rows import full9, read_exact
from mora3.research.cross_source_chemical_roles import audit_original_chemical_roles
from mora3.research.remaining_eight_registry import load_assets

_FAMILIES = {
    "NCI-conclusion": ("NCI-AIDS-source-conclusion-prior",
                       {"train": 69, "validation": 7, "test": 7}),
    "NCI-endpoint": ("NCI-AIDS-source-endpoint-prior",
                     {"train": 31, "validation": 4, "test": 4}),
    "NCI-profile": ("NCI-AIDS-paired-assay-profile-prior",
                    {"train": 2, "validation": 1, "test": 1}),
}
_ROLE_CODE = {"train": 0, "validation": 1, "test": 2}
_HEX = re.compile(r"[0-9a-f]{64}\Z")


def _digest(value):
    return sha256(canonical(value)).hexdigest()


def _strict_groups(value):
    def unique(pairs):
        output = {}
        for k, v in pairs:
            if k in output:
                raise ValueError("Duplicate original group JSON key")
            output[k] = v
        return output

    groups = json.loads(value, object_pairs_hook=unique)
    if type(groups) is not list or len(groups) != len(set(groups)) or any(
            type(x) is not str for x in groups):
        raise ValueError("Invalid source-native effective group list")
    selected = {}
    for kind in ("molecule", "scaffold"):
        values = [x for x in groups if x.startswith(kind + ":")]
        if len(values) != 1 or not re.fullmatch(kind + r":[0-9a-f]{64}", values[0]):
            raise ValueError("Missing or ambiguous source-native chemical group")
        selected[kind] = values[0]
    return selected


def summarize_group_exposure(roles):
    """Close literal molecule+scaffold links across all supplied native task roles."""
    if type(roles) is not dict or not roles:
        raise ValueError("Missing source roles")
    seen = set()
    flat = []
    for family in sorted(roles):
        if type(roles[family]) is not dict or not roles[family]:
            raise ValueError("Missing source role mapping")
        for role in sorted(roles[family]):
            if role not in _ROLE_CODE:
                raise ValueError("Unrecognized source role")
            rows = roles[family][role]
            if type(rows) not in (list, tuple) or not rows:
                raise ValueError("Empty source role")
            for rec in rows:
                if type(rec) is not dict or set(rec) != {"task_id", "molecule", "scaffold"}:
                    raise ValueError("Invalid source group record")
                task_id = rec["task_id"]
                if type(task_id) is not str or task_id in seen:
                    raise ValueError("duplicate or invalid native task identity")
                seen.add(task_id)
                for axis in ("molecule", "scaffold"):
                    value = rec[axis]
                    if type(value) is not str or not re.fullmatch(axis + r":[0-9a-f]{64}", value):
                        raise ValueError("Invalid source-native chemical group")
                flat.append((task_id, family, role, rec["molecule"], rec["scaffold"]))
    flat.sort()
    parent = list(range(len(flat)))
    sizes = [1] * len(flat)
    owners = {}
    by_role = defaultdict(lambda: {"molecule": set(), "scaffold": set()})
    role_counts = Counter()

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        i, j = find(i), find(j)
        if i == j:
            return
        if sizes[i] < sizes[j]:
            i, j = j, i
        parent[j] = i
        sizes[i] += sizes[j]

    for i, (task_id, fam, role, molecule, scaffold) in enumerate(flat):
        label = fam + "." + role
        role_counts[label] += 1
        by_role[label]["molecule"].add(molecule)
        by_role[label]["scaffold"].add(scaffold)
        for key in (molecule, scaffold):
            if key in owners:
                union(i, owners[key])
            else:
                owners[key] = i
    components = defaultdict(list)
    for i, row in enumerate(flat):
        components[find(i)].append(row)

    mixed = 0
    cross = 0
    held_held = 0
    val_test = 0
    cross_val_test = 0
    multi_family = 0
    max_mixed = 0
    digest_records = []
    for member in components.values():
        trains = [r for r in member if r[2] == "train"]
        heldouts = [r for r in member if r[2] != "train"]
        families = {r[1] for r in member}
        if len(families) > 1:
            multi_family += 1
        if trains and heldouts:
            mixed += 1
            max_mixed = max(max_mixed, len(member))
            if len(families) > 1:
                cross += 1
        if len({(r[1], r[2]) for r in heldouts}) > 1:
            held_held += 1
        if {r[2] for r in heldouts} == {"validation", "test"}:
            val_test += 1
            if len(families) > 1:
                cross_val_test += 1
        digest_records.append((
            sorted(r[0] for r in member),
            sorted({r[1] + "." + r[2] for r in member}),
        ))
    digest_records.sort()
    held = sorted(k for k in role_counts if not k.endswith(".train"))
    trains = sorted(k for k in role_counts if k.endswith(".train"))
    pairs = {}
    for h in held:
        for tr in trains:
            pairs[h + "|" + tr] = {
                axis: len(by_role[h][axis] & by_role[tr][axis])
                for axis in ("molecule", "scaffold")
            }
    return {
        "task_rows": len(flat),
        "role_rows": dict(sorted(role_counts.items())),
        "total_components": len(components),
        "multiple_source_family_components": multi_family,
        "train_heldout_components": mixed,
        "cross_source_train_heldout_components": cross,
        "heldout_heldout_shared_components": held_held,
        "validation_test_mixed_components": val_test,
        "cross_source_validation_test_mixed_components": cross_val_test,
        "largest_train_heldout_component_rows": max_mixed,
        "heldout_training_pairs": pairs,
        "all_component_membership_sha256": _digest(digest_records),
        "original_group_hash_domains_independently_equated": False,
        "global_split_admitted": False,
    }


_NCI_ORIGINAL_SHARDS = (
    "outputs/data-intake/20261101-NCI-AIDS-original-v1/"
    "admitted-source-tasks-v1/shards/"
)
_MIXED_TASKS = {
    "NCI-AIDS-source-conclusion-prior": frozenset((
        "source_NCI_assay_conclusion_prediction_prior",
        "source_NCI_assay_conclusion_generation_prior")),
    "NCI-AIDS-source-endpoint-prior": frozenset((
        "source_NCI_assay_endpoint_prediction_prior",
        "source_NCI_assay_endpoint_generation_prior")),
    "NCI-AIDS-paired-assay-profile-prior": frozenset((
        "source_NCI_paired_assay_profile_prediction_prior",
        "source_NCI_paired_assay_profile_generation_prior")),
}


def _read_mixed_nci_rows(descriptor, metadata_rows, sidecar_rows, excluded):
    """Classify exact row-level source metadata; never inspect heldout task payloads."""
    shard = descriptor["source_shard"]
    role, path = shard["split"], shard["path"]
    lookup = {src: label for label, (src, _) in _FAMILIES.items()}
    if (shard.get("source_family") is not None
            or not path.startswith(_NCI_ORIGINAL_SHARDS)
            or role not in _ROLE_CODE
            or descriptor["disposition"] != {
                "kind": "homogeneous-original-row-disposition",
                "code": _ROLE_CODE[role], "rows": shard["rows"]}
            or descriptor["primary_retained_rows"] != shard["rows"]
            or not metadata_rows or len(metadata_rows) != len(sidecar_rows)
            or len(metadata_rows) != shard["rows"]):
        raise ValueError("Mixed NCI original descriptor, retained extent or role differs")
    expected = Counter()
    for axis in descriptor["source_task_schema_counts"]:
        family = axis["source_family"]
        task = axis["task_name"]
        if (family not in lookup or task not in _MIXED_TASKS[family]
                or axis["split"] != role
                or axis["native_schema"] != "mora3.CorpusTask/v1"
                or axis["eligible_view_disposition"] != _ROLE_CODE[role]
                or axis["admitted_for_task"] is not True
                or type(axis["rows"]) is not int or axis["rows"] < 1):
            raise ValueError("Mixed NCI descriptor contains unscoped row family or task")
        expected[family, task] += axis["rows"]
    observed = Counter()
    found = defaultdict(list)
    ids = set()
    for ordinal, (row, side) in enumerate(zip(metadata_rows, sidecar_rows, strict=True)):
        family, task = row["source_family"], row["task_name"]
        task_id = row["task_id"]
        if ((path, ordinal) in excluded
                or family not in lookup or task not in _MIXED_TASKS[family]
                or row["split"] != role
                or row["native_schema"] != "mora3.CorpusTask/v1"
                or row["admitted_for_task"] is not True
                or type(task_id) is not str or not _HEX.fullmatch(task_id)
                or task_id in ids or side["task_id"] != task_id):
            raise ValueError("Mixed NCI original row/sidecar/task provenance differs")
        ids.add(task_id)
        observed[family, task] += 1
        found[lookup[family], role].append({
            "task_id": task_id,
            **_strict_groups(side["effective_groups_json"]),
        })
    if observed != expected or sum(map(len, found.values())) != shard["rows"]:
        raise ValueError("Mixed NCI original row-level family census differs")
    return dict(found)


def _read_nci_roles(root, view):
    import pyarrow as pa
    import pyarrow.parquet as pq

    lookup = {source: name for name, (source, _) in _FAMILIES.items()}
    roles = {name: {r: [] for r in spec[1]} for name, spec in _FAMILIES.items()}
    count_shards = Counter()
    original_rows = retained_rows = mixed_metadata_shards = 0
    proofs = []
    base = (root / "outputs/data-intake").resolve()
    for d in view.descriptors:
        shard = d["source_shard"]
        source = shard.get("source_family")
        is_mixed = source is None and shard["path"].startswith(_NCI_ORIGINAL_SHARDS)
        if source not in lookup and not is_mixed:
            continue
        name = lookup.get(source)
        role = shard["split"]
        if role not in _ROLE_CODE or (name is not None and role not in roles[name]):
            raise ValueError("Unexpected source-native NCI role")
        side = d["original_group_sidecar"]
        source_path = (root / shard["path"]).resolve()
        group_path = (root / side["path"]).resolve()
        if (base not in source_path.parents or base not in group_path.parents
                or "/" + role + "/" not in "/" + shard["path"]
                or tuple(d["source_full9"]) != full9(source_path)
                or not group_path.is_file()
                or side["rows"] != shard["rows"]):
            raise ValueError("NCI original source/role/stat mismatch")
        disposition = d["disposition"]
        if (disposition.get("kind") != "homogeneous-original-row-disposition"
                or disposition.get("code") != _ROLE_CODE[role]
                or disposition.get("rows") != shard["rows"]):
            raise ValueError("NCI original role/disposition mismatch")
        raw = read_exact(side, root=root)
        table = pq.read_table(pa.BufferReader(raw),
                              columns=["task_id", "effective_groups_json"],
                              use_threads=False)
        if table.num_rows != shard["rows"]:
            raise ValueError("NCI original grouping row count mismatch")
        original_rows += table.num_rows
        if is_mixed:
            # Open authenticated source bytes, decode ONLY row identity/role
            # metadata, never native_payload, conditions, targets or labels.
            metadata = pq.read_table(
                pa.BufferReader(read_exact(shard, root=root)),
                columns=["task_id", "source_family", "task_name", "split",
                         "admitted_for_task", "native_schema"],
                use_threads=False).to_pylist()
            identified = _read_mixed_nci_rows(
                d, metadata, table.to_pylist(), view._ordinal_exclusions)
            for (family, role_name), entries in identified.items():
                if role_name not in roles[family]:
                    raise ValueError("Unregistered mixed-source NCI role")
                roles[family][role_name].extend(entries)
                count_shards[family, role_name] += 1
            retained_rows += sum(len(entries) for entries in identified.values())
            mixed_metadata_shards += 1
            proofs.append(("mixed-original-row-identity", role,
                           shard["path"], shard["sha256"], side["path"],
                           side["sha256"], shard["rows"],
                           sorted((family, role_name, len(entries))
                                  for (family, role_name), entries in identified.items())))
            continue
        kept = 0
        for ordinal, entry in enumerate(table.to_pylist()):
            if (shard["path"], ordinal) in view._ordinal_exclusions:
                continue
            task_id = entry["task_id"]
            if type(task_id) is not str or _HEX.fullmatch(task_id) is None:
                raise ValueError("Invalid original NCI task identifier")
            roles[name][role].append({"task_id": task_id,
                                      **_strict_groups(entry["effective_groups_json"])})
            kept += 1
        if kept != d["primary_retained_rows"]:
            raise ValueError("NCI originally retained ordinal count mismatch")
        retained_rows += kept
        count_shards[name, role] += 1
        proofs.append((name, role, shard["path"], shard["sha256"],
                       side["path"], side["sha256"], shard["rows"], kept))
    actual = {n: {r: count_shards[n, r] for r in expected[1]} for n, expected in _FAMILIES.items()}
    expected = {n: spec[1] for n, spec in _FAMILIES.items()}
    if actual != expected:
        raise ValueError("Incomplete NCI original sidecar inventory")
    return ({f: {r: tuple(v) for r, v in rv.items()} for f, rv in roles.items()},
            actual, {
                "original_shards": len(proofs),
                "mixed_metadata_shards": mixed_metadata_shards,
                "mixed_metadata_only_no_target_decode": True,
                "original_rows": original_rows,
                "retained_rows": retained_rows,
                "registered_original_sidecar_commitment_sha256": _digest(proofs),
                "ordered_retained_original_group_sha256": _digest([
                    (f, r, [(v["task_id"], v["molecule"], v["scaffold"]) for v in roles[f][r]])
                    for f in roles for r in roles[f]
                ]),
            })


def audit_original_nci_chemical_groups(root):
    root = Path(root).resolve()
    if root != Path.cwd().resolve():
        raise ValueError("Run from exact original active source directory")
    parent = audit_original_chemical_roles(root, include_verified_roles=True)
    prior = parent.pop("_in_memory_verified_original_roles")
    if parent["common_source_snapshot_sha256"] != (
            "79d328f542d1158581875ef325e56f213b407992f95db6ee0c7352a3459669d4"):
        raise ValueError("Original 29-head common source snapshot mismatch")
    view = load_assets(root)[0]
    nci, inventory, proof = _read_nci_roles(root, view)
    if set(prior) & set(nci):
        raise ValueError("NCI source family labels collide with prior source identities")
    combined = {**prior, **nci}
    exposure = summarize_group_exposure(combined)
    if exposure["task_rows"] != (
            sum(parent["comparison"]["role_rows"].values()) + proof["retained_rows"]):
        raise ValueError("Original cross-source ordinal/role conservation mismatch")
    return {
        "schema": "Mora3.Campaign26OriginalNCIAndFourFamilyChemicalGroupExposure/v1",
        "source_snapshot_sha256": parent["common_source_snapshot_sha256"],
        "original_four_family_group_receipt_sha256": _digest(parent),
        "original_four_family_sidecar_commitment_sha256": parent["source_role_file_commitment"],
        "nci_original_shards": inventory,
        "nci_group_readback": proof,
        "exposure": exposure,
        "source_numeric_shards_read": False,
        "heldout_targets_decoded": False,
        "source_group_hash_domains_independently_equated": False,
        "global_split_admitted": False,
        "optimizer_updates": 0,
        "actual_campaign_runs": 0,
        "training_admission": False,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python -m mora3.research.nci_original_chemical_group_exposure ACTIVE_ROOT NEW_OUTPUT")
    target = Path(sys.argv[2])
    if target.exists():
        raise SystemExit("Refusing to overwrite previous original source result")
    result = audit_original_nci_chemical_groups(sys.argv[1])
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({
        "sha256": sha256(target.read_bytes()).hexdigest(),
        "nci": result["nci_group_readback"],
        "exposure": {k: v for k, v in result["exposure"].items()
                     if k not in ("heldout_training_pairs", "role_rows")},
    }, indent=2, sort_keys=True))
