"""Authenticated original SciPlex study-level source-group exposure, NOT biological pairing.

A shared study/GSM source grouping is a split/contamination guard, never a
same-cell, same-response, donor, perturbation effect or experimental pairing.
Train group sidecars and original source byte hashes only; no source target decode.
"""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re

from mora3.data.admet import canonical
from mora3.data.finite_union_native_rows import decode_disposition, full9, read_exact
from mora3.research.remaining_eight_registry import load_assets
from mora3.research.primary_campaign_source_rebind import load_inactive_common_source_rebind

_FAMILIES = ("sciplex_native_admitted", "sciplex_native_expanded")
_SCOPE = "identity/leakagegroupsonly;notpredictorfeaturesorsamecell/effectproof"
_EXPECTED_SNAPSHOT = "79d328f542d1158581875ef325e56f213b407992f95db6ee0c7352a3459669d4"
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_STUDY = re.compile(r"study:[A-Za-z0-9_.:-]+\Z")
_INVALID = {"study:", "study:-", "study:unknown", "study:None", "study:null", "study:NA"}


def _study_key(token):
    if type(token) is not str or not token or token.strip() != token:
        raise ValueError("Invalid original grouping token")
    if token.startswith("study:"):
        if token in _INVALID or not _STUDY.fullmatch(token):
            raise ValueError("Ambiguous or malformed study group")
        return token
    return None


def _strict_groups(raw):
    try:
        v = json.loads(raw)
    except (TypeError, ValueError) as e:
        raise ValueError("Invalid original group JSON") from e
    if type(v) is not list or not v or any(type(t) is not str for t in v):
        raise ValueError("Incomplete original group JSON")
    if len(v) != len(set(v)):
        raise ValueError("Duplicated original group token")
    for token in v:
        _study_key(token)
    return tuple(v)


def study_group_exposure(admitted, expanded):
    """Study-token connected components; other tokens NEVER assert same cell/effect."""
    populations = (tuple(admitted), tuple(expanded))
    if any(not p for p in populations):
        raise ValueError("Both SciPlex source populations are required")
    roots = []
    size = []
    owner = {}
    members = []
    seen_tasks = set()
    token_populations = defaultdict(int)

    def find(i):
        while roots[i] != i:
            roots[i] = roots[roots[i]]
            i = roots[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            if size[a] < size[b]:
                a, b = b, a
            roots[b] = a
            size[a] += size[b]

    for source, population in enumerate(populations):
        for record in population:
            if type(record) is not tuple or len(record) != 2:
                raise ValueError("Invalid original SciPlex group row")
            task_id, groups = record
            if type(task_id) is not str or not _HASH.fullmatch(task_id) or task_id in seen_tasks:
                raise ValueError("Duplicate or invalid original task ID")
            if type(groups) is not tuple or not groups or len(groups) != len(set(groups)):
                raise ValueError("Invalid or duplicate group list")
            seen_tasks.add(task_id)
            i = len(roots)
            roots.append(i)
            size.append(1)
            members.append((source, task_id))
            for token in groups:
                study = _study_key(token)
                if study is None:
                    continue
                token_populations[study] |= 1 << source
                if study in owner:
                    union(i, owner[study])
                else:
                    owner[study] = i

    components = defaultdict(list)
    for i, member in enumerate(members):
        components[find(i)].append(member)
    mixed = [c for c in components.values() if {source for source, _ in c} == {0, 1}]
    mixed_members = Counter(_FAMILIES[src] for c in mixed for src, _ in c)
    component_commitments = sorted(
        sha256(canonical(sorted((_FAMILIES[src], task) for src, task in c))).hexdigest()
        for c in components.values()
    )
    return {
        "schema": "Mora3.OriginalSciPlexStudyExposure/v1",
        "original_retained_task_rows": dict(zip(_FAMILIES, (len(p) for p in populations))),
        "source_reported_distinct_study_tokens": len(token_populations),
        "shared_study_tokens": sum(mask == 3 for mask in token_populations.values()),
        "component_count": len(components),
        "cross_population_study_components": len(mixed),
        "largest_cross_component": max(map(len, mixed), default=0),
        "cross_population_task_rows": {k: mixed_members[k] for k in _FAMILIES},
        "component_membership_sha256": sha256(canonical(component_commitments)).hexdigest(),
        "same_cell_pairings_proven": False,
        "independent_replication_proven": False,
        "heldout_split_independence_proven": False,
        "training_admission": False,
    }


def _sha_file(path):
    digest = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_original_sciplex_study_groups(root):
    """Read exact original train sidecars and verify source *bytes*, never values."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(root).resolve()
    if root != Path.cwd().resolve():
        raise ValueError("Original data root required; no data copy or fixture substitution")
    context = load_inactive_common_source_rebind(root)
    if context.common_snapshot.sha256 != _EXPECTED_SNAPSHOT:
        raise ValueError("Original source-family snapshot changed")
    view = load_assets(root)[0]
    data = {}
    source_counts = {}
    retained_counts = {}
    pins = {}
    for family, expected in zip(_FAMILIES, (4, 158), strict=True):
        descriptors = [d for d in view.descriptors if
                       "/shards/" + family + "/train/" in d["source_shard"].get("path", "")]
        if len(descriptors) != expected:
            raise ValueError("Incomplete original SciPlex source-role inventory")
        retained = []
        source_count = 0
        source_pins = []
        for d in descriptors:
            shard = d["source_shard"]
            side = d["original_group_sidecar"]
            if ("/shards/" + family + "/train/" not in shard["path"]
                    or "/effective-groups/" not in side["path"]
                    or shard["rows"] != side["rows"]
                    or len(d["source_full9"]) != 9):
                raise ValueError("Original sidecar/source role mismatch")
            path = root / shard["path"]
            if tuple(d["source_full9"]) != full9(path):
                raise ValueError("Source stat identity drifted")
            if _sha_file(path) != shard["sha256"]:
                raise ValueError("Original source byte SHA256 drifted")
            mask = decode_disposition(d, root=root)
            table = pq.read_table(
                pa.BufferReader(read_exact(side, root=root)),
                columns=["task_id", "source_group_ids_json", "effective_groups_json",
                         "source_file", "evidence_scope"],
                use_threads=False,
            )
            if len(mask) != shard["rows"] or table.num_rows != shard["rows"]:
                raise ValueError("Original group/mask cardinality changed")
            source_count += shard["rows"]
            accepted_this_shard = 0
            for ordinal, r in enumerate(table.to_pylist()):
                if (set(r) != {"task_id", "source_group_ids_json", "effective_groups_json",
                              "source_file", "evidence_scope"}
                        or r["source_file"] != shard["path"] or r["evidence_scope"] != _SCOPE
                        or type(r["task_id"]) is not str or not _HASH.fullmatch(r["task_id"])):
                    raise ValueError("Original task/provenance/scope was not retained")
                native = _strict_groups(r["source_group_ids_json"])
                effective = _strict_groups(r["effective_groups_json"])
                if not set(native).issubset(effective):
                    raise ValueError("Original native grouping changed in effective overlay")
                if mask[ordinal] == 0:
                    retained.append((r["task_id"], effective))
                    accepted_this_shard += 1
            if accepted_this_shard != d["primary_retained_rows"]:
                raise ValueError("Original retained view mask/metadata changed")
            source_pins.append((shard["path"], shard["sha256"], side["sha256"],
                                shard["rows"], accepted_this_shard))
        data[family] = tuple(retained)
        source_counts[family] = source_count
        retained_counts[family] = len(retained)
        pins[family] = sha256(canonical(source_pins)).hexdigest()
    result = study_group_exposure(*(data[name] for name in _FAMILIES))
    return {
        "schema": "Mora3.Campaign26OriginalSciPlexGroupAudit/v1",
        "source_snapshot_sha256": context.common_snapshot.sha256,
        "source_shards": dict(zip(_FAMILIES, (4, 158))),
        "source_rows": source_counts,
        "retained_rows": retained_counts,
        "original_source_and_group_sha256_verified": True,
        "sidecar_and_source_role_pin_sha256": pins,
        "study_exposure": result,
        "source_native_values_opened": False,
        "heldout_sources_opened": False,
        "same_cell_or_perturbation_response_join_claimed": False,
        "globally_independent_biological_split_attested": False,
        "training_admission": False,
        "optimizer_updates": 0,
        "actual_proxy_trials": 0,
        "actual_target_runs": 0,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python -m mora3.research.sciplex_original_group_exposure ORIGINAL_ROOT NEW_OUTPUT.json")
    result = audit_original_sciplex_study_groups(sys.argv[1])
    path = Path(sys.argv[2])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(result, f, sort_keys=True, indent=2)
        f.write("\n")
    print(json.dumps({"source_shards": result["source_shards"],
                      "retained_rows": result["retained_rows"],
                      "cross_population_study_components":
                          result["study_exposure"]["cross_population_study_components"]},
                     sort_keys=True))
