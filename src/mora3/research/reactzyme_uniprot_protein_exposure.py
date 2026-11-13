"""Original ReactZyme versus UniProt train-only protein group exposure.

Reads authenticated original EFFECTIVE-GROUP SIDECARS and retained ordinal masks.
Does not read source-native task targets, heldout shards, model or optimizer.
Only equal source accessions, original sequence digests, or qualified MMseqs
guard tokens connect biological observations; reaction/molecule/scaffold
co-occurrence never asserts equal proteins or paired biological effects.
"""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re

from mora3.data.finite_union_native_rows import (
    decode_disposition, full9, read_exact,
)
from mora3.research.remaining_eight_registry import load_assets

_FAMILIES = ("ReactZyme", "UniProt")
_KINDS = ("accession", "sequence", "qualified_homology_guard")
_SEQ = re.compile(r"sequence:[0-9a-f]{64}\Z")
_GUARD = re.compile(
    r"detected_protein_guard:[0-9a-f]{16}:qualified_MMseqs_linclust30cov80:[0-9a-f]{64}\Z")
_ACCESSION = re.compile(r"accession:[A-Za-z0-9_.-]+\Z")
_TASK_ID = re.compile(r"[0-9a-f]{64}\Z")
_SCOPE = "identity/leakagegroupsonly;notpredictorfeaturesorsamecell/effectproof"
_SKIP = frozenset(("accession:-", "accession:unknown", "accession:UNKNOWN",
                  "accession:NA", "accession:N/A", "accession:none"))


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def _kind(token):
    if not isinstance(token, str):
        raise ValueError("Non-string original protein grouping token")
    if token in _SKIP:
        return None
    if token.startswith("accession:"):
        if _ACCESSION.fullmatch(token) is None:
            raise ValueError("Malformed original accession")
        return "accession"
    if token.startswith("sequence:"):
        if _SEQ.fullmatch(token) is None:
            raise ValueError("Malformed original sequence digest")
        return "sequence"
    if token.startswith("detected_protein_guard:"):
        if _GUARD.fullmatch(token) is None:
            raise ValueError("Unqualified or malformed homology grouping")
        return "qualified_homology_guard"
    return None


def protein_exposure(reactzyme, uniprot):
    """Connected-component exposure over three *source-reported* token classes.

    Input is immutable (task_id, effective_groups) pairs. Nonbiological groups
    are deliberately excluded from this graph; no heldout independence is inferred.
    """
    populations = (tuple(reactzyme), tuple(uniprot))
    owner = {}
    roots = []
    seen = set()
    memberships = [[], []]
    typed = [{k: defaultdict(set) for k in _KINDS} for _ in _FAMILIES]
    accession_sequences = [defaultdict(set), defaultdict(set)]

    def find(i):
        while roots[i] != i:
            roots[i] = roots[roots[i]]
            i = roots[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            roots[a] = b

    for group, rows in enumerate(populations):
        for task_id, groups in rows:
            if not isinstance(task_id, str) or task_id in seen:
                raise ValueError("Duplicate or invalid original task identifier")
            if not isinstance(groups, (tuple, list)) or len(groups) != len(set(groups)):
                raise ValueError("Duplicate/invalid source group list")
            seen.add(task_id)
            row_accessions = [g for g in groups if _kind(g) == "accession"]
            row_sequences = {g for g in groups if _kind(g) == "sequence"}
            for accession in row_accessions:
                accession_sequences[group][accession].update(row_sequences)
            i = len(roots)
            roots.append(i)
            memberships[group].append((i, task_id))
            for token in groups:
                kind = _kind(token)
                if kind is not None:
                    typed[group][kind][token].add(task_id)
                    if token in owner:
                        union(i, owner[token])
                    else:
                        owner[token] = i

    overlap = {}
    for kind in _KINDS:
        common = set(typed[0][kind]) & set(typed[1][kind])
        overlap[kind] = len(common)

    same_accessions = set(typed[0]["accession"]) & set(typed[1]["accession"])
    both_have_sequence = [a for a in same_accessions
        if accession_sequences[0][a] and accession_sequences[1][a]]
    same_sequence = sum(bool(accession_sequences[0][a] & accession_sequences[1][a])
                        for a in both_have_sequence)

    components = defaultdict(lambda: [[], []])
    for source, rows in enumerate(memberships):
        for index, task_id in rows:
            components[find(index)][source].append(task_id)

    cross = [c for c in components.values() if c[0] and c[1]]
    joined = []
    for left, right in cross:
        joined.append((tuple(sorted(left)), tuple(sorted(right))))
    joined.sort()
    return {
        "task_rows": dict(zip(_FAMILIES, map(len, populations))),
        "shared_group_counts": overlap,
        "shared_accessions_with_equal_sequence_groups": same_sequence,
        "shared_accessions_without_equal_sequence_groups": len(both_have_sequence)-same_sequence,
        "shared_accessions_with_incomplete_sequence_grouping": len(same_accessions)-len(both_have_sequence),
        "cross_source_components": len(cross),
        "largest_cross_component": max((len(a)+len(b) for a,b in cross),default=0),
        "cross_source_member_rows": {
            "ReactZyme": sum(len(a) for a,b in cross),
            "UniProt": sum(len(b) for a,b in cross),
        },
        "source_local_and_cross_components": len(components),
        "cross_membership_sha256": sha256(_canonical(joined)).hexdigest(),
        "scientific_scope": "source-reported accession/sequence or qualified homology exposure; not independent experimental identity or response pairing",
        "global_evaluation_split_admitted": False,
    }


def _original_retained(root, descriptors):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = []
    pinned = []
    observed = 0
    for d in descriptors:
        source = d["source_shard"]
        side = d["original_group_sidecar"]
        path = root / source["path"]
        if (source["split"] != "train" or "multisource/" + source["source_family"] + "/train/" not in source["path"]
                or "effective-groups/" not in side["path"]
                or len(d["source_full9"]) != 9
                or tuple(d["source_full9"]) != full9(path)
                or side["rows"] != source["rows"]):
            raise ValueError("Original train source/stat/sidecar role mismatch")
        mask = decode_disposition(d, root=root)
        raw = read_exact(side, root=root)
        table = pq.read_table(pa.BufferReader(raw),
                columns=["task_id", "effective_groups_json",
                         "source_group_ids_json", "source_file", "evidence_scope"],
                use_threads=False)
        if table.num_rows != source["rows"]:
            raise ValueError("Original group sidecar row conservation failed")
        observed += table.num_rows
        for ordinal, record in enumerate(table.to_pylist()):
            if record["source_file"] != source["path"] or record["evidence_scope"] != _SCOPE:
                raise ValueError("Original source/evidence role was altered")
            if mask[ordinal] != 0:
                continue
            if _TASK_ID.fullmatch(record["task_id"]) is None:
                raise ValueError("Unsealed original task identifier")
            source_groups = json.loads(record["source_group_ids_json"])
            groups = json.loads(record["effective_groups_json"])
            if (type(groups) is not list or type(source_groups) is not list
                    or len(groups) != len(set(groups))
                    or not set(source_groups).issubset(set(groups))
                    or any(not g.startswith("detected_protein_guard:")
                           for g in set(groups) - set(source_groups))):
                raise ValueError("Original group and homology overlay scope mismatch")
            rows.append((record["task_id"], tuple(groups)))
        pinned.append((source["path"], side["sha256"], d["primary_retained_rows"]))
    if len(rows) != sum(x["primary_retained_rows"] for x in descriptors):
        raise ValueError("Eligible original ordinal conservation failed")
    return tuple(rows), observed, sha256(_canonical(pinned)).hexdigest()


def audit_original_reactzyme_uniprot(root):
    root = Path(root).resolve()
    if root != Path.cwd().resolve():
        raise ValueError("Original active source root needed for source-bound pins")
    from mora3.research.primary_campaign_source_rebind import load_inactive_common_source_rebind
    context = load_inactive_common_source_rebind(root)
    if context.common_snapshot.sha256 != "79d328f542d1158581875ef325e56f213b407992f95db6ee0c7352a3459669d4":
        raise ValueError("Staged 29-head original-source snapshot changed")
    view = load_assets(root)[0]
    lists = {}
    for name in _FAMILIES:
        descriptors = [
            d for d in view.descriptors
            if d["source_shard"].get("source_family") == name
            and d["source_shard"].get("split") == "train"
        ]
        if len(descriptors) != {"ReactZyme": 161, "UniProt": 407}[name]:
            raise ValueError("Incomplete original source inventory")
        lists[name] = (descriptors, *_original_retained(root, descriptors))
    return {
        "schema": "Mora3.Campaign26OriginalReactZymeUniProtProteinExposure/v1",
        "source_snapshot_sha256": context.common_snapshot.sha256,
        "original_shards": {n:len(lists[n][0]) for n in _FAMILIES},
        "original_rows": {n:lists[n][2] for n in _FAMILIES},
        "retained_rows": {n:len(lists[n][1]) for n in _FAMILIES},
        "original_sidecar_pin_sets": {n:lists[n][3] for n in _FAMILIES},
        "exposure": protein_exposure(lists["ReactZyme"][1], lists["UniProt"][1]),
        "original_native_source_targets_opened": False,
        "heldout_targets_opened": False,
        "training_admission": False,
        "optimizer_updates": 0,
        "actual_proxy_trials": 0,
        "actual_target_runs": 0,
    }
