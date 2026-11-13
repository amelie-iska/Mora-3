"""Exact-original three-family protein identity exposure; no model or heldout access.

The input is the native effective-group evidence, not a constructed dataset split.
Protein accessions, SHA sequence identities and qualified original homology guard
groups alone define identity edges. Publication/study and chemical co-occurrences
are NEVER treated as evidence that two protein observations are the same.
"""
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import json
import re

from mora3.data.admet import canonical
from mora3.data.finite_union_native_rows import decode_disposition, full9, read_exact
from mora3.research.cross_source_uniprot_kinetics import (
    _KINETIC_READBACK, _KINETIC_READBACK_SHA, _POLICY, _strict_json,
)
from mora3.research.reactzyme_uniprot_protein_exposure import (
    _original_retained, _kind,
)
from mora3.research.remaining_eight_registry import load_assets
from mora3.research.primary_native_routes import finite_view_policy_sha256

_FAMILIES = ("ReactZyme", "UniProt", "UniProt-kinetics")
_TYPES = ("accession", "sequence", "qualified_homology_guard")
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_PUBMED = re.compile(r"pubmed:[0-9]+\Z")
_SOURCE_SNAPSHOT = "79d328f542d1158581875ef325e56f213b407992f95db6ee0c7352a3459669d4"
_SCOPE = "identity/leakagegroupsonly;notpredictorfeaturesorsamecell/effectproof"


def combined_protein_closure(reactzyme, uniprot, kinetics):
    """Transitive ORIGINAL identity-group closure without adding cross-task labels.

    A study shared across sources remains a reported publication intersection,
    NOT an identity edge. All inputs are kept as separately counted task records.
    """
    populations = (reactzyme, uniprot, kinetics)
    if any(type(pop) is not tuple or not pop for pop in populations):
        raise ValueError("Three nonempty fixed original train populations required")
    roots = []
    sizes = []
    ownership = {}
    members = []
    token_sources = defaultdict(int)
    study_sources = defaultdict(int)
    seen = set()
    def find(i):
        while roots[i] != i:
            roots[i] = roots[roots[i]]
            i = roots[i]
        return i
    def union(i, j):
        i, j = find(i), find(j)
        if i == j:
            return
        if sizes[i] < sizes[j]:
            i, j = j, i
        roots[j] = i
        sizes[i] += sizes[j]
    for source, pop in enumerate(populations):
        for record in pop:
            if (type(record) is not tuple or len(record) != 2
                    or type(record[0]) is not str or not _HEX.fullmatch(record[0])
                    or record[0] in seen or type(record[1]) is not tuple
                    or not record[1] or len(record[1]) != len(set(record[1]))):
                raise ValueError("Invalid, repeated or unqualified original task and group binding")
            task_id, groups = record
            seen.add(task_id)
            i = len(roots)
            roots.append(i)
            sizes.append(1)
            members.append((source, task_id))
            for token in groups:
                kind = _kind(token)  # original protein token validator, excludes missing ID
                if kind in _TYPES:
                    token_sources[token] |= 1 << source
                    if token in ownership:
                        union(i, ownership[token])
                    else:
                        ownership[token] = i
                elif token.startswith("pubmed:"):
                    if _PUBMED.fullmatch(token) is None:
                        raise ValueError("Unqualified original publication/study token")
                    study_sources[token] |= 1 << source
    components = defaultdict(list)
    for i, member in enumerate(members):
        components[find(i)].append(member)
    direct_rk = {find(ownership[token]) for token, bitmap in token_sources.items()
                 if bitmap & 5 == 5}
    crossed = [c for c in components.values() if len({s for s, _ in c}) > 1]
    triples = [c for c in crossed if len({s for s, _ in c}) == 3]
    exposed = Counter(_FAMILIES[s] for comp in crossed for s, _ in comp)
    # Direct ReactZyme/kinetics links must not be inferred from a UniProt bridge.
    triple_indirect = sum(root not in direct_rk for root, comp in components.items()
                          if len({s for s, _ in comp}) == 3)
    paired = Counter(tuple(_FAMILIES[s] for s in sorted({src for src, _ in c}))
                     for c in crossed)
    direct_counts = {
        kind: sum(1 for token, bitmap in token_sources.items()
                  if bitmap & 5 == 5 and _kind(token) == kind)
        for kind in _TYPES
    }
    member_commitments = sorted(sha256(canonical(sorted((_FAMILIES[s], task) for s,task in comp))).hexdigest()
                                for comp in components.values())
    return dict(schema="Mora3.ThreeOriginalProteinFamilyIdentityClosure/v1",
                rows=dict(zip(_FAMILIES, (len(p) for p in populations))),
                components=len(components), cross_family_components=len(crossed),
                triple_family_components=len(triples),
                largest_triple_component=max((len(c) for c in triples), default=0),
                indirect_reactzyme_kinetics_components=triple_indirect,
                direct_reactzyme_kinetics_group_counts=direct_counts,
                shared_study_tokens_all_three=sum(bitmap == 7 for bitmap in study_sources.values()),
                cross_family_component_rows={name: exposed[name] for name in _FAMILIES},
                source_combinations={" + ".join(pair): count for pair, count in sorted(paired.items())},
                component_membership_sha256=sha256(canonical(member_commitments)).hexdigest(),
                cross_component_membership_sha256=sha256(canonical(sorted(
                    sha256(canonical(sorted((_FAMILIES[s], task) for s,task in comp))).hexdigest()
                    for comp in crossed))).hexdigest(),
                source_identity_only=True, study_not_identity=True,
                no_original_target_or_heldout_values=True, global_split_admitted=False,
                training_admission=False)


def _original_kinetics(root, descriptors):
    """Revalidate each original kinetics group against prior native train census."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    prior_path = Path(__file__).resolve().parents[3] / _KINETIC_READBACK
    prior_raw = prior_path.read_bytes()
    if sha256(prior_raw).hexdigest() != _KINETIC_READBACK_SHA:
        raise ValueError("Independent complete kinetics source census changed")
    prior = _strict_json(prior_raw)
    if (prior["native_train_rows"] != 618 or prior["original_source_rows"] != 622
            or prior["original_view_policy_sha256"] != _POLICY
            or prior["both_original_train_iterators_exhausted"] is not True
            or prior["training_admission"] is not False):
        raise ValueError("Original kinetic train evidence was not qualified")
    expected = {(r["source_shard_sha256"], r["source_ordinal"]): r for r in prior["rows"]}
    if len(expected) != 618 or len(descriptors) != 3:
        raise ValueError("Incomplete independent kinetic retained population")
    rows = []
    pins = []
    original_count = 0
    for d in descriptors:
        shard, side = d["source_shard"], d["original_group_sidecar"]
        if (shard["source_family"] != "UniProt-kinetics"
                or shard["split"] != "train"
                or len(d["source_full9"]) != 9
                or tuple(d["source_full9"]) != full9(root / shard["path"])
                or shard["rows"] != side["rows"]
                or "effective-groups/" not in side["path"]):
            raise ValueError("Kinetics original shard/role/sidecar changed")
        groups = pq.read_table(pa.BufferReader(read_exact(side, root=root)),
                               use_threads=False).to_pylist()
        if len(groups) != shard["rows"]:
            raise ValueError("Original kinetics group sidecar cardinality mismatch")
        mask = decode_disposition(d, root=root)
        if len(mask) != len(groups):
            raise ValueError("Kinetics original mask length mismatch")
        for ordinal, group in enumerate(groups):
            original_count += 1
            if (set(group) != {"task_id", "source_group_ids_json",
                               "effective_groups_json", "source_file", "evidence_scope"}
                    or group["source_file"] != shard["path"]
                    or group["evidence_scope"] != _SCOPE):
                raise ValueError("Kinetics original group evidence lost its source identity")
            source_groups = _strict_json(group["source_group_ids_json"])
            effective = _strict_json(group["effective_groups_json"])
            if (type(source_groups) is not list or type(effective) is not list
                    or not effective or len(effective) != len(set(effective))
                    or not set(source_groups).issubset(set(effective))):
                raise ValueError("Kinetics original effective group overlay invalid")
            if mask[ordinal] != 0:
                if (shard["sha256"], ordinal) in expected:
                    raise ValueError("Previously retained kinetics source row is now excluded")
                continue
            witness = expected.pop((shard["sha256"], ordinal), None)
            if (witness is None or witness["effective_group_sha256"] != sha256(canonical(effective)).hexdigest()
                    or witness["source_group_sha256"] != sha256(canonical(source_groups)).hexdigest()
                    or type(group["task_id"]) is not str or not _HEX.fullmatch(group["task_id"])):
                raise ValueError("Kinetics identity differs from independent complete source")
            rows.append((group["task_id"], tuple(effective)))
        pins.append((shard["path"], shard["sha256"], side["sha256"], d["primary_retained_rows"]))
    if expected or original_count != 622 or len(rows) != 618:
        raise ValueError("Original kinetics complete train census changed")
    return tuple(rows), dict(original=original_count, retained=len(rows),
                             shards=3, sidecar_pin_sha256=sha256(canonical(pins)).hexdigest(),
                             independent_original_census_sha256=_KINETIC_READBACK_SHA)


def audit_three_source_protein_closure(root):
    """Read actual original train identity sidecars (no task or heldout values)."""
    from mora3.research.primary_campaign_source_rebind import load_inactive_common_source_rebind
    from mora3.research.reactzyme_uniprot_protein_exposure import protein_exposure
    root = Path(root).resolve()
    if root != Path.cwd().resolve():
        raise ValueError("Real original-source checkout required, not isolated or copied data")
    context = load_inactive_common_source_rebind(root)
    if context.common_snapshot.sha256 != _SOURCE_SNAPSHOT:
        raise ValueError("Reviewed 29-head source owner snapshot changed")
    view = load_assets(root)[0]
    if finite_view_policy_sha256(view) != _POLICY:
        raise ValueError("Original shared native view/exclusion policy changed")
    groups = {}
    provenance = {}
    for source, expected in zip(_FAMILIES, (161, 407, 3), strict=True):
        desc = [d for d in view.descriptors if
                d["source_shard"].get("source_family") == source
                and d["source_shard"].get("split") == "train"]
        if len(desc) != expected:
            raise ValueError("Incomplete original eligible protein family inventory")
        if source == "UniProt-kinetics":
            groups[source], provenance[source] = _original_kinetics(root, desc)
        else:
            groups[source], original, pin = _original_retained(root, desc)
            provenance[source] = dict(original=original, retained=len(groups[source]),
                                      shards=expected, sidecar_pin_sha256=pin)
    baseline_path = (Path(__file__).resolve().parents[3] /
                     "outputs/common-source-reactzyme-uniprot-20261111/actual-original-protein-exposure-v2.json")
    baseline_bytes = baseline_path.read_bytes()
    baseline_sha = "d42772a60b7337abd64ea7cc72e15c66c2e824bffe955a1160046d4a521d1f36"
    if sha256(baseline_bytes).hexdigest() != baseline_sha:
        raise ValueError("Independent original ReactZyme/UniProt census changed")
    previous = _strict_json(baseline_bytes)
    paired = protein_exposure(groups["ReactZyme"], groups["UniProt"])
    if paired != previous["exposure"]:
        raise ValueError("Three-family input disagrees with independently audited native pair")
    combined = combined_protein_closure(*(groups[s] for s in _FAMILIES))
    if combined["rows"] != {"ReactZyme": 40464, "UniProt": 103171, "UniProt-kinetics": 618}:
        raise ValueError("Actual three-family original retained cardinalities changed")
    return dict(schema="Mora3.Campaign26OriginalThreeProteinFamilyClosure/v1",
                source_snapshot_sha256=context.common_snapshot.sha256,
                original_train_provenance=provenance,
                independent_original_protein_pair_sha256=baseline_sha,
                closure=combined, source_train_shards=sum(p["shards"] for p in provenance.values()),
                original_heldout_shards_opened=False, source_native_values_opened=False,
                experiment_equivalence_claimed=False, global_split_admitted=False,
                model_construction_owner_attested=False, model_training_admitted=False,
                optimizer_updates=0, actual_muP_proxy_trials=0, actual_target_runs=0)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python -m mora3.research.cross_source_protein_three_family ORIGINAL_ROOT NEW_OUTPUT_JSON")
    result = audit_three_source_protein_closure(sys.argv[1])
    destination = Path(sys.argv[2])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": "qualified_original_train_identity_only",
                      "rows": result["closure"]["rows"],
                      "cross_family_components": result["closure"]["cross_family_components"],
                      "triple_family_components": result["closure"]["triple_family_components"]},
                     sort_keys=True))
