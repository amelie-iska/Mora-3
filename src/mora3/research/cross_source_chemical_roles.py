"""Original chemical group exposure across four registered 29-head source families.

Read ONLY authenticated original group sidecars. In particular do not open
validation/test source shards or their target, label or condition columns.
This is a scoped leakage diagnostic, not global split or source admission.
"""
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import json
import re

from mora3.data.admet import canonical
from mora3.data.finite_union_native_rows import read_exact
from mora3.data.finite_union_annotation_study_filter import load_study_filtered_native_view
from mora3.model.source_assembly_inputs import CURRENT_VIEW_PINS
from mora3.research.primary_native_routes import finite_view_policy_sha256

_GROUP_VIEW_POLICY = 'c3812ecfcfea8a01f81769bf5ef6fb3880e04a0ac054a6349e9272db14765393'


def _load_group_metadata_view(root):
    """Load only the pinned study-filtered descriptor and ordinal metadata.

    The full task registry also constructs inverse-literal readers that reopen
    original target-containing shards. A group audit must not trigger that work.
    """
    root = Path(root).resolve()
    path, pin = CURRENT_VIEW_PINS['effective_source_view']
    view = load_study_filtered_native_view(root / path, pin, root=root)
    if finite_view_policy_sha256(view) != _GROUP_VIEW_POLICY:
        raise ValueError('Exact original metadata group-view policy required')
    return view

ROLES={
    "AZ":("train",),
    "FreeSolv":("train","validation","test"),
    "Tox21":("train","validation","test"),
    "ExpansionRx":("train","test"),
}
FAMILIES={
    "AZ":("ChEMBL37-AZ-experimental-logD7_4",{"train":17}),
    "FreeSolv":("FreeSolv",{"train":3,"validation":1,"test":1}),
    "Tox21":("Tox21-original-source-call-prior",{"train":51,"validation":1,"test":2}),
    "ExpansionRx":("ExpansionRx-source-native-ADMET-readouts",{"train":76,"test":45}),
}
_COMMON="79d328f542d1158581875ef325e56f213b407992f95db6ee0c7352a3459669d4"
_CANDIDATE="1091e3f27d57cb70203ab9fce1136836d0cd96eef955b6682a3e6264abadc733"
_HEX=re.compile(r"[0-9a-f]{64}\Z")
_GROUPS=("molecule","scaffold")
_STATUS={"train":0,"validation":1,"test":2}


def digest(obj):
    return sha256(canonical(obj)).hexdigest()


def _strict_json(raw):
    def unique(pairs):
        d={}
        for k,v in pairs:
            if k in d:raise ValueError("Duplicate original source group JSON key")
            d[k]=v
        return d
    return json.loads(raw,object_pairs_hook=unique)


def _pair(name_a,rows_a,name_b,rows_b):
    result={"left_rows":len(rows_a),"right_rows":len(rows_b)}
    for kind in _GROUPS:
        a=defaultdict(set);b=defaultdict(set)
        for r in rows_a:a[r[kind]].add(r["task_id"])
        for r in rows_b:b[r[kind]].add(r["task_id"])
        common=sorted(a.keys()&b.keys())
        result[kind]={
            "shared_hashes":len(common),
            "left_tasks":len(set().union(*(a[k] for k in common))) if common else 0,
            "right_tasks":len(set().union(*(b[k] for k in common))) if common else 0,
            "collision_commitment_sha256":digest([
                (k,sorted(a[k]),sorted(b[k])) for k in common]),
        }
    return result


def compare_scoped_role_groups(roles):
    """Exactly scoped literal SHA256 group equality, no chemical equivalence inference."""
    if type(roles) is not dict or set(roles)!=set(ROLES):
        raise ValueError("Missing/extra source role family")
    seen=set();role_map={}
    for source,expected in ROLES.items():
        got=roles[source]
        if type(got) is not dict or set(got)!=set(expected):
            raise ValueError("Missing or unexpected source role")
        for role in expected:
            rows=got[role]
            if type(rows) is not tuple or not rows:
                raise ValueError("Missing source role rows")
            for r in rows:
                if (type(r) is not dict or set(r)!={"task_id","molecule","scaffold"}
                        or type(r["task_id"]) is not str
                        or _HEX.fullmatch(r["task_id"]) is None or r["task_id"] in seen):
                    raise ValueError("Duplicate, invalid or unbound original task identity")
                seen.add(r["task_id"])
                for kind in _GROUPS:
                    g=r[kind]
                    if (type(g) is not str
                            or re.fullmatch(kind+r":[0-9a-f]{64}",g) is None):
                        raise ValueError("Invalid source-native chemical group hash")
            role_map[source+"."+role]=rows
    trains=[s+".train" for s in ROLES]
    heldouts=[s+"."+r for s in ROLES for r in ROLES[s] if r!="train"]
    required=[(h,t) for h in heldouts for t in trains]
    required.extend((heldouts[i],heldouts[j])
                    for i in range(len(heldouts)) for j in range(i+1,len(heldouts)))
    pairs={h+"|"+t:_pair(h,role_map[h],t,role_map[t]) for h,t in required}
    # Positive-control view: train/train intersections reveal whether native
    # group-hash vocabularies overlap. Even a positive match cannot establish
    # stereochemical equivalence without independent source structure evidence.
    train_pairs=[(trains[i],trains[j]) for i in range(len(trains))
                 for j in range(i+1,len(trains))]
    train_collisions={a+"|"+b:_pair(a,role_map[a],b,role_map[b])
                      for a,b in train_pairs}
    return {
        "role_rows":{k:len(v) for k,v in role_map.items()},
        "required_pairs":[h+"|"+t for h,t in required],
        "pairwise":pairs,
        "train_pairwise":train_collisions,
        "any_heldout_vs_train_molecule_hash_collision":any(
            pairs[h+"|"+t]["molecule"]["shared_hashes"]>0
            for h in heldouts for t in trains),
        "any_heldout_vs_train_scaffold_hash_collision":any(
            pairs[h+"|"+t]["scaffold"]["shared_hashes"]>0
            for h in heldouts for t in trains),
        "original_hash_protocol":"Literal original source-native molecule/scaffold hashes only",
        "source_group_hash_domains_independently_equated":False,
        "exact_stereochemical_identity_proven":False,
        "global_split_admitted":False,
        "training_admission":False,
    }


def audit_original_chemical_roles(root, *, include_verified_roles=False):
    """Authenticate registered sidecars, never open original heldout source data.

    The opt-in verified rows are in-memory only for independent group closure;
    the ordinary immutable readback and command-line output remain identical.
    """
    if type(include_verified_roles) is not bool:
        raise ValueError("Exact boolean verified-role option required")
    import pyarrow as pa
    import pyarrow.parquet as pq
    from mora3.research.primary_campaign_source_rebind import load_inactive_common_source_rebind
    from mora3.research.primary_source_factory_preflight import document

    root=Path(root).resolve()
    if root!=Path.cwd().resolve():
        raise ValueError("Execute in exact authenticated active source root")
    context=load_inactive_common_source_rebind(root)
    document_readback=document(root,context.evidence)
    if (context.common_snapshot.sha256!=_COMMON
            or document_readback["inactive_candidate_snapshot"]["sha256"]!=_CANDIDATE):
        raise ValueError("Frozen common source or candidate snapshot has changed")
    v=_load_group_metadata_view(root)
    by_role={s:{r:[] for r in ROLES[s]} for s in ROLES}
    registered={f:s for s,(f,_) in FAMILIES.items()}
    for d in v.descriptors:
        shard=d["source_shard"]
        family=shard.get("source_family")
        if family in registered:
            src=registered[family];role=shard.get("split")
            if role not in ROLES[src]:
                raise ValueError("Unexpected original role for scoped source")
            by_role[src][role].append(d)
    proofs=[];rows={s:{} for s in ROLES};excluded={}
    base=(root/"outputs/data-intake").resolve()
    for source in ROLES:
        for role in ROLES[source]:
            descriptors=by_role[source][role]
            if len(descriptors)!=FAMILIES[source][1][role]:
                raise ValueError("Original source/role shard inventory changed")
            accepted=[];excluded_count=0
            for d in descriptors:
                shard=d["source_shard"];side=d["original_group_sidecar"]
                source_path=(root/shard["path"]).resolve()
                side_path=(root/side["path"]).resolve()
                # Never read/have a parser open an original source numerical shard.
                if (base not in source_path.parents or base not in side_path.parents
                        or "/"+role+"/" not in "/"+shard["path"]
                        or not source_path.is_file() or source_path.stat().st_size!=shard["bytes"]
                        or not side_path.is_file() or shard["split"]!=role
                        or shard["source_family"]!=FAMILIES[source][0]):
                    raise ValueError("Source role, path or physical byte count changed")
                raw=read_exact(side,root=root)
                table=pq.read_table(pa.BufferReader(raw),
                                    columns=["task_id","effective_groups_json"],use_threads=False)
                if table.num_rows!=shard["rows"] or table.num_rows!=side["rows"]:
                    raise ValueError("Sidecar and original registered row counts differ")
                disposition=d["disposition"]
                if (disposition.get("kind")!="homogeneous-original-row-disposition"
                        or disposition.get("code")!=_STATUS[role]
                        or disposition.get("rows")!=shard["rows"]):
                    raise ValueError("Changed original train/validation/test row disposition")
                kept=0
                for ordinal,r in enumerate(table.to_pylist()):
                    if (shard["path"],ordinal) in v._ordinal_exclusions:
                        excluded_count+=1
                        continue
                    groups=_strict_json(r["effective_groups_json"])
                    if (type(groups) is not list or len(groups)!=len(set(groups))
                            or any(type(g) is not str for g in groups)):
                        raise ValueError("Invalid original effective groups")
                    selected={}
                    for kind in _GROUPS:
                        matches=[g for g in groups if g.startswith(kind+":")]
                        if len(matches)!=1 or re.fullmatch(kind+r":[0-9a-f]{64}",matches[0]) is None:
                            raise ValueError("Missing/duplicate original chemical group")
                        selected[kind]=matches[0]
                    accepted.append({"task_id":r["task_id"],**selected})
                    kept+=1
                if kept!=d["primary_retained_rows"]:
                    raise ValueError("Original retained source disposition changed")
                proofs.append({
                    "source":source,"role":role,
                    "source_path":shard["path"],"source_sha256_pin":shard["sha256"],
                    "source_bytes_stat_only":shard["bytes"],
                    "sidecar_path":side["path"],"sidecar_sha256":side["sha256"],
                    "original_rows":shard["rows"],"retained_group_rows":kept,
                    "source_bytes_read":0})
            rows[source][role]=tuple(accepted)
            excluded[source+"."+role]=excluded_count
    compared=compare_scoped_role_groups(rows)
    result={
        "schema":"Mora3.OriginalFourFamilyChemicalRoleGroupExposure/v1",
        "common_source_snapshot_sha256":_COMMON,
        "candidate_source_snapshot_sha256":_CANDIDATE,
        "original_sidecar_shards":len(proofs),
        "source_role_file_commitment":digest(proofs),
        "ordered_retained_group_commitment":digest(
            [(s,r,[(x["task_id"],x["molecule"],x["scaffold"]) for x in rows[s][r]])
             for s in ROLES for r in ROLES[s]]),
        "original_excluded_role_rows":excluded,
        "heldout_native_shards_opened":False,
        "heldout_target_or_condition_columns_decoded":False,
        "source_numeric_shards_read":False,
        "source_snapshot_metadata_is_not_live_training_admission":True,
        "original_source_proofs":proofs,
        "comparison":compared,
        "optimizer_updates":0,
        "actual_proxy_runs":0,
        "actual_target_runs":0,
        "training_admission":False,
    }
    if include_verified_roles:
        result["_in_memory_verified_original_roles"] = rows
    return result


if __name__=="__main__":
    import sys
    if len(sys.argv)!=3:raise SystemExit("Usage: python -m mora3.research.cross_source_chemical_roles ACTIVE_ROOT EXCLUSIVE_OUTPUT")
    out=Path(sys.argv[2])
    if out.exists():raise SystemExit("Refusing to overwrite original source qualification evidence")
    result=audit_original_chemical_roles(sys.argv[1])
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("x") as fh:json.dump(result,fh,indent=2,sort_keys=True);fh.write("\n")
    print(json.dumps({
        "roles":result["comparison"]["role_rows"],
        "scoped_pairs":len(result["comparison"]["pairwise"]),
        "any_train_vs_heldout_molecule_overlap":result["comparison"]["any_heldout_vs_train_molecule_hash_collision"],
        "any_train_vs_heldout_scaffold_overlap":result["comparison"]["any_heldout_vs_train_scaffold_hash_collision"],
        "original_sidecar_shards":result["original_sidecar_shards"],
        "immutable_readback_sha256":sha256(out.read_bytes()).hexdigest(),
    },indent=2,sort_keys=True))
