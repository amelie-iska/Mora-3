"""Original BioGRID/Rosetta train grouping audit, not a training-admission claim.

All identity links are literal source-effective group tokens. In particular,
a BioGRID missing-chemical placeholder is NOT a molecular identity; the
sensitivity readout does not silently alter the source's effective policy.
Only provenance/grouping columns of immutable source originals are decoded.
"""
from collections import Counter
from hashlib import sha256
from itertools import combinations
from pathlib import Path
import json
import re

from mora3.data.admet import canonical
from mora3.data.finite_union_native_rows import decode_disposition, full9, read_exact
from mora3.research.remaining_eight_registry import load_assets

_FAMILIES = {
    "biogrid":("/BioGRID/train/","BioGRID","source_annotated_chemical_gene_action",12,2916),
    "l1000":("/Rosetta-L1000/train/","Rosetta-L1000","unconditional_source_profile_modeling",47,11862),
    "cellpainting":("/Rosetta-CellPainting/train/","Rosetta-CellPainting","unconditional_source_profile_modeling",114,29060),
}
_PLACEHOLDER="chemical_identifier:BIOGRID:-"
_GROUP_SCOPE="identity/leakagegroupsonly;notpredictorfeaturesorsamecell/effectproof"
_HEX=re.compile(r"[0-9a-f]{64}\Z")


def digest(x): return sha256(canonical(x)).hexdigest()


def strict_json(s):
    def unique(pairs):
        d={}
        for k,v in pairs:
            if k in d: raise ValueError("Duplicate original JSON key")
            d[k]=v
        return d
    return json.loads(s,object_pairs_hook=unique)


def _components(rows, *, omit_placeholder=False, namespaced=False):
    entries=[r for name in _FAMILIES for r in rows[name]]
    parent=list(range(len(entries)));group_owner={}
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    for i,r in enumerate(entries):
        for g in r["groups"]:
            if omit_placeholder and g==_PLACEHOLDER:continue
            key=(r["source"],g) if namespaced else g
            other=group_owner.get(key)
            if other is None:group_owner[key]=i
            else:
                a,b=find(other),find(i)
                if a!=b:parent[b]=a
    members={}
    for i,r in enumerate(entries):
        members.setdefault(find(i),[]).append(r)
    cross=[rs for rs in members.values() if len({r["source"] for r in rs})>1]
    return {
        "components":len(members),
        "cross_source_components":len(cross),
        "cross_rows":dict(Counter(r["source"] for rs in cross for r in rs)),
        "largest_cross_component_rows":max((len(rs) for rs in cross),default=0),
        "cross_component_member_commitment":digest(sorted(
            digest(sorted(r["task_id"] for r in rs)) for rs in cross)),
    }


def audit_gene_group_closure(rows):
    """Report only original literal group links; never infer gene/protein aliases."""
    if type(rows)is not dict or set(rows)!=set(_FAMILIES):
        raise ValueError("All three complete original source populations required")
    seen=set();groups={}
    for source in _FAMILIES:
        rs=rows[source]
        if type(rs)is not tuple or not rs:raise ValueError("Missing original source population")
        local=set()
        for r in rs:
            if (type(r)is not dict or set(r)!={"source","task_id","groups"}
                or r["source"]!=source or type(r["task_id"])is not str
                or _HEX.fullmatch(r["task_id"]) is None
                or r["task_id"] in seen
                or type(r["groups"])is not tuple or not r["groups"]
                or len(r["groups"])!=len(set(r["groups"]))
                or any(type(g)is not str or ":" not in g
                       or not g.split(":",1)[0] or not g.split(":",1)[1]
                       for g in r["groups"])):
                raise ValueError("Unqualified or duplicate original source/group identity")
            seen.add(r["task_id"])
            local.update(r["groups"])
        groups[source]=local
    pair_genes={}
    pair_studies={}
    for x,y in combinations(_FAMILIES,2):
        pair_genes[f"{x}:{y}"]=len({g for g in groups[x]&groups[y] if g.startswith("gene:9606:")})
        pair_studies[f"{x}:{y}"]=len({g for g in groups[x]&groups[y] if g.startswith("study:")})
    return {
        "source_rows":{s:len(rows[s]) for s in _FAMILIES},
        "source_local_grouping":_components(rows,namespaced=True),
        "global_literal_grouping":_components(rows),
        "placeholder_omitted_sensitivity":_components(rows,omit_placeholder=True),
        "placeholder_rows":sum(_PLACEHOLDER in r["groups"] for r in rows["biogrid"]),
        "placeholder_original_token":_PLACEHOLDER,
        "direct_pairwise_gene_tokens":pair_genes,
        "direct_pairwise_study_tokens":pair_studies,
        "direct_cross_source_tokens_by_class":dict(sorted(Counter(
            g.split(":",1)[0] for g in set.union(*groups.values())
            if sum(g in groups[s] for s in _FAMILIES)>1
        ).items())),
        "grouping_policy":"original literal source effective groups; non-admitted diagnostic removing only missing BioGRID chemical placeholder; no invented gene/protein joins",
        "literal_groups_are_true_chemical_equivalence":False,
        "unpaired_omics_are_paired_effect_claim":False,
        "target_values_read":False,
        "global_split_admitted":False,
        "training_admission":False,
    }


def audit_original_biogrid_rosetta(root):
    """Independent readback of all 173 pinned train-side grouping sidecars."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from mora3.research.primary_campaign_source_rebind import load_inactive_common_source_rebind
    from mora3.research.primary_source_factory_preflight import document
    root=Path(root).resolve()
    if root != Path.cwd().resolve():
        raise ValueError("Audit from exact original active source root")
    ctx=load_inactive_common_source_rebind(root)
    doc=document(root,ctx.evidence)
    if (ctx.common_snapshot.sha256!="79d328f542d1158581875ef325e56f213b407992f95db6ee0c7352a3459669d4"
        or doc["inactive_candidate_snapshot"]["sha256"]!="1091e3f27d57cb70203ab9fce1136836d0cd96eef955b6682a3e6264abadc733"):
        raise ValueError("Reviewed source snapshot binding changed")
    v=load_assets(root)[0]
    by_family={s:[] for s in _FAMILIES}
    for d in v.descriptors:
        p=d["source_shard"]["path"]
        for s,(needle,_,_,_,_) in _FAMILIES.items():
            if needle in p:by_family[s].append(d)
    if {s:len(ds) for s,ds in by_family.items()} != {s:info[3] for s,info in _FAMILIES.items()}:
        raise ValueError("Pinned source shard inventories changed")
    records={s:[] for s in _FAMILIES};found=Counter();excluded=Counter();files=[]
    for s,(needle,fam,task,nshards,nrows) in _FAMILIES.items():
        for d in by_family[s]:
            shard=d["source_shard"];side=d["original_group_sidecar"]
            if (shard["source_family"]!=fam or shard["split"]!="train"
                or full9(root/shard["path"])!=tuple(d["source_full9"])):
                raise ValueError("Changed original source path, family or file identity")
            source=pq.read_table(pa.BufferReader(read_exact(shard,root=root)),
                                 columns=["task_id","source_family","task_name","split","group_ids_json"],
                                 use_threads=False)
            groups=pq.read_table(pa.BufferReader(read_exact(side,root=root)),use_threads=False)
            if (source.num_rows!=shard["rows"] or groups.num_rows!=shard["rows"]
                or set(groups.column_names)!={"effective_groups_json","evidence_scope","source_file",
                    "source_group_ids_json","task_id"}):
                raise ValueError("Original group and native source cardinality/columns changed")
            codes=decode_disposition(d,root=root)
            if len(codes)!=shard["rows"]:raise ValueError("Original disposition length changed")
            retained_this=0
            for i,(a,b,code) in enumerate(zip(source.to_pylist(),groups.to_pylist(),codes,strict=True)):
                found[s]+=1
                if (a["task_id"]!=b["task_id"] or a["source_family"]!=fam or a["task_name"]!=task
                    or a["split"]!="train" or b["source_file"]!=shard["path"] or b["evidence_scope"]!=_GROUP_SCOPE):
                    raise ValueError("Original source task or provenance mismatch")
                native=strict_json(a["group_ids_json"]);side_orig=strict_json(b["source_group_ids_json"])
                effective=strict_json(b["effective_groups_json"])
                if (type(native)is not list or type(effective)is not list or native!=side_orig
                    or not set(native).issubset(effective)
                    or len(effective)!=len(set(effective))):
                    raise ValueError("Invalid source-effective grouping overlay")
                if code!=0 or (shard["path"],i) in v._ordinal_exclusions:
                    excluded[s]+=1
                    continue
                records[s].append({"source":s,"task_id":a["task_id"],"groups":tuple(effective)})
                retained_this+=1
            if retained_this!=d["primary_retained_rows"]:
                raise ValueError("Retained original row disposition disagrees with registered count")
            files.append({"source":s,"source_shard_sha256":shard["sha256"],
                          "original_group_sidecar_sha256":side["sha256"],
                          "original_rows":shard["rows"],"retained_rows":retained_this})
        if found[s]!=nrows:
            raise ValueError("Original family row census changed")
    grouped=audit_gene_group_closure({s:tuple(v) for s,v in records.items()})
    return {
        "schema":"Mora3.OriginalBioGRIDRosettaTrainGroupExposure/v1",
        "common_source_snapshot_sha256":ctx.common_snapshot.sha256,
        "candidate_source_snapshot_sha256":doc["inactive_candidate_snapshot"]["sha256"],
        "original_family_rows":dict(found),
        "original_excluded_rows":dict(excluded),
        "original_shards":len(files),
        "source_original_sidecar_manifest_commitment":digest(files),
        "source_group_closure":grouped,
        "original_source_targets_decoded":False,
        "heldout_original_shards_read":False,
        "optimizer_updates":0,
        "actual_proxy_trials":0,
        "actual_target_runs":0,
        "global_split_admitted":False,
        "same_owner_validation_reference":False,
        "training_admission":False,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv)!=3:raise SystemExit("Usage: python -m mora3.research.cross_source_biogrid_rosetta ACTIVE_ROOT OUTPUT_PATH")
    output=Path(sys.argv[2])
    if output.exists():raise SystemExit("Refusing existing original evidence path")
    evidence=audit_original_biogrid_rosetta(sys.argv[1])
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x") as fh:json.dump(evidence,fh,indent=2,sort_keys=True);fh.write("\n")
    print(json.dumps({
        "original_family_rows":evidence["original_family_rows"],
        "original_excluded_rows":evidence["original_excluded_rows"],
        "native_retained":evidence["source_group_closure"]["source_rows"],
        "grouping":evidence["source_group_closure"],
        "source_manifest_commitment":evidence["source_original_sidecar_manifest_commitment"],
        "result_sha256":sha256(output.read_bytes()).hexdigest()},indent=2,sort_keys=True))
