"""Exhaust original Rfam15.1 dual-direction source, not an admitted campaign route.

The same source-native parent produces a sequence->family prediction and a
family->sequence generation view. Both are source-assigned reference labels;
they are not independent experiments, negative examples, or observed effects.
No heldout targets, optimizer, model weights or campaign slots are accessed.
"""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json
import re

from mora3.data.admet import canonical

_FAMILY = "Rfam15.1-source-family-prior"
_REVISION = "Rfam15.1 January2026"
_PRED = "Rfam_source_family_prediction_prior"
_GEN = "Rfam_source_family_generation_prior"
_EVIDENCE = "source_covariance_model_assigned_reference_family_annotation;notexperimentaltruth"
_HEX = re.compile(r"[0-9a-f]{64}\Z")

def _hash(x):
    return sha256(canonical(x)).hexdigest()

def _unique(pairs):
    result = {}
    for k,v in pairs:
        if k in result: raise ValueError("Duplicate original source JSON key")
        result[k] = v
    return result

def _strict(raw):
    value = json.loads(raw, object_pairs_hook=_unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite native JSON")))
    if type(value) is not dict:raise ValueError("Native source JSON must be an object")
    return value

def validate_dual_record_pair(pred, gen):
    """Check two genuinely separate supervision roles share exactly one parent."""
    if (type(pred) is not dict or type(gen) is not dict
        or set(pred) != {"schema","task","conditions","targets","audit","missing"}
        or set(gen) != set(pred)
        or pred["schema"] != "mora3.CorpusTask/v1"
        or gen["schema"] != pred["schema"]
        or pred["task"] != _PRED or gen["task"] != _GEN):
        raise ValueError("Distinct original Rfam prediction/generation directions required")
    pc,pt = pred["conditions"],pred["targets"]
    gc,gt = gen["conditions"],gen["targets"]
    if (type(pc) is not dict or set(pc) != {"source_native_sequence"}
        or type(pt) is not dict or set(pt) != {"source_Rfam_family_ids","source_clan_ids","family_label_policy"}
        or type(gc) is not dict or set(gc) != {"desired_source_Rfam_family_ids","desired_source_clan_ids"}
        or type(gt) is not dict or set(gt) != {"source_native_sequence"}):
        raise ValueError("Original directional condition/target vocabulary changed")
    s = pc["source_native_sequence"]
    if (type(s) is not str or not s or s != gt["source_native_sequence"]
        or pt["source_Rfam_family_ids"] != gc["desired_source_Rfam_family_ids"]
        or pt["source_clan_ids"] != gc["desired_source_clan_ids"]):
        raise ValueError("Source-native sequence or assigned family/clan pairing differs")
    if not all(type(v) is list and len(v) == len(set(v))
        and all(type(x) is str and x for x in v)
        for v in (pt["source_Rfam_family_ids"],pt["source_clan_ids"])):
        raise ValueError("Original family/clan axis must be a unique ordered typed list")
    if (type(pt["family_label_policy"]) is not str or not pt["family_label_policy"]
        or type(pred["audit"]) is not dict or type(gen["audit"]) is not dict
        or not all(k in pred["audit"] and k in gen["audit"]
           for k in ("source_release","source_evidence","negative_labels_available"))
        or any(pred["audit"][k] != gen["audit"][k]
           for k in ("source_release","source_evidence","negative_labels_available"))
        or pred["audit"]["negative_labels_available"] is not False
        or type(pred["missing"]) is not list or type(gen["missing"]) is not list):
        raise ValueError("Source-assigned nonnegative evidence provenance differs")
    return {
        "same_parent_observation": True,
        "source_native_sequence_length": len(s),
        "family_ids": len(pt["source_Rfam_family_ids"]),
        "clan_ids": len(pt["source_clan_ids"]),
        "source_assigned_evidence_only": True,
        "negative_labels_available": False,
    }

def _parent_components(parents):
    """Only declared source-effective tokens, not independently validated splits."""
    owner={};link=list(range(len(parents)))
    def find(i):
        while link[i]!=i:
            link[i]=link[link[i]]
            i=link[i]
        return i
    for i,(_,groups) in enumerate(parents):
        for token in groups:
            j=owner.setdefault(token,i)
            a,b=find(i),find(j)
            if a != b:link[a]=b
    comp={}
    for i,(identifier,_) in enumerate(parents):
        comp.setdefault(find(i),[]).append(identifier)
    components=sorted((tuple(sorted(v)) for v in comp.values()),key=_hash)
    hist=dict(sorted(Counter(map(len,components)).items()))
    return dict(connected_components=len(components),
        largest_component_parents=max(map(len,components),default=0),
        size_histogram=hist,
        component_membership_sha256=_hash(components),
        globally_heldout_leakage_safe=False)

def audit_original_rfam_pairs(root):
    """Verify all original 72 Rfam shards, effective groups and dual-role parents."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from mora3.research.remaining_eight_registry import load_assets
    from mora3.data.finite_union_native_rows import read_exact,full9,decode_disposition
    from mora3.research.primary_campaign_source_rebind import load_inactive_common_source_rebind
    root=Path(root).resolve()
    if root!=Path.cwd().resolve():raise ValueError("Original active source root required")
    context=load_inactive_common_source_rebind(root)
    if context.common_snapshot.sha256 != "79d328f542d1158581875ef325e56f213b407992f95db6ee0c7352a3459669d4":
        raise ValueError("Staged 29-head common source identity changed")
    view=load_assets(root)[0]
    ds=[d for d in view.descriptors if "rfam-family-annotation-v1/curated-v1/shards/" in d["source_shard"]["path"]]
    if len(ds)!=72:raise ValueError("Complete original Rfam source shard inventory required")
    ds.sort(key=lambda d:d["source_shard"]["path"])
    by_parent={}; files=[]; tasks=Counter(); evidence=Counter(); flags=Counter()
    for d in ds:
        shard,side=d["source_shard"],d["original_group_sidecar"]
        path=shard["path"]
        if (shard["source_family"]!=_FAMILY or shard["split"]!="train"
            or full9(root/path)!=tuple(d["source_full9"])):
            raise ValueError("Original Rfam source/family/revision/physical identity changed")
        source=pq.read_table(pa.BufferReader(read_exact(shard,root=root)),use_threads=False)
        groups=pq.read_table(pa.BufferReader(read_exact(side,root=root)),use_threads=False)
        codes=decode_disposition(d,root=root)
        if (source.num_rows!=shard["rows"] or groups.num_rows!=source.num_rows
            or len(codes)!=source.num_rows
            or set(groups.column_names)!={"task_id","effective_groups_json"}
            or set(source.column_names)!={"task_id","record_id","source_family","source_revision",
              "source_sha256","task_name","native_schema","native_payload","native_payload_sha256",
              "split","split_policy","group_ids_json","evidence_class","admitted_for_task",
              "model_adapter_supported","admission_receipt_json","missing_context_json",
              "full_axis_values","condition_context_json"}):
            raise ValueError("Original full native Rfam rows/sidecars/ordinals changed")
        retained=0
        for ordinal,(row,group,code) in enumerate(zip(source.to_pylist(),groups.to_pylist(),codes,strict=True)):
            if (code!=0 or (path,ordinal) in view._ordinal_exclusions
                or row["task_name"] not in (_PRED,_GEN)
                or row["source_family"]!=_FAMILY or row["source_revision"]!=_REVISION
                or row["split"]!="train" or row["native_schema"]!="mora3.CorpusTask/v1"
                or row["evidence_class"]!=_EVIDENCE
                or type(row["admitted_for_task"]) is not bool
                or row["admitted_for_task"] is not True
                or row["model_adapter_supported"] is not False
                or group["task_id"]!=row["task_id"]
                or not all(type(row[k]) is str and _HEX.fullmatch(row[k]) is not None
                    for k in ("task_id","record_id","source_sha256","native_payload_sha256"))
                or type(row["native_payload"]) is not bytes
                or sha256(row["native_payload"]).hexdigest()!=row["native_payload_sha256"]):
                raise ValueError("Unqualified source row, role, model flag, payload hash or excluded ordinal")
            native=_strict(row["native_payload"])
            local=_strict('{"groups":'+row["group_ids_json"]+'}')["groups"]
            effective=_strict('{"groups":'+group["effective_groups_json"]+'}')["groups"]
            if (type(local) is not list or not local or type(effective) is not list or not effective
                or len(set(local))!=len(local) or len(set(effective))!=len(effective)
                or not set(local).issubset(effective)
                or any(type(g) is not str or ":" not in g for g in effective)
                or native.get("task")!=row["task_name"]):
                raise ValueError("Original/effective grouping, direction or target roles changed")
            item=by_parent.setdefault(row["record_id"],{})
            if row["task_name"] in item:raise ValueError("Duplicate Rfam directional task for one parent")
            item[row["task_name"]]=(native,tuple(effective),row["source_sha256"],
                                  row["task_id"],row["native_payload_sha256"])
            retained+=1;tasks[row["task_name"]]+=1
            evidence[row["evidence_class"]]+=1
            flags[(row["admitted_for_task"],row["model_adapter_supported"])]+=1
        if retained!=d["primary_retained_rows"]:
            raise ValueError("Original Rfam retained disposition changed")
        files.append(dict(source_shard_sha256=shard["sha256"],
            source_sidecar_sha256=side["sha256"],rows=shard["rows"],retained=retained))
    if len(by_parent)!=18423 or sum(tasks.values())!=36846 or set(tasks)!={_PRED,_GEN}:
        raise ValueError("Complete Rfam source population and exact paired-parent coverage required")
    parents=[];profiles=Counter();pair_commitment=[]
    for parent,entries in sorted(by_parent.items()):
        if set(entries)!={_PRED,_GEN}:
            raise ValueError("Missing Rfam direction for original parent")
        pred,pg,ps,pt,ph=entries[_PRED]
        gen,gg,gs,gt,gh=entries[_GEN]
        if pg!=gg or ps!=gs or pt==gt:
            raise ValueError("Rfam source parent grouping/provenance or distinct task identity differs")
        check=validate_dual_record_pair(pred,gen)
        profiles["family_label_count"]+=check["family_ids"]
        profiles["clan_label_count"]+=check["clan_ids"]
        profiles["sequence_nucleotide_positions"]+=check["source_native_sequence_length"]
        parents.append((parent,pg))
        pair_commitment.append((parent,pt,gt,ph,gh,_hash(pg)))
    result=dict(schema="Mora3.OriginalRfamCompleteDualDirectionCensus/v1",
        source_family=_FAMILY,source_revision=_REVISION,
        source_snapshot_sha256=context.common_snapshot.sha256,
        original_shards=len(ds),source_rows=sum(tasks.values()),
        parent_observations=len(parents),
        prediction_rows=tasks[_PRED],generation_rows=tasks[_GEN],
        original_group_sidecar_manifest_sha256=_hash(files),
        paired_role_evidence_sha256=_hash(pair_commitment),
        exact_source_role_pair_counts=dict(tasks),actual_train_projection_counts=dict(profiles),
        parent_source_group_components=_parent_components(parents),
        all_parent_pairs_role_equal=True,
        source_evidence_classes=dict(evidence),admission_vs_adapter_flags={str(k):v for k,v in flags.items()},
        model_adapter_ready=False,independent_experimental_labels=False,
        global_split_admitted=False,heldout_targets_opened=False,
        active_native_62_route_replay_complete=False,
        training_admission=False,optimizer_updates=0,actual_proxy_trials=0,actual_target_runs=0,
        note="Original source-assigned Rfam family/clan reference directions, not 36846 independent experiments; no matched model validation")
    return result

if __name__=="__main__":
    import sys
    if len(sys.argv)!=3:raise SystemExit("Usage: python -m mora3.research.rfam_original_dual_direction_census ACTIVE_ROOT NEW_RESULT")
    output=Path(sys.argv[2])
    if output.exists():raise SystemExit("Refusing to overwrite existing evidence")
    value=audit_original_rfam_pairs(sys.argv[1])
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x") as f:json.dump(value,f,indent=2,sort_keys=True);f.write("\n")
    print(json.dumps({k:value[k] for k in ("original_shards","source_rows","parent_observations","parent_source_group_components","training_admission")},sort_keys=True))
