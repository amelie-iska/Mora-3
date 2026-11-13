"""Read two real, frozen STRING v12.0 native source windows in two owner bindings.

Scope is *only* the reviewed two-window train replay. STRING confidence is a
computed association score, not a direct experimental protein-binding label.
No evaluated model, optimizer, training admission, or validation-source read.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import json

from mora3.data.admet import canonical
from mora3.data.string_score_current import SCORE_AXIS, UNIT, FAMILY, TASK as SOURCE_TASK, valid_score_profile
from mora3.research.native_annotation_registry import load_original_annotation_assets, VIEW
from mora3.research.native_annotation_campaign_routes import bind_string
from mora3.research.mup_campaign import executable_adapter_sha256
from mora3.research.primary_campaign_source_rebind import load_inactive_common_source_rebind
from mora3.research.primary_source_factory_preflight import document
from mora3.research.primary_native_routes import finite_view_policy_sha256

TASK="string_score_profile"
COMMON="79d328f542d1158581875ef325e56f213b407992f95db6ee0c7352a3459669d4"
CANDIDATE="1091e3f27d57cb70203ab9fce1136836d0cd96eef955b6682a3e6264abadc733"

def h(blob):
    return sha256(blob).hexdigest()

def verified_score_packet(source_target, actual):
    """Verify four original integer score axes without converting evidence class."""
    if (type(source_target) is not dict or set(source_target)!=set(SCORE_AXIS)
            or not valid_score_profile(actual)
            or any(type(source_target[key]) is not int or
                   source_target[key]!=value for key,value in zip(SCORE_AXIS,actual,strict=True))):
        raise ValueError("Exact native STRING four-axis integer source target changed")
    return len(actual)

def _one_record(window, source, binding):
    b=window.batch
    if (window.original_source_rows!=1 or b.original_split!="train"
            or b.native_rows.num_rows!=1):
        raise ValueError("One original train STRING source record required")
    source._view.require_native_rows(b,split="train")
    row,=b.native_rows.to_pylist()
    raw=row["native_payload"]
    if type(raw) is not bytes or h(raw)!=row["native_payload_sha256"]:
        raise ValueError("Original native STRING payload SHA-256 mismatch")
    literal=json.loads(raw)
    if (canonical(literal)!=raw or literal.get("schema")!="mora3.CorpusTask/v1"
            or literal.get("task")!=SOURCE_TASK):
        raise ValueError("Original canonical STRING source task changed")
    if (row["source_family"]!=FAMILY or row["split"]!="train"
            or row["evidence_class"]!="source_integrated_computed_association_confidence_prior_not_experimental_binding"):
        raise ValueError("Original STRING evidence class or train role changed")
    packets=source.packets(b,split="train")
    packet,=packets
    four=verified_score_packet(
        {axis:literal["targets"][axis]["value"] for axis in SCORE_AXIS},packet.scores)
    if any(literal["targets"][axis]["unit"]!=UNIT for axis in SCORE_AXIS):
        raise ValueError("Original native source confidence unit changed")
    if packet.source_coordinate!=(b.source_shard["path"],b.source_row_ordinals[0],h(raw)):
        raise ValueError("Original native source coordinate or payload changed")
    prepared=binding.prepare(window)
    if prepared.observed_positions!=four or prepared.evaluate is None:
        raise ValueError("Exact source-only target preparation count changed")
    return {
        "native_payload_sha256":h(raw),
        "source_shard_sha256":b.source_shard["sha256"],
        "source_shard_path":b.source_shard["path"],
        "source_row_ordinal":b.source_row_ordinals[0],
        "source_task_id_sha256":h(row["task_id"].encode()),
        "native_integer_score_sha256":h(canonical(packet.scores)),
        "same_release_sequences_sha256":h(canonical(packet.condition.sequences)),
        "provider_pair_sha256":h(canonical(packet.provider_pair)),
        "group_sha256":h(packet.source_pair_group.encode()),
        "source_oriented_lines_sha256":h(canonical(packet.oriented_source_lines)),
        "native_target_positions":four,
        "evidence_class":row["evidence_class"],
        "score_unit":UNIT,
    }

def _two(binding,source):
    it=binding.route.iter_from(binding.route.initial_cursor)
    try:
        first=next(it);second=next(it)
    finally:
        close=getattr(it,"close",None)
        if close:close()
    if (first.cursor_after==second.cursor_after or
            (first.batch.source_shard==second.batch.source_shard and
             first.batch.source_row_ordinals==second.batch.source_row_ordinals)):
        raise ValueError("Original STRING native cursor failed to advance")
    it=binding.route.iter_from(first.cursor_after)
    try:
        resumed=next(it)
    finally:
        close=getattr(it,"close",None)
        if close:close()
    if (second.cursor_after!=resumed.cursor_after or
            second.batch.source_shard!=resumed.batch.source_shard or
            second.batch.source_row_ordinals!=resumed.batch.source_row_ordinals or
            not second.batch.native_rows.equals(resumed.batch.native_rows,check_metadata=True)):
        raise ValueError("Native STRING generator cursor recovery differs")
    return (first,second),[_one_record(w,source,binding) for w in (first,second)]

def audit_string_pair(root):
    root=Path(root).resolve()
    if not root.is_dir():
        raise ValueError("Authenticated original source checkout required")
    context=load_inactive_common_source_rebind(root)
    frozen=document(root,context.evidence)
    rows=[r for r in frozen["route_bindings"] if r["task_key"]==TASK]
    if len(rows)!=1:
        raise ValueError("Exact one frozen STRING source task required")
    row,=rows
    if (frozen["training_authorized"] is not False or
        frozen["global_split_accepted"] is not False or
        len(frozen["route_bindings"])!=62 or
        row["common_view_policy_sha256"]!=VIEW or
        row["candidate"]["max_windows"]!=2 or
        row["common"]["max_windows"]!=2):
        raise ValueError("Frozen two-window string source is not the inactive campaign")
    candidate=frozen["inactive_candidate_snapshot"]["sha256"]
    common=frozen["common_source_snapshot"]["sha256"]
    if candidate!=CANDIDATE or common!=COMMON:
        raise ValueError("Frozen candidate/common owner snapshots changed")
    view,sources,availability,_,_=load_original_annotation_assets(root)
    if finite_view_policy_sha256(view)!=VIEW:
        raise ValueError("Current native STRING original view has changed")
    source=sources["string"]
    if (availability[TASK]["train_rows"]!=474656
            or availability[TASK]["validation_rows"]!=0
            or availability[TASK]["test_rows"]!=0):
        raise ValueError("Original train-only STRING source census changed")
    bindings={}
    for label,snapshot in (("candidate",candidate),("common",common)):
        b=bind_string(source,view,snapshot_sha256=snapshot,batch_size=1)
        got={"task_key":b.route.task_key,"max_windows":2,
             "initial_cursor_sha256":h(b.route.initial_cursor),
             "adapter_sha256":executable_adapter_sha256(b.prepare,root=Path(__file__).resolve().parents[3])}
        if got!=row[label] or b.route.source_snapshot_sha256!=snapshot:
            raise ValueError("Live STRING adapter/cursor differs from frozen "+label)
        bindings[label]=b
    originals,old=_two(bindings["candidate"],source)
    rebound,new=_two(bindings["common"],source)
    for a,b in zip(originals,rebound,strict=True):
        if (a.batch.source_shard!=b.batch.source_shard
                or a.batch.source_row_ordinals!=b.batch.source_row_ordinals
                or not a.batch.native_rows.equals(b.batch.native_rows,check_metadata=True)):
            raise ValueError("Candidate and common original STRING Arrow rows differ")
    if old!=new or sum(r["native_target_positions"] for r in old)!=8:
        raise ValueError("Candidate/common STRING native score positions differ")
    return {
        "schema":"Mora3.Campaign26StringTwoOriginalCommon/v1",
        "task_key":TASK,"windows_per_binding":2,
        "native_targets_per_binding":8,"candidate_common_native_equal":True,
        "candidate_snapshot":candidate,"source_snapshot":common,
        "source_view_policy":VIEW,"source_census_rows":474656,
        "frozen_candidate":row["candidate"],"frozen_common":row["common"],
        "original_source_records":old,
        "evidence_semantics":"STRING-v12 computed/integrated association confidence, not direct experimental physical binding",
        "source_replay_scope":"two frozen original train windows only, not entire STRING corpus or complete 62-route admission",
        "training_admission":False,"global_split_admitted":False,
        "optimizer_updates":0,"actual_counted_trials":0,
    }

if __name__=="__main__":
    import os,sys
    if len(sys.argv)!=3:raise SystemExit("Usage: python -m mora3.research.campaign26_string_original_common ACTIVE_ROOT NEW_JSON")
    if os.uname().nodename!="iska":raise ValueError("Iska workstation required")
    dst=Path(sys.argv[2])
    if dst.exists():raise FileExistsError("Original evidence must not be overwritten")
    data=audit_string_pair(sys.argv[1])
    dst.parent.mkdir(parents=True,exist_ok=True)
    with dst.open("x") as f:
        json.dump(data,f,sort_keys=True,indent=2);f.write("\n")
    print(json.dumps({"audit_sha256":h(dst.read_bytes()),"windows_per_binding":2,
                      "target_positions":data["native_targets_per_binding"]},sort_keys=True))
