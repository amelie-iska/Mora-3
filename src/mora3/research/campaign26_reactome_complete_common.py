"""Full original Reactome98 *source-only* readback under an exact frozen common binding.

A full single-pass native common-source scan and a two-window frozen candidate
comparison are distinct from the 2-window campaign allocation, training admission,
a validated model, or an independent biological split. No objective is evaluated.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json
import os

from mora3.data.admet import canonical
from mora3.data.reactome_static_pathways import FAMILY, TASK, SCHEMA, ROLE_AXIS
from mora3.data.finite_union_native_rows import decode_disposition, full9
from mora3.research.native_annotation_registry import load_original_annotation_assets, VIEW
from mora3.research.native_annotation_campaign_routes import bind_reactome
from mora3.research.primary_campaign_source_rebind import load_inactive_common_source_rebind
from mora3.research.primary_source_factory_preflight import document
from mora3.research.primary_native_routes import finite_view_policy_sha256
from mora3.research.mup_campaign import executable_adapter_sha256

KEY="reactome_static_participant_roles"
COMMON="79d328f542d1158581875ef325e56f213b407992f95db6ee0c7352a3459669d4"
CANDIDATE="1091e3f27d57cb70203ab9fce1136836d0cd96eef955b6682a3e6264abadc733"


def h(raw):
    return sha256(raw).hexdigest()


def verify_reactome_masked_packet(packet, graph_slots, prepared):
    """Check that every withheld input incidence is target-only, and count native roles."""
    graph, slots=graph_slots
    targets=packet.targets
    if (not isinstance(targets,tuple) or not isinstance(slots,tuple)
            or len(targets)!=len(slots) or not slots):
        raise ValueError("Native Reactome masked participant axes differ")
    for slot in slots:
        if (type(slot) is not int or slot<len(graph.nodes)
                or slot>=len(graph.nodes)+len(graph.edges)
                or graph.edges[slot-len(graph.nodes)].measurements):
            raise ValueError("Reactome withheld role leaked into known input mask")
    supplied=0
    for item in targets:
        if (not isinstance(item,tuple) or len(item)!=2 or item[0] not in ROLE_AXIS):
            raise ValueError("Native Reactome target role invalid")
        if item[1] is not None:
            supplied+=1
    count=len(targets)+supplied
    if (prepared.observed_positions!=count or prepared.unavailable_reason is not None
            or not callable(prepared.evaluate)):
        raise ValueError("Native Reactome target coordinate count or gated objective differs")
    return len(targets),supplied


def _record(window, source, binding):
    row_count=window.original_source_rows
    if row_count!=1 or window.batch.original_split!="train":
        raise ValueError("Single complete original Reactome train row required")
    b=window.batch
    source._view.require_native_rows(b,split="train")
    row,=b.native_rows.to_pylist()
    raw=row["native_payload"]
    if (type(raw) is not bytes or h(raw)!=row["native_payload_sha256"]
            or (row["source_family"],row["task_name"],row["native_schema"],row["split"])
                !=(FAMILY,TASK,SCHEMA,"train")):
        raise ValueError("Native Reactome original identity, provenance or payload differs")
    packet,=source.packets(b,split="train")
    if packet.source_coordinate!=(b.source_shard["path"],b.source_row_ordinals[0],h(raw)):
        raise ValueError("Native Reactome source coordinate mismatch")
    counts=verify_reactome_masked_packet(packet,packet.condition.graph(),binding.prepare(window))
    return (
        b.source_shard["path"],b.source_row_ordinals[0],
        h(canonical([row["task_id"],row["native_payload_sha256"]])),
        h(canonical(packet.targets)),counts,
    )


def _frozen_prefix(binding, source, count=2):
    iterator=binding.route.iter_from(binding.route.initial_cursor)
    try:
        items=[next(iterator) for _ in range(count)]
    finally:
        iterator.close()
    restarted=binding.route.iter_from(items[0].cursor_after)
    try:
        second=next(restarted)
    finally:
        restarted.close()
    if (not items[1].batch.native_rows.equals(second.batch.native_rows,check_metadata=True)
            or items[1].cursor_after!=second.cursor_after):
        raise ValueError("Frozen Reactome native cursor recovery changed")
    return items,[_record(w,source,binding) for w in items]


def _original_parquet_signatures(root, view, paths):
    """Independent original Parquet scan, never through the campaign route/adapter."""
    import pyarrow.parquet as pq
    signatures=[]
    for path in paths:
        descriptor=json.loads(view._sealed_by_path[path])
        original=root/path
        if full9(original)!=tuple(descriptor["source_full9"]):
            raise ValueError("Original Reactome source shard changed before direct scan")
        codes=decode_disposition(descriptor,root=root)
        absolute=0
        parquet=pq.ParquetFile(original)
        for rb in parquet.iter_batches(batch_size=64,use_threads=False):
            for row in rb.to_pylist():
                if (codes[absolute]==0 and row["source_family"]==FAMILY
                        and row["task_name"]==TASK and row["native_schema"]==SCHEMA
                        and row["source_revision"]=="Reactome98" and row["split"]=="train"):
                    raw=row["native_payload"]
                    if type(raw) is not bytes or h(raw)!=row["native_payload_sha256"]:
                        raise ValueError("Original Reactome Parquet payload SHA changed")
                    signatures.append((path,absolute,h(canonical([row["task_id"],row["native_payload_sha256"]]))))
                absolute+=1
        if absolute!=len(codes) or full9(original)!=tuple(descriptor["source_full9"]):
            raise ValueError("Original Reactome Parquet source changed during direct scan")
    return signatures


def audit_reactome_complete(root):
    """Full native common route exactly once; no optimizer, GPU or new training slot."""
    root=Path(root).resolve()
    context=load_inactive_common_source_rebind(root)
    frozen=document(root,context.evidence)
    match=[row for row in frozen["route_bindings"] if row["task_key"]==KEY]
    if (len(match)!=1 or len(frozen["route_bindings"])!=62
            or frozen["training_authorized"] is not False
            or frozen["global_split_accepted"] is not False):
        raise ValueError("Exact inactive frozen 62-route source evidence required")
    row,=match
    if (frozen["inactive_candidate_snapshot"]["sha256"]!=CANDIDATE
            or frozen["common_source_snapshot"]["sha256"]!=COMMON
            or row["common_view_policy_sha256"]!=VIEW
            or row["candidate"]["max_windows"]!=2 or row["common"]["max_windows"]!=2):
        raise ValueError("Frozen native Reactome candidate/common source family differs")
    view,sources,availability,_,_=load_original_annotation_assets(root)
    source=sources["reactome"]
    if (finite_view_policy_sha256(view)!=VIEW
            or availability[KEY]["train_rows"]!=1653
            or availability[KEY]["validation_rows"]!=0 or availability[KEY]["test_rows"]!=0):
        raise ValueError("Full original train-only Reactome census/source policy differs")
    bound={}
    code_root=Path(__file__).resolve().parents[3]
    for label,snapshot in (("candidate",CANDIDATE),("common",COMMON)):
        b=bind_reactome(source,view,snapshot_sha256=snapshot,batch_size=1)
        observed=dict(task_key=KEY,max_windows=2,
                      adapter_sha256=executable_adapter_sha256(b.prepare,root=code_root),
                      initial_cursor_sha256=h(b.route.initial_cursor))
        if observed!=row[label] or b.route.source_snapshot_sha256!=snapshot:
            raise ValueError("Live Reactome executable/cursor differs from frozen "+label)
        bound[label]=b
    candidate_items,candidate_signatures=_frozen_prefix(bound["candidate"],source)
    common_items,common_prefix=_frozen_prefix(bound["common"],source)
    if (candidate_signatures!=common_prefix or any(
            not x.batch.native_rows.equals(y.batch.native_rows,check_metadata=True)
            for x,y in zip(candidate_items,common_items,strict=True))):
        raise ValueError("Exact original Reactome candidate/common prefix rows differ")
    common=bound["common"]
    iterator=common.route.iter_from(common.route.initial_cursor)
    signatures=[]
    reference=[]
    role_positions=0
    stoichiometry_positions=0
    unique=set()
    sampled_cursor_recovery=0
    previous=None
    try:
        for window in iterator:
            got=_record(window,source,common)
            if got[:2] in unique:
                raise ValueError("Duplicate original Reactome native source position")
            unique.add(got[:2])
            signatures.append(got[:3])
            reference.append(got)
            role_positions+=got[4][0]
            stoichiometry_positions+=got[4][1]
            n=len(signatures)
            if n in (2,826,1652):
                iterator_after=common.route.iter_from(window.cursor_after)
                try:
                    successor=next(iterator_after)
                except StopIteration as e:
                    raise ValueError("Native Reactome cursor prematurely exhausted") from e
                finally:
                    iterator_after.close()
                previous=window.cursor_after,successor
            if previous is not None and n in (3,827,1653):
                if (previous[1].cursor_after!=window.cursor_after
                        or not previous[1].batch.native_rows.equals(window.batch.native_rows,check_metadata=True)):
                    raise ValueError("Native Reactome common cursor replay changed")
                sampled_cursor_recovery+=1
                previous=None
    finally:
        iterator.close()
    if len(signatures)!=1653 or len(set(signatures))!=1653 or sampled_cursor_recovery!=3:
        raise ValueError("Full Reactome common source count/cursor exhaustion differs")
    direct=_original_parquet_signatures(root,view,tuple(common.route._paths))
    if signatures!=direct:
        raise ValueError("Complete Reactome source route differs from independently scanned Parquet")
    if reference[:2]!=common_prefix:
        raise ValueError("Full Reactome common reader differs from frozen two-window prefix")
    return dict(
        schema="Mora3.Campaign26ReactomeCompleteCommonOriginal/v1",
        frozen_62_route_metadata_only=True,
        native_source_exhausted=True,
        original_common_native_train_rows=len(signatures),
        original_validation_rows=0,original_test_rows=0,
        independent_original_parquet_equal=True,
        candidate_prefix_original_rows_equal=True,
        candidate_and_common_frozen=row,
        source_snapshot_sha256=COMMON,
        source_view_policy_sha256=VIEW,
        native_masked_role_positions=role_positions,
        native_supplied_stoichiometry_positions=stoichiometry_positions,
        original_signature_chain_sha256=h(canonical(signatures)),
        source_only_no_objective_call=True,
        original_cursor_recoveries=sampled_cursor_recovery,
        native_evidence_semantics="Reactome98 static pathway masked participants; source-known incidence excludes masked role and stoichiometry",
        candidate_common_full_replay=False,
        complete_62_source_family_admitted=False,global_biological_split_admitted=False,
        training_admission=False,optimizer_updates=0,counted_proxy_trials=0,
    )


if __name__=="__main__":
    import sys
    if len(sys.argv)!=3:raise SystemExit("Usage: python -m mora3.research.campaign26_reactome_complete_common ACTIVE_ROOT NEW_AUDIT_JSON")
    if os.uname().nodename!="iska":raise ValueError("Authorized Iska workstation required")
    output=Path(sys.argv[2])
    if output.exists():raise FileExistsError("No overwritten evidence allowed")
    result=audit_reactome_complete(sys.argv[1])
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x") as f:
        json.dump(result,f,indent=2,sort_keys=True);f.write("\n")
    print(json.dumps({"sha256":h(output.read_bytes()),"rows":result["original_common_native_train_rows"],
                      "masked_role_positions":result["native_masked_role_positions"],
                      "supplied_stoichiometry_positions":result["native_supplied_stoichiometry_positions"]},sort_keys=True))
