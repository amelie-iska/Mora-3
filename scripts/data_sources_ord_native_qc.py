"""Finite, loss-aware ORD native source QC; no corpus task admission."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import pyarrow as pa
import pyarrow.parquet as pq
sys.path.insert(0,str(Path(__file__).resolve().parent))
from data_sources_ord_static_proto import load_reaction_class, decode_reaction

SCHEMA=pa.schema([
    ("dataset_id",pa.string()),("source_file_sha256",pa.string()),("source_row",pa.int64()),
    ("reaction_id",pa.string()),("source_proto_sha256",pa.string()),("source_proto_bytes",pa.int64()),
    ("semantic_json",pa.string()),("quality_reasons_json",pa.string()),("unknown_wire_fields",pa.bool_()),
    ("semantic_json_roundtrip_equal",pa.bool_()),("task_admitted",pa.bool_()),("adapter_supported",pa.bool_())])

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)

def feed(h,row):
    b=canonical(row).encode();h.update(len(b).to_bytes(8,"big"));h.update(b)

def verify_typed(path,expected_digest,expected_rows):
    h=hashlib.sha256();n=0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256,use_threads=False):
        for row in batch.to_pylist():
            feed(h,row);n+=1
    if h.hexdigest()!=expected_digest or n!=expected_rows:
        raise ValueError("typed source conservation digest/row mismatch")
    return n

def guard(free_floor_bytes,deadline):
    v=os.statvfs(".")
    if v.f_bavail*v.f_frsize < free_floor_bytes:
        raise RuntimeError("hardware free-space floor reached")
    if deadline is not None and time.monotonic()>deadline:
        raise RuntimeError("finite QC deadline reached")

def collect_annotations(obj,counters):
    if isinstance(obj,dict):
        for key,value in obj.items():
            if key in ("reaction_role","units") and isinstance(value,(str,int)):
                counters[key][str(value)]+=1
            if key=="identifiers" and isinstance(value,list):
                for ident in value:
                    if isinstance(ident,dict):counters["identifier_type"][str(ident.get("type","UNSPECIFIED_ABSENT"))]+=1
            collect_annotations(value,counters)
    elif isinstance(obj,list):
        for value in obj:collect_annotations(value,counters)

def process_asset(asset,reaction_class,output,batch_size=256,free_floor_bytes=64*1024**3,deadline=None,progress=None):
    path=Path(asset["path"]);output=Path(output)
    guard(free_floor_bytes,deadline)
    if path.stat().st_size!=asset["bytes"] or sha(path)!=asset["sha256"]:
        raise ValueError("source SHA/size mismatch")
    if output.exists():raise FileExistsError(output)
    native=pq.ParquetFile(path)
    if native.schema_arrow.names!=["reaction_id","reaction"] or not pa.types.is_string(native.schema_arrow.field("reaction_id").type) or not pa.types.is_binary(native.schema_arrow.field("reaction").type):
        raise ValueError("unexpected native ORD Parquet row schema")
    metadata={k.decode():v.decode() for k,v in (native.schema_arrow.metadata or {}).items()}
    counts=Counter();reasons=Counter();fields=Counter()
    counters={k:Counter() for k in ["reaction_role","units","identifier_type"]}
    n=0;digest=hashlib.sha256();started=time.monotonic()
    writer=pq.ParquetWriter(output,SCHEMA,compression="zstd")
    try:
        for batch in native.iter_batches(batch_size=batch_size,use_threads=False):
            guard(free_floor_bytes,deadline);typed=[]
            for raw in batch.to_pylist():
                body=raw["reaction"];rid=raw["reaction_id"]
                try:
                    d=decode_reaction(reaction_class,body,rid)
                    semantic=canonical(d["semantic"])
                    why=d["quality_reasons"]
                    fields.update(d["semantic"].keys())
                    collect_annotations(d["semantic"],counters)
                    unknown=d["unknown_wire_fields"];equal=d["semantic_json_roundtrip_equal"]
                except ValueError as exc:
                    semantic=None;why=["native_proto_refused:"+str(exc)];unknown=None;equal=False
                row={"dataset_id":asset["id"],"source_file_sha256":asset["sha256"],"source_row":n,
                     "reaction_id":rid,"source_proto_sha256":hashlib.sha256(body).hexdigest() if isinstance(body,bytes) else None,
                     "source_proto_bytes":len(body) if isinstance(body,bytes) else None,"semantic_json":semantic,
                     "quality_reasons_json":canonical(why),"unknown_wire_fields":unknown,
                     "semantic_json_roundtrip_equal":equal,"task_admitted":False,"adapter_supported":False}
                feed(digest,row);typed.append(row);n+=1
                counts["quarantined_rows" if why else "native_semantic_quality_pass"]+=1
                reasons.update(why)
            writer.write_table(pa.Table.from_pylist(typed,schema=SCHEMA))
            if progress:progress(n,native.metadata.num_rows)
    finally:writer.close()
    if n!=native.metadata.num_rows:raise ValueError("native footer/scan count mismatch")
    readback=verify_typed(output,digest.hexdigest(),n)
    if sha(path)!=asset["sha256"]:raise ValueError("source SHA changed during QC")
    return {
        "source_id":asset["id"],"source_path":str(path),"source_bytes":asset["bytes"],"source_sha256":asset["sha256"],
        "source_metadata":metadata,"source_schema":str(native.schema_arrow),"source_rows":n,
        "native_semantic_quality_pass":counts["native_semantic_quality_pass"],"quarantined_rows":counts["quarantined_rows"],
        "quality_reason_counts":dict(reasons),"native_root_field_presence_rows":dict(fields),
        "native_role_unit_identifier_occurrences":{k:dict(v) for k,v in counters.items()},
        "task_admitted_rows":0,"adapter_supported":False,"typed_path":str(output),"typed_bytes":output.stat().st_size,
        "typed_sha256":sha(output),"semantic_rows_digest":digest.hexdigest(),"full_typed_readback_rows":readback,
        "elapsed_seconds":time.monotonic()-started,
        "limits":["native binary file+row remains authoritative","proto3 scalar default presence may be unresolvable",
                  "format/semantic roundtrip QC is not chemical validation, original-study verification or corpus admission",
                  "original roles/maps/yields/units/citations retained; no gold class or target-derived map predictor admission"]}

def write_new(path,obj):
    with Path(path).open("x") as f:json.dump(obj,f,indent=2,sort_keys=True);f.write("\n")

def main():
    args=argparse.ArgumentParser();args.add_argument("--plan",required=True);opts=args.parse_args()
    plan_path=Path(opts.plan);plan=json.loads(plan_path.read_text())
    for path,expected in plan["code_and_input_bindings"].items():
        if sha(path)!=expected:raise ValueError("pinned code/input mismatch: "+path)
    out=Path(plan["output_directory"]);out.mkdir(exist_ok=False,parents=True)
    os.sched_setaffinity(0,{plan["cpu"]});os.nice(10)
    resource.setrlimit(resource.RLIMIT_AS,(plan["virtual_address_bytes"],plan["virtual_address_bytes"]))
    deadline=time.monotonic()+plan["deadline_seconds"]
    avail=json.loads(Path(plan["available_manifest"]).read_text())
    C=load_reaction_class(plan["reaction_pb2_source"],plan["reaction_pb2_sha256"])
    launch={"created_utc":datetime.now(timezone.utc).isoformat(),"pid":os.getpid(),"sid":os.getsid(0),"pgid":os.getpgid(0),
            "argv":sys.argv,"cpu_affinity":sorted(os.sched_getaffinity(0)),"plan_sha256":sha(plan_path),
            "source_assets":len(avail["assets"]),"state":"native semantic QC active; no admission"}
    write_new(out/"launch.json",launch)
    receipts=[];all_start=time.monotonic()
    try:
        for index,asset in enumerate(avail["assets"]):
            def progress(n,total):
                if n%4096<256:
                    print(canonical({"asset":asset["id"],"asset_number":index+1,"rows":n,"footer_rows":total}),flush=True)
            receipt=process_asset(asset,C,out/(asset["id"]+".typed.parquet"),free_floor_bytes=plan["free_space_floor_bytes"],
                                  deadline=deadline,progress=progress)
            rp=out/(asset["id"]+".receipt.json");write_new(rp,receipt)
            receipts.append({"receipt_path":str(rp),"receipt_sha256":sha(rp),"source_rows":receipt["source_rows"],
                             "native_semantic_quality_pass":receipt["native_semantic_quality_pass"],"quarantined_rows":receipt["quarantined_rows"],
                             "typed_bytes":receipt["typed_bytes"],"typed_path":receipt["typed_path"],"typed_sha256":receipt["typed_sha256"]})
            print(canonical({"completed_asset":index+1,"source_rows_total":sum(x["source_rows"] for x in receipts)}),flush=True)
        manifest={"completed_utc":datetime.now(timezone.utc).isoformat(),"plan_path":str(plan_path),"plan_sha256":sha(plan_path),
                  "original_available_manifest":plan["available_manifest"],"original_available_manifest_sha256":sha(plan["available_manifest"]),
                  "source_files":len(receipts),"source_rows":sum(x["source_rows"] for x in receipts),
                  "native_semantic_quality_pass":sum(x["native_semantic_quality_pass"] for x in receipts),
                  "quarantined_rows":sum(x["quarantined_rows"] for x in receipts),"typed_bytes":sum(x["typed_bytes"] for x in receipts),
                  "task_admitted_rows":0,"adapter_supported":False,"assets":receipts,"elapsed_seconds":time.monotonic()-all_start,
                  "data_license":"CC-BY-SA-4.0","source_original_and_derived_roles":"ORD contributed/native patent-mined and measured context kept distinct; no independent experiment assumption"}
        write_new(out/"manifest.json",manifest)
    except BaseException as exc:
        write_new(out/"failure.json",{"utc":datetime.now(timezone.utc).isoformat(),"error":type(exc).__name__+":"+str(exc),"completed_assets":len(receipts),"all_partials_retained":True})
        raise

if __name__=="__main__":main()
