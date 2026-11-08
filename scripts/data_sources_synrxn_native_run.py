"""Finite whole native SynRXN synthesis protocol-table QC; no corpus admission."""
from __future__ import annotations
import argparse,csv,gzip,json,os,resource,sys,time
import hashlib
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit.Chem import rdChemReactions
from rdkit import rdBase
sys.path.insert(0,str(Path(__file__).resolve().parent))
from data_sources_synrxn_native import project_native_row
from data_sources_paroutes_native_qc import check_reaction
from data_sources_ord_native_qc import sha,canonical,feed,verify_typed,guard,write_new
SCHEMA=pa.schema([(x,pa.string()) for x in [
"source_file_sha256","reaction_id","native_columns_json","native_cells_json","native_split","source_protocol_role",
"gold_target_columns_json","default_predictor_columns_json","quality_reasons_json"]]+[
("source_row",pa.int64()),("task_admitted",pa.bool_()),("adapter_supported",pa.bool_()),("model_use_allowed",pa.bool_()),
("strict_molecule_reaction_parse_pass",pa.bool_()),("query_reaction_parse_pass",pa.bool_())])

def process_asset(asset,output,free_floor_bytes=64*1024**3,deadline=None):
    p=Path(asset["path"]);out=Path(output);guard(free_floor_bytes,deadline)
    if p.stat().st_size!=asset["bytes"] or sha(p)!=asset["sha256"]:raise ValueError("original source integrity mismatch")
    if out.exists():raise FileExistsError(out)
    writer=pq.ParquetWriter(out,SCHEMA,compression="zstd")
    n=0;seen=set();dups=0;split=Counter();roles=Counter();chem=Counter();reasons=Counter();h=hashlib.sha256();started=time.monotonic()
    try:
        with gzip.open(p,"rt",encoding="utf-8",newline="") as f:
            csv.field_size_limit(32*1024**2);rows=csv.reader(f);columns=next(rows)
            if columns!=asset["columns"]:raise ValueError("pinned native column schema mismatch")
            batch=[]
            for cells in rows:
                if n%256==0:guard(free_floor_bytes,deadline)
                row=project_native_row(columns,cells,asset["sha256"],n);d=dict(zip(columns,cells));rid=d.get("r_id")
                if rid in seen:dups+=1
                seen.add(rid)
                text=d.get("aam",d.get("rxn"))
                strict=check_reaction(text)["parse_pass"]
                try:query=rdChemReactions.ReactionFromSmarts(text) is not None if text else False
                except Exception:query=False
                row.update(strict_molecule_reaction_parse_pass=strict,query_reaction_parse_pass=query)
                chem[str((strict,query))]+=1
                split[d.get("split")]+=1;roles[row["source_protocol_role"]]+=1
                reasons.update(json.loads(row["quality_reasons_json"]))
                feed(h,row);batch.append(row);n+=1
                if len(batch)>=256:writer.write_table(pa.Table.from_pylist(batch,schema=SCHEMA));batch=[]
                if n%8192==0:print(canonical({"file":p.name,"rows":n}),flush=True)
            if batch:writer.write_table(pa.Table.from_pylist(batch,schema=SCHEMA))
    finally:writer.close()
    if n!=asset["expected_rows"]:raise ValueError("declared source row count mismatch; partial output retained, not promoted")
    readback=verify_typed(out,h.hexdigest(),n)
    if sha(p)!=asset["sha256"]:raise ValueError("original changed during QC")
    return {"source_path":str(p),"source_bytes":asset["bytes"],"source_sha256":asset["sha256"],"source_rows":n,
"native_ordered_columns":columns,"native_split_counts":dict(split),"source_protocol_role_counts":dict(roles),
"native_ID_unique":len(seen),"duplicate_native_ID_occurrences":dups,"quality_reason_occurrences":dict(reasons),
"parser_acceptance_counts_strict_molecule_vs_query":dict(chem),"parser_count_limits":"parser acceptance only, not a chemical feasibility verdict or a source-format relabeling",
"gzip_full_read_CRC_passed":True,"typed_path":str(out),"typed_bytes":out.stat().st_size,"typed_sha256":sha(out),
"full_typed_readback_rows":readback,"semantic_rows_digest":h.hexdigest(),"task_admitted_rows":0,"adapter_supported":False,
"model_use_allowed":False,"RDKit_version":rdBase.rdkitVersion,"elapsed_seconds":time.monotonic()-started,
"default_predictor_columns":"none; constructor must explicitly authorize query fields; source maps/rc/reagent targets and IDs stay target/audit roles",
"source_count_semantics":"overlapping curated protocol view of existing USPTO patent family, not independent new experimental observations"}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--plan",required=True);o=ap.parse_args();pp=Path(o.plan);p=json.loads(pp.read_text())
    for path,expected in p["bindings"].items():
        if sha(path)!=expected:raise ValueError("pinned input/code hash mismatch: "+path)
    os.sched_setaffinity(0,{p["cpu"]});os.nice(10);resource.setrlimit(resource.RLIMIT_AS,(p["virtual_address_bytes"],p["virtual_address_bytes"]))
    deadline=time.monotonic()+p["deadline_seconds"];out=Path(p["output"]);out.mkdir(exist_ok=False,parents=True)
    write_new(out/"launch.json",{"utc":datetime.now(timezone.utc).isoformat(),"pid":os.getpid(),"sid":os.getsid(0),"pgid":os.getpgid(0),
                 "argv":sys.argv,"plan_sha256":sha(pp),"cpu":sorted(os.sched_getaffinity(0))})
    refs=[]
    try:
        for a in p["assets"]:
            r=process_asset(a,out/(Path(a["path"]).name+".typed.parquet"),free_floor_bytes=p["free_floor_bytes"],deadline=deadline)
            rp=out/(Path(a["path"]).name+".receipt.json");write_new(rp,r);refs.append({"receipt":str(rp),"receipt_sha256":sha(rp),**{k:r[k] for k in ["source_rows","typed_path","typed_bytes","typed_sha256","native_split_counts","source_protocol_role_counts"]}})
        write_new(out/"manifest.json",{"completed_utc":datetime.now(timezone.utc).isoformat(),"plan_path":str(pp),"plan_sha256":sha(pp),"files":refs,
                  "source_files":len(refs),"native_record_views_nonadditive":sum(x["source_rows"] for x in refs),"task_admitted":0,"adapter_supported":False,"model_use_allowed":False,
                  "rights":"exact curated tables CC-BY4; original Lowe patent data CC0","all_native_splits_preserved":True,"all_validation_test_EVAL_exposure_only":True})
    except BaseException as exc:
        write_new(out/"failure.json",{"utc":datetime.now(timezone.utc).isoformat(),"error":type(exc).__name__+":"+str(exc),"completed_assets":len(refs),"all_partials_retained":True});raise

if __name__=="__main__":main()
