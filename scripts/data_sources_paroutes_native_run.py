"""Finite complete PaRoutes data-format/chemistry QC, immutable original row refs."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
import csv,gzip,hashlib,json,os,resource,sys,time
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import rdBase
from rdkit.Chem import rdChemReactions
sys.path.insert(0,str(Path(__file__).resolve().parent))
from data_sources_paroutes_native_qc import iter_json_array,check_route,check_reaction,smiles_parse,file_role
from data_sources_ord_native_qc import sha,canonical,feed,verify_typed,guard,write_new

SCHEMA=pa.schema([("source_file_sha256",pa.string()),("source_row",pa.int64()),("source_role",pa.string()),
                 ("payload_json",pa.string()),("quality_reasons_json",pa.string()),("source_reference_hash_exposure",pa.bool_()),
                 ("task_admitted",pa.bool_()),("adapter_supported",pa.bool_())])

def process_file(asset,output,reference_hashes,free_floor_bytes=64*1024**3,deadline=None,progress=None):
    p=Path(asset["path"]);out=Path(output)
    guard(free_floor_bytes,deadline)
    if p.stat().st_size!=asset["bytes"] or sha(p)!=asset["sha256"]:
        raise ValueError("original source integrity mismatch")
    if out.exists():raise FileExistsError(out)
    n=0;counts=Counter();why_counts=Counter();role=file_role(p.name);digest=hashlib.sha256();headers=None
    start=time.monotonic();gz=p.name.endswith(".gz");stem=p.name[:-3] if gz else p.name
    kind="TSV" if stem.endswith(".csv") else "JSON_ARRAY" if stem.endswith(".json") else "TEXT_LINES"
    opener=gzip.open if gz else open
    writer=pq.ParquetWriter(out,SCHEMA,compression="zstd")
    try:
        with opener(p,"rt",encoding="utf-8",newline="") as f:
            if kind=="TSV":
                csv.field_size_limit(32*1024*1024);reader=csv.reader(f,delimiter="\t");headers=next(reader)
                if not headers or len(set(headers))!=len(headers):raise ValueError("native TSV empty/duplicate header")
            elif kind=="JSON_ARRAY":reader=iter_json_array(f)
            else:reader=(line.rstrip("\r\n") for line in f)
            batch=[]
            for payload in reader:
                if n%256==0:guard(free_floor_bytes,deadline)
                why=[];exposure=None
                if kind=="TSV":
                    if len(payload)!=len(headers):why.append("native_TSV_width_mismatch");d={}
                    else:d=dict(zip(headers,payload))
                    rtext=d.get("RxnSmilesClean",d.get("rsmi_processed",d.get("reaction_smiles")))
                    if rtext is not None and not check_reaction(rtext)["parse_pass"]:
                        why.append("native_reaction_SMILES_parse_failed")
                    template=d.get("retro_template")
                    if template is not None:
                        try:valid=rdChemReactions.ReactionFromSmarts(template) is not None
                        except Exception:valid=False
                        if not valid:why.append("native_computed_template_SMARTS_parse_failed")
                    h=d.get("PseudoHash",d.get("reaction_hash"))
                    if h is not None:exposure=h in reference_hashes
                elif kind=="JSON_ARRAY":
                    if "routes" in stem:
                        d=check_route(payload);why.extend(d["quality_reasons"])
                        counts["source_molecule_nodes"]+=d["molecule_nodes"];counts["source_reaction_nodes"]+=d["reaction_nodes"]
                        exposure=any(h in reference_hashes for h in d["source_reaction_hashes"])
                    elif stem=="uspto_lookup.json":
                        if not isinstance(payload,dict):why.append("native_lookup_item_not_object")
                        else:
                            t=payload.get("reaction_smarts")
                            try:valid=rdChemReactions.ReactionFromSmarts(t) is not None
                            except Exception:valid=False
                            if not valid:why.append("native_computed_template_SMARTS_parse_failed")
                else:
                    if stem.startswith(("stock_","targets_")) and not smiles_parse(payload):why.append("native_molecule_SMILES_parse_failed")
                    if not payload:why.append("empty_source_text_line")
                row={"source_file_sha256":asset["sha256"],"source_row":n,"source_role":role,
                     "payload_json":canonical(payload),"quality_reasons_json":canonical(why),
                     "source_reference_hash_exposure":exposure,"task_admitted":False,"adapter_supported":False}
                feed(digest,row);batch.append(row);n+=1
                counts["quality_pass_rows" if not why else "quarantined_rows"]+=1
                if exposure:counts["source_reference_hash_exposure_rows"]+=1
                why_counts.update(why)
                if len(batch)>=256:
                    writer.write_table(pa.Table.from_pylist(batch,schema=SCHEMA));batch=[]
                if progress and n%4096==0:progress(n)
            if batch:writer.write_table(pa.Table.from_pylist(batch,schema=SCHEMA))
    finally:writer.close()
    readback=verify_typed(out,digest.hexdigest(),n)
    if sha(p)!=asset["sha256"]:raise ValueError("source changed during native QC")
    return {"source_path":str(p),"source_bytes":asset["bytes"],"source_sha256":asset["sha256"],"source_rows":n,
            "source_native_format":kind,"native_TSV_ordered_columns":headers,"source_role":role,
            "quality_pass_rows":counts["quality_pass_rows"],"quarantined_rows":counts["quarantined_rows"],
            "quality_reason_counts":dict(why_counts),"source_reference_hash_exposure_rows":counts["source_reference_hash_exposure_rows"],
            "source_molecule_nodes":counts["source_molecule_nodes"],"source_reaction_nodes":counts["source_reaction_nodes"],
            "gzip_full_read_CRC_pass":True if gz else None,"full_typed_readback_rows":readback,
            "typed_path":str(out),"typed_bytes":out.stat().st_size,"typed_sha256":sha(out),"semantic_rows_digest":digest.hexdigest(),
            "task_admitted_rows":0,"adapter_supported":False,"RDKit_version":rdBase.rdkitVersion,
            "elapsed_seconds":time.monotonic()-start,
            "limits":["all source rows/cells/tree attributes retained even when quarantined","format/graph parsing does not prove synthetic feasibility or measured yield",
                      "source atom maps/templates/classes are computed or source labels, not unknown-reaction predictor inputs",
                      "reference hash membership is an exposure audit only; nonmembership is not full leakage-safety proof",
                      "n1/n5 routes,targets,stocks/reference hashes are evaluation/exposure controls, never training"]}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--plan",required=True);a=ap.parse_args()
    pp=Path(a.plan);plan=json.loads(pp.read_text())
    for p,expected in plan["code_and_input_bindings"].items():
        if sha(p)!=expected:raise ValueError("pinned code/input mismatch: "+p)
    os.sched_setaffinity(0,{plan["cpu"]});os.nice(10)
    resource.setrlimit(resource.RLIMIT_AS,(plan["virtual_address_bytes"],plan["virtual_address_bytes"]))
    deadline=time.monotonic()+plan["deadline_seconds"]
    out=Path(plan["output_directory"]);out.mkdir(exist_ok=False,parents=True)
    guard(plan["free_space_floor_bytes"],deadline)
    assets=plan["assets"]
    hashes_asset=next(x for x in assets if Path(x["path"]).name=="ref_routes_reaction_hashes.txt")
    if sha(hashes_asset["path"])!=hashes_asset["sha256"]:raise ValueError("reference hashes source integrity mismatch")
    hashes=set(Path(hashes_asset["path"]).read_text().splitlines())
    write_new(out/"launch.json",{"utc":datetime.now(timezone.utc).isoformat(),"pid":os.getpid(),"start_ticks":Path("/proc/self/stat").read_text().split()[21],
                "sid":os.getsid(0),"pgid":os.getpgid(0),"argv":sys.argv,"cpu_affinity":sorted(os.sched_getaffinity(0)),
                "plan_sha256":sha(pp),"reference_hash_distinct":len(hashes),"state":"full native QC active; no admission"})
    refs=[];t=time.monotonic()
    try:
        for index,asset in enumerate(assets):
            name=Path(asset["path"]).name
            def progress(n):print(canonical({"file":name,"asset_number":index+1,"rows":n}),flush=True)
            r=process_file(asset,out/(name+".typed.parquet"),hashes,free_floor_bytes=plan["free_space_floor_bytes"],deadline=deadline,progress=progress)
            rp=out/(name+".receipt.json");write_new(rp,r)
            refs.append({"receipt_path":str(rp),"receipt_sha256":sha(rp),"source_rows":r["source_rows"],
                         "quality_pass_rows":r["quality_pass_rows"],"quarantined_rows":r["quarantined_rows"],
                         "typed_bytes":r["typed_bytes"],"typed_path":r["typed_path"],"typed_sha256":r["typed_sha256"],"source_role":r["source_role"]})
            print(canonical({"completed_file":name,"source_rows":r["source_rows"],"quarantines":r["quarantined_rows"]}),flush=True)
        write_new(out/"manifest.json",{"completed_utc":datetime.now(timezone.utc).isoformat(),"plan_path":str(pp),"plan_sha256":sha(pp),
                   "source_files":len(refs),"source_rows_nonadditive":sum(x["source_rows"] for x in refs),
                   "typed_bytes":sum(x["typed_bytes"] for x in refs),"assets":refs,"task_admitted_rows":0,"adapter_supported":False,
                   "source_license":"CC-BY-4.0; original Lowe USPTO CC0","elapsed_seconds":time.monotonic()-t,
                   "counts_nonadditive":"reaction rows/template references/route node occurrences and heldout controls overlap; never sum as independent reactions"})
    except BaseException as exc:
        write_new(out/"failure.json",{"utc":datetime.now(timezone.utc).isoformat(),"error":type(exc).__name__+":"+str(exc),"completed_assets":len(refs),"all_partials_retained":True});raise

if __name__=="__main__":main()
