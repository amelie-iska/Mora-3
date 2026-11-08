"""Leased finite source-range download with immutable prefix/pieces and separate final file."""
import argparse
import concurrent.futures
import datetime
import gzip
import hashlib
import json
import os
import pathlib
import re
import shutil
import threading
import time
import urllib.request
from data_sources_fetch import digest
from data_sources_fetch_fast import asset_lease,is_retryable,retry_delay


def plan_ranges(start,total,size):
    if not 0<=start<=total or size<=0:raise ValueError("invalid range plan")
    return [(i,min(i+size,total)-1) for i in range(start,total,size)]


def validate_range_response(asset,start,end,status,headers):
    hdr={k.lower():v for k,v in headers.items()}
    if status!=206:raise ValueError("checked source range requires206")
    if hdr.get("content-range")!="bytes %d-%d/%d"%(start,end,asset["expected_bytes"]):raise ValueError("source range mismatch")
    if asset.get("expected_etag") and hdr.get("etag")!=asset["expected_etag"]:raise ValueError("source range entity ETag mismatch")
    if hdr.get("content-length") and int(hdr["content-length"])!=end-start+1:raise ValueError("source range length mismatch")
    if hdr.get("content-encoding","") not in ("","identity"):raise ValueError("unexpected HTTP content encoding")


def assemble_asset(asset,root,pieces):
    root=pathlib.Path(root);prefix=asset.get("prefix");offset=0;ordered=sorted(pieces,key=lambda p:p["start"])
    if prefix:
        p=pathlib.Path(prefix["path"])
        if p.stat().st_size!=prefix["bytes"] or digest(p)!=prefix["sha256"]:raise ValueError("preserved prefix integrity failure")
        offset=prefix["bytes"]
    for piece in ordered:
        p=pathlib.Path(piece["path"])
        if piece["start"]!=offset or p.stat().st_size!=piece["end"]-piece["start"]+1 or digest(p)!=piece["sha256"]:raise ValueError("source piece integrity/coverage failure")
        offset=piece["end"]+1
    if offset!=asset["expected_bytes"]:raise ValueError("source range suffix incomplete")
    dest=root/asset["destination_path"];dest.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();md5=hashlib.md5();n=0
    paths=([pathlib.Path(prefix["path"])] if prefix else [])+[pathlib.Path(p["path"]) for p in ordered]
    with dest.open("xb") as out:
        for p in paths:
            with p.open("rb") as source:
                for chunk in iter(lambda:source.read(2**20),b""):
                    out.write(chunk);h.update(chunk);md5.update(chunk);n+=len(chunk)
        out.flush();os.fsync(out.fileno())
    if n!=asset["expected_bytes"] or asset.get("expected_sha256") and h.hexdigest()!=asset["expected_sha256"]:raise ValueError("assembled source byte/checksum failure")
    if asset.get("expected_md5") and md5.hexdigest()!=asset["expected_md5"]:raise ValueError("upstream MD5 mismatch; raw pieces and assembled quarantine retained")
    return {**asset,"path":str(dest.resolve()),"bytes":n,"sha256":h.hexdigest(),"md5":md5.hexdigest(),"upstream_md5_passed":True if asset.get("expected_md5") else None,"prefix_readonly_reused":bool(prefix),"retained_piece_count":len(ordered)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--plan",required=True);args=parser.parse_args();planpath=pathlib.Path(args.plan).resolve();root=planpath.parent;plan=json.loads(planpath.read_text())
    if not plan.get("prior_writer_retired"):raise ValueError("no exclusive prior-writer retirement checkpoint")
    if plan.get("old_pid") and pathlib.Path("/proc",str(plan["old_pid"])).exists():raise ValueError("old writer PID still exists; refuse ownership")
    for a in plan["assets"]:
        for key in ("destination_path","piece_directory"):
            p=pathlib.Path(a[key])
            if p.is_absolute() or ".." in p.parts:raise ValueError("new destination path escape")
    name=plan.get("job_name","segmented-string-v1");lock=threading.Lock();starttime=time.monotonic();done=[];failed=[];state={};initial={};current_asset=None
    live=root/("live-acquisition-"+name+".json");receipts=root/"audits"/("asset-receipts-"+name);receipts.mkdir(exist_ok=True)
    def status(phase):
        elapsed=max(time.monotonic()-starttime,0.001);transferred=sum(max(state.get(k,0)-initial.get(k,0),0) for k in state)
        live.write_text(json.dumps(dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),phase=phase,current_asset=current_asset,workers=plan["workers"],segment_bytes=plan["segment_bytes"],piece_bytes=state,verified_asset_ids=done,failed_asset_ids=failed,elapsed_seconds=elapsed,transferred_bytes=transferred,measured_bytes_per_second=transferred/elapsed,free_bytes=shutil.disk_usage(root).free),indent=2)+"\n")
    def fetch_piece(asset,bounds):
        begin,end=bounds;pid=asset["id"]+"-%012d-%012d"%(begin,end);p=root/asset["piece_directory"]/(pid+".bin");p.parent.mkdir(parents=True,exist_ok=True);receipt=p.with_suffix(".receipt.json")
        with asset_lease(p.with_suffix(".lease")):
            if receipt.exists():
                d=json.loads(receipt.read_text())
                if p.stat().st_size!=end-begin+1 or digest(p)!=d["sha256"]:raise ValueError("immutable source piece changed")
                return d
            offset=p.stat().st_size if p.exists() else 0
            if offset>end-begin+1:raise ValueError("oversized pending piece")
            with lock:initial[pid]=offset;state[pid]=offset;status("downloading_checked_ranges")
            for attempt in range(plan.get("max_attempts_per_piece",4)):
                try:
                    offset=p.stat().st_size if p.exists() else 0
                    if offset<end-begin+1:
                        headers={"Accept-Encoding":"identity","User-Agent":"Mora3-source-range-intake/1.0","Range":"bytes=%d-%d"%(begin+offset,end)}
                        if asset.get("expected_etag"):headers["If-Range"]=asset["expected_etag"]
                        with urllib.request.urlopen(urllib.request.Request(asset["url"],headers=headers),timeout=120) as response:
                            validate_range_response(asset,begin+offset,end,response.status,dict(response.headers.items()))
                            with p.open("ab" if offset else "xb") as out:
                                n=offset;last=0
                                while True:
                                    data=response.read(2**20)
                                    if not data:break
                                    if n+len(data)>end-begin+1:raise ValueError("source piece byte cap exceeded")
                                    out.write(data);n+=len(data)
                                    if time.monotonic()-last>10:
                                        if shutil.disk_usage(root).free<plan["minimum_free_bytes"]+plan["reserved_future_bytes"]:raise ValueError("free-space staging reserve gate")
                                        with lock:state[pid]=n;status("downloading_checked_ranges")
                                        last=time.monotonic()
                                out.flush();os.fsync(out.fileno())
                    if p.stat().st_size!=end-begin+1:raise ValueError("source piece incomplete")
                    d=dict(asset_id=asset["id"],start=begin,end=end,path=str(p.resolve()),bytes=p.stat().st_size,sha256=digest(p),source_url=asset["url"],expected_etag=asset.get("expected_etag"),completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                    with receipt.open("x") as out:json.dump(d,out,indent=2);out.write("\n")
                    with lock:state[pid]=d["bytes"];status("downloading_checked_ranges")
                    return d
                except (OSError,ValueError) as exc:
                    if not is_retryable(exc) or attempt+1>=plan.get("max_attempts_per_piece",4):raise
                    time.sleep(retry_delay(attempt,dict(getattr(exc,"headers",{}) or {})))
    for asset in plan["assets"]:
        current_asset=asset["id"];receipt=receipts/(asset["id"]+".json")
        if receipt.exists():
            old=json.loads(receipt.read_text());p=pathlib.Path(old["path"])
            if p.stat().st_size!=old["bytes"] or digest(p)!=old["sha256"]:raise ValueError("immutable completed source changed")
            done.append(asset["id"]);continue
        prefix=asset.get("prefix");start=prefix["bytes"] if prefix else 0;ranges=plan_ranges(start,asset["expected_bytes"],plan["segment_bytes"])
        with asset_lease(receipts/(asset["id"]+".asset.lease")):
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=plan["workers"]) as pool:
                    pieces=list(pool.map(lambda r:fetch_piece(asset,r),ranges))
                with lock:status("verifying_and_assembling_new_final_asset")
                d=assemble_asset(asset,root,pieces)
                if asset["url"].endswith(".gz"):
                    total_uncompressed=0
                    with gzip.open(d["path"],"rb") as f:
                        for chunk in iter(lambda:f.read(8*2**20),b""):total_uncompressed+=len(chunk)
                    d["gzip_crc_full_stream_passed"]=True;d["decompressed_bytes_streamed_not_materialized"]=total_uncompressed
                d["state"]="verified_available";d["completed_utc"]=datetime.datetime.now(datetime.timezone.utc).isoformat()
                with receipt.open("x") as out:json.dump(d,out,indent=2);out.write("\n")
                done.append(asset["id"]);print("VERIFIED_SEGMENTED_ASSET",asset["id"],d["bytes"],flush=True)
            except Exception as exc:
                failed.append(asset["id"]);suffix=str(time.time_ns())
                with (receipts/(asset["id"]+"-failure-"+suffix+".json")).open("x") as out:json.dump(dict(asset_id=asset["id"],error_type=type(exc).__name__,error=str(exc),all_prefix_and_piece_bytes_preserved=True),out,indent=2)
                print("FAILED_SEGMENTED_ASSET",asset["id"],type(exc).__name__,str(exc),flush=True)
    with lock:status("finite_segmented_acquisition_finished")
    print("SEGMENTED_FINISHED",len(done),"verified",len(failed),"failed",flush=True)

if __name__=="__main__":main()
