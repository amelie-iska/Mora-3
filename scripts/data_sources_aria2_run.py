"""Finite exclusive checkpointed STRING aria2 owner, streaming final QC and live evidence."""
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
import subprocess
import threading
import time
import urllib.request
from data_sources_aria2 import aria2_command,final_sha256
from data_sources_fetch import digest
from data_sources_fetch_fast import asset_lease


def sockets(pid):
    root=pathlib.Path("/proc")/str(pid);inodes=set()
    try:
        for p in (root/"fd").iterdir():
            try:
                m=re.fullmatch(r"socket:\[(\d+)\]",os.readlink(p))
                if m:inodes.add(m[1])
            except OSError:pass
        count=0
        for name in ("tcp","tcp6"):
            for line in (root/"net"/name).read_text().splitlines()[1:]:
                fields=line.split()
                if len(fields)>9 and fields[9] in inodes and fields[3]=="01":count+=1
        return count
    except OSError:return None


def console_progress(path):
    if not path.exists():return None
    with path.open("rb") as f:
        f.seek(max(0,path.stat().st_size-65536));text=f.read().decode(errors="replace")
    pattern=r"([0-9.,]+)(B|KiB|MiB|GiB)/([0-9.,]+)(B|KiB|MiB|GiB)\((\d+)%\)\s+CN:(\d+)\s+DL:([0-9.,]+)(B|KiB|MiB|GiB)"
    matches=list(re.finditer(pattern,text))
    if not matches:return None
    m=matches[-1];scale={"B":1,"KiB":2**10,"MiB":2**20,"GiB":2**30}
    return dict(downloaded_bytes=int(float(m[1].replace(",",""))*scale[m[2]]),total_bytes=int(float(m[3].replace(",",""))*scale[m[4]]),percent=int(m[5]),reported_connections=int(m[6]),reported_bytes_per_second=int(float(m[7].replace(",",""))*scale[m[8]]))


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--plan",required=True);args=parser.parse_args();planpath=pathlib.Path(args.plan).resolve();root=planpath.parent;plan=json.loads(planpath.read_text());old=pathlib.Path("/proc")/str(plan["old_pid"])
    if old.exists() and not old.joinpath("status").read_text().split("State:")[1].strip().startswith("Z"):raise ValueError("old queue not retired; refuse concurrent writer")
    if not plan["prior_writer_retired"]:raise ValueError("retirement receipt required")
    receipts=root/"audits/asset-receipts-aria2-string-v1";receipts.mkdir(exist_ok=True);live=root/"live-acquisition-aria2-string-v1.json";lock=threading.Lock();began=time.monotonic();transferred=[];verified=[];failures=[];crc_state={};active={};futures=[]
    def status(phase):
        live.write_text(json.dumps(dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wrapper_pid=os.getpid(),phase=phase,active_transfer=active,completed_byte_transfers=transferred,verified_asset_ids=verified,failures=failures,crc_state=crc_state,elapsed_seconds=time.monotonic()-began,connection_policy=plan["global_concurrency_bound"],free_bytes=shutil.disk_usage(root).free),indent=2)+"\n")
    def finish_crc(asset,record):
        n=0;last=0
        with gzip.open(record["path"],"rb") as f:
            while True:
                data=f.read(8*2**20)
                if not data:break
                n+=len(data)
                if n-last>=2**30:
                    with lock:crc_state[asset["id"]]=dict(state="streaming_gzip_CRC",decompressed_bytes=n);status("downloads_and_streaming_source_integrity")
                    last=n
        record.update(state="verified_available",gzip_crc_full_stream_passed=True,decompressed_bytes_streamed_not_materialized=n,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        with (receipts/(asset["id"]+".json")).open("x") as f:json.dump(record,f,indent=2);f.write("\n")
        with lock:verified.append(asset["id"]);crc_state[asset["id"]]=dict(state="full_gzip_CRC_passed",decompressed_bytes=n);status("source_integrity_verified")
        print("SOURCE_VERIFIED",asset["id"],record["bytes"],"gzip_uncompressed",n,flush=True)
    pool=concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        for asset in plan["assets"]:
            with asset_lease(receipts/(asset["id"]+".asset.lease")):
                destination=root/asset["destination_path"];destination.parent.mkdir(parents=True,exist_ok=True)
                prefix=asset.get("prefix")
                if destination.exists():raise FileExistsError("new aria2 destination already exists; preserve for explicit native aria2 resume")
                if prefix:
                    src=pathlib.Path(prefix["path"])
                    if src.stat().st_size!=prefix["bytes"] or digest(src)!=prefix["sha256"]:raise ValueError("old prefix changed")
                    with src.open("rb") as i,destination.open("xb") as out:shutil.copyfileobj(i,out,8*2**20);out.flush();os.fsync(out.fileno())
                    if digest(destination)!=prefix["sha256"]:raise ValueError("read-only prefix copy mismatch")
                with urllib.request.urlopen(urllib.request.Request(asset["url"],method="HEAD"),timeout=30) as response:
                    if response.headers.get("ETag")!=asset["expected_etag"] or int(response.headers["Content-Length"])!=asset["expected_bytes"]:raise ValueError("source entity changed before aria2")
                if shutil.disk_usage(root).free<plan["minimum_free_bytes"]+plan["derived_budget_bytes"]:raise ValueError("free-space reserve gate")
                cmd=aria2_command(asset,destination,connections=plan["connections_per_file"]);cmd.insert(-1,"--human-readable=false")
                log=root/"logs"/("aria2-"+asset["id"]+".log");f=log.open("x");proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT);f.close();started=time.monotonic()
                launch=dict(asset_id=asset["id"],aria2_pid=proc.pid,command=cmd,log=str(log),prefix_bytes_reused=prefix["bytes"] if prefix else 0,launched_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                with (receipts/(asset["id"]+"-launch.json")).open("x") as f:json.dump(launch,f,indent=2);f.write("\n")
                while proc.poll() is None:
                    progress=console_progress(log)
                    with lock:active={**launch,"established_socket_connections":sockets(proc.pid),"aria2_console_progress":progress,"elapsed_seconds":time.monotonic()-started};status("aria2_transfer_active")
                    time.sleep(5)
                if proc.returncode!=0:
                    error=dict(asset_id=asset["id"],aria2_pid=proc.pid,exit_code=proc.returncode,log=str(log),all_old_and_new_source_bytes_preserved=True)
                    failures.append(error)
                    with (receipts/(asset["id"]+"-failure.json")).open("x") as f:json.dump(error,f,indent=2)
                    print("ARIA2_FAILED",asset["id"],proc.returncode,flush=True);continue
                checksum=final_sha256(destination,asset)
                if prefix:
                    if digest(pathlib.Path(prefix["path"]))!=prefix["sha256"]:raise ValueError("protected old prefix changed")
                    h=hashlib.sha256()
                    with destination.open("rb") as f:
                        remaining=prefix["bytes"]
                        while remaining:
                            chunk=f.read(min(8*2**20,remaining));h.update(chunk);remaining-=len(chunk)
                    if h.hexdigest()!=prefix["sha256"]:raise ValueError("final source prefix mismatch")
                elapsed=time.monotonic()-started;record={**asset,"path":str(destination.resolve()),"bytes":destination.stat().st_size,"sha256":checksum,"aria2_exit_code":0,"aria2_pid":proc.pid,"network_transfer_elapsed_seconds":elapsed,"reused_prefix_bytes":prefix["bytes"] if prefix else 0,"network_suffix_mean_bytes_per_second":(asset["expected_bytes"]-(prefix["bytes"] if prefix else 0))/elapsed,"expected_etag_verified_before_start":True,"original_prefix_unchanged":True}
                with lock:transferred.append(dict(asset_id=asset["id"],bytes=record["bytes"],sha256=checksum,source_integrity_state="gzip_CRC_pending"));active={};crc_state[asset["id"]]=dict(state="queued_gzip_CRC");status("byte_transfer_complete_integrity_pending")
                futures.append(pool.submit(finish_crc,asset,record))
                print("ARIA2_BYTES_COMPLETE",asset["id"],record["bytes"],"suffix_mean_Bps",record["network_suffix_mean_bytes_per_second"],flush=True)
        for future in futures:future.result()
    finally:pool.shutdown(wait=True)
    with lock:active={};status("finite_aria2_source_job_finished")
    print("ARIA2_JOB_FINISHED","verified",len(verified),"failures",len(failures),flush=True)

if __name__=="__main__":main()
