"""Bounded concurrent pinned public transfers in a separate leased asset namespace.

Live serial downloader, its plan and existing source bytes remain untouched.
"""
import argparse
import concurrent.futures
import contextlib
import datetime
import fcntl
import json
import os
import pathlib
import shutil
import threading
import time
import urllib.error
import urllib.parse
from data_sources_fetch import fetch_asset, digest


@contextlib.contextmanager
def asset_lease(path):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a+") as f:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        f.write(json.dumps(dict(pid=os.getpid(),acquired_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))+"\n");f.flush()
        try:yield
        finally:fcntl.flock(f.fileno(),fcntl.LOCK_UN)


def validate_plan(plan):
    assets=plan["assets"];ids=[a["id"] for a in assets];paths=[a["relative_path"] for a in assets]
    if len(ids)!=len(set(ids)) or len(paths)!=len(set(paths)):raise ValueError("duplicate asset IDs/destination writers")
    if not 1<=plan["workers"]<=16 or not 1<=plan["per_host_connections"]<=16:raise ValueError("connection cap outside tested bounds")
    if any(pathlib.Path(p).is_absolute() or ".." in pathlib.Path(p).parts for p in paths):raise ValueError("destination path escape")
    if any(a["expected_bytes"]<=0 for a in assets):raise ValueError("invalid source expected byte count")
    if sum(a["expected_bytes"] for a in assets)+plan.get("other_registered_raw_bytes",0)>plan["raw_budget_bytes"]:raise ValueError("aggregate raw budget exceeded")


def retry_delay(attempt,headers):
    retry={k.lower():v for k,v in headers.items()}.get("retry-after")
    if retry:
        try:value=int(retry)
        except ValueError:value=60
        if value>300:raise ValueError("server Retry-After exceeds finite retry budget; preserve and defer")
        return max(0,value)
    return [5,15,45][min(attempt,2)]


def is_retryable(exc):
    if isinstance(exc,urllib.error.HTTPError):
        return exc.code in (403,408,429,500,502,503,504)
    return isinstance(exc,(urllib.error.URLError,TimeoutError,ConnectionError))


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--plan",required=True);args=parser.parse_args()
    planpath=pathlib.Path(args.plan).resolve();root=planpath.parent;plan=json.loads(planpath.read_text());validate_plan(plan)
    name=plan.get("job_name","fast-uniref50-v1");out=root/"audits"/("asset-receipts-"+name);out.mkdir(exist_ok=True);leases=root/"audits"/("leases-"+name)
    lock=threading.Lock();started=time.monotonic();current={};initial={};done=[];failed=[];attempts={}
    for a in plan["assets"]:
        p=root/a["relative_path"];initial[a["id"]]=p.stat().st_size if p.exists() else 0;current[a["id"]]=initial[a["id"]]
    hosts={urllib.parse.urlparse(a["url"]).hostname for a in plan["assets"]};sems={h:threading.BoundedSemaphore(plan["per_host_connections"]) for h in hosts}
    live=root/("live-acquisition-"+name+".json")
    def status(state="live_finite_fast_acquisition"):
        elapsed=max(time.monotonic()-started,0.001);transferred=sum(max(current[k]-initial[k],0) for k in current)
        d=dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),state=state,configured_workers=plan["workers"],per_host_connections=plan["per_host_connections"],active_initial_bytes=initial,observed_asset_bytes=current,verified_asset_ids=done,failed_asset_ids=failed,attempts=attempts,elapsed_seconds=elapsed,transferred_bytes=transferred,measured_bytes_per_second=transferred/elapsed,free_bytes=shutil.disk_usage(root).free)
        live.write_text(json.dumps(d,indent=2)+"\n")
    def progress(a,n):
        if shutil.disk_usage(root).free<plan["minimum_free_bytes"]+plan["derived_budget_bytes"]:raise ValueError("reserved free-space gate")
        with lock:current[a["id"]]=n;status()
    def worker(a):
        with asset_lease(leases/(a["id"]+".lease")):
            receipt=out/(a["id"]+".json")
            if receipt.exists():
                d=json.loads(receipt.read_text());p=pathlib.Path(d["path"])
                if p.stat().st_size!=d["bytes"] or digest(p)!=d["sha256"]:raise ValueError("immutable completed asset changed")
                with lock:done.append(a["id"]);status()
                return
            for attempt in range(plan.get("max_attempts_per_asset",4)):
                with lock:attempts[a["id"]]=attempt+1
                try:
                    with sems[urllib.parse.urlparse(a["url"]).hostname]:
                        d=fetch_asset(a,root,progress=progress)
                    with receipt.open("x") as f:json.dump(d,f,indent=2);f.write("\n")
                    with lock:current[a["id"]]=d["bytes"];done.append(a["id"]);status()
                    print("VERIFIED",a["id"],d["bytes"],flush=True);return
                except (OSError,ValueError) as exc:
                    p=root/a["relative_path"];n=p.stat().st_size if p.exists() else 0
                    retryable=is_retryable(exc)
                    # Checksum, range or entity changes are terminal; never overwrite/restart those bytes.
                    if not retryable or attempt+1>=plan.get("max_attempts_per_asset",4):
                        failure=dict(asset_id=a["id"],error_type=type(exc).__name__,error=str(exc),preserved_partial_bytes=n,attempts=attempt+1,failed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                        with (out/(a["id"]+"-failure-"+str(time.time_ns())+".json")).open("x") as f:json.dump(failure,f,indent=2)
                        with lock:current[a["id"]]=n;failed.append(a["id"]);status()
                        print("FAILED",a["id"],type(exc).__name__,str(exc),flush=True);return
                    delay=retry_delay(attempt,dict(getattr(exc,"headers",{}) or {}));time.sleep(delay)
            raise RuntimeError("finite attempts exhausted")
    with lock:status()
    with concurrent.futures.ThreadPoolExecutor(max_workers=plan["workers"]) as pool:
        futures={pool.submit(worker,a):a["id"] for a in plan["assets"]}
        for f in concurrent.futures.as_completed(futures):
            try:f.result()
            except Exception as exc:
                with lock:failed.append(futures[f]);status()
                print("LEASE_OR_WORKER_FAILURE",futures[f],type(exc).__name__,str(exc),flush=True)
    with lock:status("finite_fast_acquisition_finished")
    print("FAST_FINISHED","verified",len(done),"failed",len(failed),flush=True)

if __name__=="__main__":main()
