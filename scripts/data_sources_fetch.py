"""Finite pinned public acquisition; append-only resume, immutable completed assets."""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import re
import shutil
import time
import urllib.request


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(2**20),b""):h.update(chunk)
    return h.hexdigest()


def validate_response(asset, offset, status, headers):
    hdr={k.lower():v for k,v in headers.items()}
    expected=asset["expected_bytes"]
    if offset:
        if status!=206:raise ValueError("resume requires HTTP206; existing bytes preserved")
        match=re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)",hdr.get("content-range",""))
        if not match or int(match[1])!=offset or int(match[3])!=expected or int(match[2])!=expected-1:
            raise ValueError("resume Content-Range mismatch")
    elif status!=200:raise ValueError("new acquisition requires HTTP200")
    if asset.get("expected_etag") and hdr.get("etag") and hdr["etag"]!=asset["expected_etag"]:
        raise ValueError("source ETag changed; existing bytes preserved")
    if hdr.get("content-length") and int(hdr["content-length"])!=expected-offset:
        raise ValueError("response content length mismatch")


def fetch_asset(asset, root, opener=urllib.request.urlopen, progress=None):
    root=pathlib.Path(root);path=root/asset["relative_path"];path.parent.mkdir(parents=True,exist_ok=True)
    expected=asset["expected_bytes"];offset=path.stat().st_size if path.exists() else 0
    if offset>expected:raise ValueError("existing pending asset larger than expected")
    if offset<expected:
        headers={"Accept-Encoding":"identity","User-Agent":"Mora3-public-source-intake/1.0"}
        if offset:
            headers["Range"]="bytes=%d-"%offset
            if asset.get("expected_etag"):headers["If-Range"]=asset["expected_etag"]
        req=urllib.request.Request(asset["url"],headers=headers)
        with opener(req,timeout=120) as response:
            validate_response(asset,offset,response.status,dict(response.headers.items()))
            with path.open("ab" if offset else "xb") as f:
                total=offset;last=0
                while True:
                    chunk=response.read(2**20)
                    if not chunk:break
                    if total+len(chunk)>expected:raise ValueError("source byte cap exceeded")
                    f.write(chunk);total+=len(chunk)
                    if progress and time.monotonic()-last>=10:
                        progress(asset,total);last=time.monotonic()
                f.flush();os.fsync(f.fileno())
        if path.stat().st_size!=expected:raise ValueError("incomplete source preserved for verified resume")
    checksum=digest(path)
    if asset.get("expected_sha256") and checksum!=asset["expected_sha256"]:
        raise ValueError("source checksum mismatch; original bytes preserved")
    return {**asset,"path":str(path.resolve()),"bytes":path.stat().st_size,"sha256":checksum,
        "upstream_sha256_passed":bool(asset.get("expected_sha256")),"state":"verified_available",
        "resumed_from_bytes":offset,"completed_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--plan",required=True);parser.add_argument("--group",required=True);args=parser.parse_args()
    plan_path=pathlib.Path(args.plan).resolve();root=plan_path.parent;plan=json.loads(plan_path.read_text())
    if sum(a["expected_bytes"] for a in plan["assets"])>plan["raw_budget_bytes"]:raise ValueError("raw plan budget exceeded")
    assets=[a for a in plan["assets"] if a["group"]==args.group];receipt_dir=root/"audits"/("asset-receipts-"+args.group);receipt_dir.mkdir(exist_ok=True)
    live=root/("live-acquisition-"+args.group+".json");done=[];fail=[]
    def progress(asset,total):
        if shutil.disk_usage(root).free<plan["minimum_free_bytes"]+plan["derived_budget_bytes"]:raise ValueError("free-space reserve gate")
        live.write_text(json.dumps(dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),state="live_finite_acquisition",group=args.group,asset_id=asset["id"],downloaded_bytes=total,expected_bytes=asset["expected_bytes"],completed_asset_ids=done,failed_asset_ids=fail),indent=2)+"\n")
    for asset in assets:
        progress(asset,(root/asset["relative_path"]).stat().st_size if (root/asset["relative_path"]).exists() else 0)
        receipt_path=receipt_dir/(asset["id"]+".json")
        if receipt_path.exists():
            prior=json.loads(receipt_path.read_text())
            if prior["state"]=="verified_available":
                path=pathlib.Path(prior["path"])
                if path.stat().st_size!=prior["bytes"] or digest(path)!=prior["sha256"]:raise ValueError("completed immutable source changed")
                done.append(asset["id"]);continue
        try:
            receipt=fetch_asset(asset,root,progress=progress)
            with receipt_path.open("x") as f:json.dump(receipt,f,indent=2);f.write("\n")
            done.append(asset["id"]);print("VERIFIED",asset["id"],receipt["bytes"],flush=True)
        except Exception as exc:
            fail.append(asset["id"])
            error=dict(asset_id=asset["id"],error_type=type(exc).__name__,error=str(exc),partial_path=str(root/asset["relative_path"]),state="failed_preserved_partial",updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            suffix=datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
            with (receipt_dir/(asset["id"]+"-failure-"+suffix+".json")).open("x") as f:json.dump(error,f,indent=2)
            print("FAILED",asset["id"],type(exc).__name__,str(exc),flush=True)
    live.write_text(json.dumps(dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),state="finite_acquisition_finished",group=args.group,completed_asset_ids=done,failed_asset_ids=fail),indent=2)+"\n")
    print("GROUP_FINISHED",args.group,"verified",len(done),"failed",len(fail),flush=True)

if __name__=="__main__":main()
