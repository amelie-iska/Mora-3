"""Finite original RNA/Transition1x acquisitions with pinned entity and whole-byte QC."""
import argparse
import datetime
import gzip
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import time
from data_sources_aria2 import aria2_command
from data_sources_aria2_run import console_progress,sockets

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write_new(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
def finalize(path,asset):
    if path.stat().st_size!=asset['expected_bytes']:raise ValueError('expected source byte length mismatch')
    sha=hashlib.sha256();md5=hashlib.md5()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):sha.update(block);md5.update(block)
    if asset.get('expected_sha256') and sha.hexdigest()!=asset['expected_sha256']:raise ValueError('upstreamSHA256 mismatch; source retained')
    if asset.get('expected_md5') and md5.hexdigest()!=asset['expected_md5']:raise ValueError('upstreamMD5 mismatch; source retained')
    decoded=None;is_hdf=False;crc=False
    if asset['format']=='gzip_fasta':
        decoded=0
        with gzip.open(path,'rb') as f:
            for block in iter(lambda:f.read(8*1024**2),b''):decoded+=len(block)
        crc=True
    elif asset['format']=='hdf5':
        with path.open('rb') as f:
            if f.read(8)!=b'\x89HDF\r\n\x1a\n':raise ValueError('sourceHDF5signaturemismatch')
        is_hdf=True
    else:raise ValueError('unreviewed source format')
    return dict(sha256=sha.hexdigest(),md5=md5.hexdigest(),upstream_md5_passed=True if asset.get('expected_md5') else None,gzip_crc_full_stream_passed=crc,uncompressed_bytes_streamed=decoded,HDF5_signature_passed=is_hdf,HDF5_structural_QC_performed=False)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);args=parser.parse_args();pp=pathlib.Path(args.plan).resolve();root=pp.parent;plan=json.loads(pp.read_text());job=plan['job_id']
    if plan['aggregate_committed_raw_bytes']>80*1024**3 or plan['aggregate_committed_derived_bytes']>160*1024**3 or plan['global_connection_accounting']['total_max']>32:raise ValueError('aggregate budget/connection gate')
    lease=root/('lease-'+job+'.json');write_new(lease,dict(pid=os.getpid(),job_id=job,started_utc=now(),plan=str(pp),exclusive_assets=[x['id'] for x in plan['assets']]))
    receipts=root/'audits/asset-receipts-original-rna-ts-v1';receipts.mkdir(exist_ok=True);done=[];live=root/('live-acquisition-'+job+'.json');current=None
    def status(phase,extra=None):
        data=dict(updated_utc=now(),wrapper_pid=os.getpid(),job_id=job,phase=phase,verified_assets=done,active_asset=current,global_connection_accounting=plan['global_connection_accounting'],free_bytes=shutil.disk_usage(root).free)
        if extra:data.update(extra)
        live.write_text(json.dumps(data,indent=2)+'\n')
    try:
        for asset in plan['assets']:
            current=asset['id'];destination=root/asset['relative_path'];destination.parent.mkdir(parents=True,exist_ok=True)
            if destination.exists():raise FileExistsError('new original destination already exists; no overlap/overwrite')
            if shutil.disk_usage(root).free<plan['minimum_free_bytes']+plan['reserved_derived_bytes']:raise ValueError('free-space reserve')
            log=root/'logs'/('aria2-'+asset['id']+'.log');cmd=aria2_command(asset,destination,connections=plan['connections_per_asset'],min_split='16M');cmd.insert(-1,'--human-readable=false')
            with log.open('x') as f:
                child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
                while child.poll() is None:
                    status('live finite aria2 original-source transfer',dict(aria2_pid=child.pid,log=str(log),progress=console_progress(log),actual_established_connections=sockets(child.pid),expected_bytes=asset['expected_bytes']));time.sleep(5)
                code=child.returncode
            if code:raise ValueError('source transfer failed with code'+str(code)+'; original/control retained')
            status('whole original source integrity finalization',dict(log=str(log),expected_bytes=asset['expected_bytes']))
            integrity=finalize(destination,asset);receipt=dict(asset,path=str(destination.resolve()),bytes=destination.stat().st_size,state='verified_available',completed_utc=now(),reader_support=False,task_admitted_records=0,**integrity)
            write_new(receipts/(asset['id']+'.json'),receipt);done.append(asset['id']);status('original source verified available');print('ORIGINAL_SOURCE_VERIFIED',asset['id'],asset['expected_bytes'],flush=True)
        current=None;status('finite original source acquisition completed');print('ORIGINALS_FINISHED',job,len(done),flush=True)
    except Exception as exc:status('stopped; original partials retained',dict(error=type(exc).__name__+':'+str(exc)));raise
if __name__=='__main__':main()
