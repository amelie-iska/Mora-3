import argparse,pathlib,json,os,time,shutil,urllib.request,datetime
from data_sources_fetch import fetch_asset
from data_sources_originals import finalize,write_new,now

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);args=parser.parse_args();pp=pathlib.Path(args.plan).resolve();root=pp.parent;plan=json.loads(pp.read_text());job=plan['job_id'];asset=plan['assets'][0];path=root/asset['relative_path']
    if len(plan['assets'])!=1 or plan['connections_per_asset']!=1:raise ValueError('unreviewed serial stream plan')
    if path.exists():raise FileExistsError('new immutable namespace required')
    if shutil.disk_usage(root).free<plan['minimum_free_bytes']+plan['reserved_derived_bytes']:raise ValueError('free-space reserve')
    write_new(root/('lease-'+job+'.json'),dict(pid=os.getpid(),started_utc=now(),source_asset=asset['id'],plan=str(pp)))
    live=root/('live-acquisition-'+job+'.json');started=time.monotonic()
    def status(phase,n=0,error=None):
        live.write_text(json.dumps(dict(updated_utc=now(),pid=os.getpid(),job_id=job,phase=phase,asset_id=asset['id'],downloaded_bytes=n,expected_bytes=asset['expected_bytes'],elapsed_seconds=time.monotonic()-started,measured_bytes_per_second=n/max(time.monotonic()-started,0.001),source_connections=1,error=error),indent=2)+'\n')
    status('live finite public append-only original stream')
    try:
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));record=fetch_asset(asset,root,opener=opener.open,progress=lambda a,n:status('live finite public append-only original stream',n))
        status('whole original source MD5/SHA/format finalization',record['bytes']);integrity=finalize(path,asset);record.update(integrity,state='verified_available',task_admitted_records=0,reader_support=False)
        target=root/'audits/asset-receipts-original-rna-ts-v1'/(asset['id']+'-stream-v2.json');write_new(target,record);status('finite original source stream completed and upstream MD5 verified',record['bytes']);print('TS_ORIGINAL_STREAM_VERIFIED',record['bytes'],integrity['upstream_md5_passed'],flush=True)
    except Exception as exc:
        status('stopped; immutable original partial retained',path.stat().st_size if path.exists() else 0,type(exc).__name__+':'+str(exc));raise
if __name__=='__main__':main()
