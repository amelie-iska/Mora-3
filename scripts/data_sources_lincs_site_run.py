"""Exactly one approved native five-channel LINCS site, existing finite gated aria2."""
import argparse,datetime,fcntl,hashlib,json,os,pathlib,subprocess,sys,time,urllib.request
from data_sources_aria2 import aria2_command
from data_sources_lincs_download import write_new,mutable,stop_only_owned,digest
from data_sources_mmseqs_supervisor import spawn_gated_group,release_gate
from data_sources_workflow_guard import _wait_empty_session
from data_sources_lincs_pixels import decode_qc
NOW=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def raw_bytes(B,P):
    return sum(p.stat().st_size for p in B.rglob('*') if p.is_file() and not p.is_symlink() and 'curated' not in p.relative_to(B).parts and not p.is_relative_to(P/'qc'))
def site_command(assets,P,inputfile):
    cmd=aria2_command(dict(url=assets[0]['source_url'],expected_bytes=assets[0]['expected_bytes']),P/'raw/unused-not-created',connections=1,min_split='8M',max_tries=3)[:-1]
    cmd=[x for x in cmd if not x.startswith(('--dir=','--out=','--max-concurrent-downloads='))]+['--max-concurrent-downloads=5','--input-file='+str(inputfile),'--no-netrc=true','--netrc-path=/dev/null','--check-certificate=true']
    return cmd

def main():
    a=argparse.ArgumentParser();a.add_argument('--plan',required=True);args=a.parse_args();pp=pathlib.Path(args.plan).resolve();P=pp.parent;B=P.parent;p=json.loads(pp.read_text());assets=p['assets'];source=p['lineage_source'];lineage=json.loads(pathlib.Path(source['path']).read_text())
    if digest(source['path'])!=source['sha256'] or len(assets)!=5 or p['payload_bytes']!=42322896 or {x['native_acquisition_fields']['ChannelID']['value'] for x in assets}!=set('12345'):raise ValueError('exact original complete native site required')
    expected=[x for row in lineage['rows'] if row['native_source_row']['Metadata_Well']=='A07' for x in row['image_records'] if x['native_acquisition_fields']['FieldID']['value']=='1']
    if len(expected)!=5 or any(not any(x['source_object']==e['official_object'] and x['native_acquisition_fields']==e['native_acquisition_fields'] for e in expected) for x in assets):raise ValueError('exact site lineage drift')
    if any(pathlib.Path(x['destination']).exists() or pathlib.Path(x['destination']+'.aria2').exists() for x in assets):raise FileExistsError('exclusive immutable originals; no reset')
    for x in assets:
        if pathlib.Path(x['destination']).parent!=P/'raw' or x['source_url']!='https://cellpainting-gallery.s3.amazonaws.com/'+x['source_object']['Key']:raise ValueError('different source or destination')
        with urllib.request.urlopen(urllib.request.Request(x['source_url'],method='HEAD',headers={'If-Match':x['expected_etag']}),timeout=30) as r:
            if r.headers['ETag']!=x['expected_etag'] or int(r.headers['Content-Length'])!=x['expected_bytes']:raise ValueError('source entity drift')
    used=raw_bytes(B,P);free=os.statvfs(B).f_bavail*os.statvfs(B).f_frsize
    if used+p['payload_bytes']+p['maximum_new_transfer_overhead_bytes']+p['raw_stop_margin_bytes']>p['raw_cap_bytes'] or free<p['minimum_free_bytes']+p['payload_bytes']:raise ValueError('whole five-channel site does not fit unchanged reservation')
    with (P/'audits/exclusive-site.lease').open('x+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);lease.write(str(os.getpid()));lease.flush();os.fsync(lease.fileno());entries=[]
        for x in assets:entries.append(x['source_url']+'\n  dir='+str(P/'raw')+'\n  out='+pathlib.Path(x['destination']).name+'\n  header=If-Match: '+x['expected_etag']+'\n')
        inputfile=P/'audits/pinned-aria2-five-assets.txt'
        with inputfile.open('x') as f:f.write(''.join(entries));f.flush();os.fsync(f.fileno())
        cmd=site_command(assets,P,inputfile)
        start=time.monotonic();proc=None;identity=None;control=P/'audits/native-control';control.mkdir();spec=dict(command=cmd,cwd=str(control),state_path=str(control/'state.json'),gate_timeout=5,guard=dict(stage_cap_bytes=p['raw_cap_bytes'],stop_margin_bytes=p['raw_stop_margin_bytes'],minimum_free_bytes=p['minimum_free_bytes'],max_seconds=p['download_max_seconds']),started_monotonic=start,address_space_limit_bytes=1024**3,threads=1)
        try:
            with (P/'logs/aria2-five-originals.log').open('xb') as log:proc,identity,command=spawn_gated_group(spec,stdout=log)
            write_new(P/'audits/stable-owner-before-release.json',dict(created_utc=NOW(),identity=identity,supervisor_command=command,native_command=cmd,raw_bytes_before=used,remaining_raw_before=p['raw_cap_bytes']-used));release_gate(proc)
            while proc.poll() is None:
                used=raw_bytes(B,P);fs=os.statvfs(B)
                if used>=p['raw_cap_bytes']-p['raw_stop_margin_bytes'] or fs.f_bavail*fs.f_frsize<p['minimum_free_bytes'] or time.monotonic()-start>p['download_max_seconds']:stop_only_owned(proc,identity);raise ValueError('finite unchanged raw/free guard; all partials retained')
                mutable(P/'live.json',dict(updated_utc=NOW(),wrapper_pid=os.getpid(),phase='complete five-channel native original transfer',supervisor=identity,raw_bytes=used,all_channels_required=True));time.sleep(.2)
            if proc.returncode:raise ValueError('finite native transfer exit '+str(proc.returncode)+'; no wrapper retry/reset')
            proc.stdin.close();_wait_empty_session(proc.pid,timeout=5);records=[]
            for x in assets:
                path=pathlib.Path(x['destination'])
                if path.stat().st_size!=x['expected_bytes']:raise ValueError('whole original length mismatch')
                before=digest(path)
                with urllib.request.urlopen(urllib.request.Request(x['source_url'],method='HEAD',headers={'If-Match':x['expected_etag']}),timeout=30) as r:
                    after=dict(r.headers)
                    if after['ETag']!=x['expected_etag'] or int(after['Content-Length'])!=x['expected_bytes']:raise ValueError('source entity after transfer drift')
                qc=decode_qc(path)
                if digest(path)!=before:raise ValueError('immutable original changed during full pixel QC')
                record=dict(source=x,original_path=str(path),original_bytes=path.stat().st_size,original_sha256=before,source_HEAD_after=after,ETag_is_not_assumed_whole_MD5=True,QC=qc,acquisition_registration_verified=False,task_admitted_records=0);records.append(record);write_new(P/'qc'/('channel-'+x['native_acquisition_fields']['ChannelID']['value']+'.json'),record)
            if len(records)!=5 or sum(r['original_bytes'] for r in records)!=42322896:raise ValueError('complete site conservation failed')
            derived=sum(x.stat().st_size for x in (P/'qc').rglob('*') if x.is_file())
            if derived>p['derived_QC_cap_bytes'] or raw_bytes(B,P)>=p['raw_cap_bytes']-p['raw_stop_margin_bytes']:raise ValueError('final raw/derived cap gate')
            write_new(P/'qc/manifest.json',dict(schema='mora3.LINCS_A07_native_complete_five_channel_site.v1',completed_utc=NOW(),registered_plan=str(pp),plan_sha256=digest(pp),lineage_source=source,channel_order=['1:HOECHST 33342','2:Alexa 488','3:488 long','4:Alexa 568','5:Alexa 647'],channels=records,raw_bytes=42322896,decoded_pixel_count=sum(r['QC']['pixels'] for r in records),decoded_copies_written=False,source_root_conservative_raw_bytes=raw_bytes(B,P),shared_raw_cap_bytes=p['raw_cap_bytes'],new_QC_derived_bytes_before_manifest=derived,new_QC_derived_cap_bytes=p['derived_QC_cap_bytes'],source_context_comparison=p['source_context_comparison'],native_profile_source_row=p['native_profile_source_row'],same_cell_GE_or_masks_or_optical_registration=False,profile_pixel_reaggregation_verified=False,CP_author_hours=48,GE_separate_author_hours=24,task_admitted_records=0,elapsed_seconds=time.monotonic()-start));mutable(P/'live.json',dict(updated_utc=NOW(),phase='complete original pixels, full native decode/readback QC',wrapper_pid=None,task_admitted_records=0));print('LINCS_A07_COMPLETE_FIVE_NATIVE_CHANNELS_QC',flush=True)
        except BaseException as exc:
            if proc is not None and proc.poll() is None and identity is not None:stop_only_owned(proc,identity)
            write_new(P/'audits/failed-preserved.json',dict(at_utc=NOW(),error=type(exc).__name__+':'+str(exc),partials_preserved=True,no_partial_channel_admission=True));raise
if __name__=='__main__':main()
