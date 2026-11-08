"""Finite reviewed RNAcentral immutable-original copy/resume; no original writes."""
import argparse,datetime,json,os,pathlib,shutil,subprocess,time,urllib.request
from data_sources_rnacentral_resume import sha,validate_checkpoint,clone_partial,checked_new_bytes
from data_sources_aria2 import aria2_command
from data_sources_aria2_run import console_progress,sockets
from data_sources_originals import finalize
REVIEWED_FILES=['scripts/data_sources_rnacentral_resume.py','scripts/data_sources_rnacentral_resume_run.py','scripts/data_sources_aria2.py','scripts/data_sources_aria2_run.py','scripts/data_sources_originals.py','scripts/data_sources_fetch.py','scripts/data_sources_fetch_fast.py']
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def new(p,j):
    with p.open('x') as f:json.dump(j,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def verify_review(plan_path,approval_path):
    a=json.loads(pathlib.Path(approval_path).read_text())
    if not a.get('approved') or a.get('action')!='RNAcentral28_immutable_copy_resume_v2' or a.get('plan_sha256')!=sha(plan_path):raise ValueError('exact RNAcentral copy/resume execution review required')
    if set(a.get('reviewed_code_sha256',{}))!=set(REVIEWED_FILES):raise ValueError('exact source helper/gate hash map required')
    for p in REVIEWED_FILES:
        if sha(p)!=a['reviewed_code_sha256'][p]:raise ValueError('reviewed RNAcentral helper or gate changed')
    return json.loads(pathlib.Path(plan_path).read_text())
def preflight(assets,original):
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));refs=[]
    for a in assets:
        with opener.open(urllib.request.Request(a['url'],method='HEAD'),timeout=20) as r:
            if r.status!=200 or int(r.headers.get('Content-Length','-1'))!=a['expected_bytes'] or r.headers.get('ETag')!=a['expected_etag']:raise ValueError('source HEAD/strong entity mismatch')
            refs.append(dict(asset_id=a['id'],status=r.status,ETag=r.headers['ETag'],bytes=int(r.headers['Content-Length'])))
    a=assets[0];q=urllib.request.Request(a['url'],headers={'Range':'bytes=0-65535','If-Match':a['expected_etag']})
    with opener.open(q,timeout=20) as r:
        body=r.read(65537)
        if r.status!=206 or r.headers.get('Content-Range')!='bytes 0-65535/'+str(a['expected_bytes']) or r.headers.get('ETag')!=a['expected_etag'] or len(body)!=65536:raise ValueError('exact source206 boundary/entity gate')
    data=next(r for r in original['immutable_original_files'] if not r['path'].endswith('.aria2'))
    with pathlib.Path(data['path']).open('rb') as f:
        if f.read(65536)!=body:raise ValueError('source boundary differs from protected old bytes')
    return refs

def run(plan_path,approval_path):
    plan=verify_review(plan_path,approval_path);root=pathlib.Path(plan_path).resolve().parent;job=plan['job_id'];assets=plan['assets'];cp=json.loads(pathlib.Path(plan['original_partial_checkpoint']).read_text());validate_checkpoint(cp,assets[0]);child=None;done=[];active=None
    if plan['aggregate_committed_raw_bytes']>80*1024**3 or plan['aggregate_committed_derived_bytes']>160*1024**3 or plan['global_connection_accounting']['total_max']>32 or plan['connections_per_asset']>16:raise ValueError('budget/global/perhost connection gate')
    if shutil.disk_usage(root).free<200*1024**3+plan['reserved_derived_bytes']:raise ValueError('free-space reserve')
    new(root/('lease-'+job+'.json'),dict(pid=os.getpid(),started_utc=now(),plan=str(pathlib.Path(plan_path).resolve()),plan_sha256=sha(plan_path),approval=str(pathlib.Path(approval_path).resolve()),exclusive_asset_IDs=[a['id'] for a in assets],old_source_worker_exited=True,new_namespace_only=True))
    live=root/('live-acquisition-'+job+'.json');receipt_dir=root/plan['receipt_directory'];receipt_dir.mkdir(exist_ok=False)
    def status(phase,**extra):
        live.write_text(json.dumps(dict(updated_utc=now(),wrapper_pid=os.getpid(),job_id=job,phase=phase,active_asset=active,verified_assets=done,source_connections_max=16,global_connections_max=24,free_bytes=shutil.disk_usage(root).free,**extra),indent=2)+'\n')
    try:
        status('bounded source identity preflight');heads=preflight(assets,cp);new(receipt_dir/'source-preflight.json',dict(observed_utc=now(),HEADs=heads,exact206original64KiB_match=True))
        active=assets[0]['id'];status('copy/hash protected original partial/control into exclusive new namespace')
        clone=clone_partial(cp,assets[0],root/plan['reconstruction_directory'],pathlib.Path(assets[0]['relative_path']).name);new(receipt_dir/'immutable-partial-clone.json',dict(completed_utc=now(),clone=clone,original_checkpoint=plan['original_partial_checkpoint']))
        for i,a in enumerate(assets):
            active=a['id'];destination=root/a['relative_path'];previous=cp['reported_completed_bytes'] if i==0 else 0;maximum_new=plan['additional_active_source_byte_reserve'] if i==0 else a['expected_bytes']
            if i==0:
                if str(destination.resolve())!=clone['destination'] or not pathlib.Path(str(destination)+'.aria2').is_file():raise ValueError('only verified clonedpartial/control may resume')
            else:
                destination.parent.mkdir(parents=True,exist_ok=False)
                if destination.exists():raise FileExistsError('new inactive destination exists')
            if shutil.disk_usage(root).free<200*1024**3+plan['reserved_derived_bytes']:raise ValueError('free-space reserve')
            log=root/'logs'/('aria2-'+job+'-'+a['id']+'.log');cmd=aria2_command(a,destination,connections=16,min_split='16M');cmd.insert(-1,'--human-readable=false')
            with log.open('x') as f:
                child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT);child_pid=child.pid;parts=pathlib.Path('/proc/'+str(child.pid)+'/stat').read_text().split();child_start=int(parts[21]);new(receipt_dir/(a['id']+'-launch.json'),dict(launched_utc=now(),child_pid=child.pid,start_ticks=child_start,command=cmd,original_bytes_unchanged=True,initial_completed_bytes=previous,maximum_new_bytes=maximum_new))
                while child.poll() is None:
                    p=console_progress(log);delta=checked_new_bytes(previous,p,maximum_new)
                    if p is not None and p.get('total_bytes')!=a['expected_bytes']:raise ValueError('source total length drift in resume status')
                    status('live finite aria2 strict resume/neworiginal transfer',aria2_pid=child.pid,aria2_start_ticks=child_start,log=str(log),progress=p,new_downloaded_bytes=delta,actual_established_connections=sockets(child.pid),expected_bytes=a['expected_bytes'],old_completed_bytes=previous);time.sleep(5)
                code=child.returncode
            if code:raise ValueError('finite aria2 source transfer failed code'+str(code)+'; old/new partials/control retained')
            child=None;status('whole source SHA/MD5/length/gzipCRC finalization',source_path=str(destination))
            integrity=finalize(destination,a);validate_checkpoint(cp,assets[0]);receipt=dict(a,path=str(destination.resolve()),bytes=destination.stat().st_size,state='verified_available',completed_utc=now(),task_admitted_records=0,reader_support=False,original_partial_checkpoint=plan['original_partial_checkpoint'],immutable_original_bytes_reread_unchanged=True,reconstructed_from_frozen_partial=i==0,clone_receipt=str(receipt_dir/'immutable-partial-clone.json') if i==0 else None,**integrity);rp=receipt_dir/(a['id']+'.json');new(rp,receipt);done.append(dict(asset_id=a['id'],receipt=str(rp),bytes=a['expected_bytes'],sha256=receipt['sha256']));status('whole native source available after integrity gates');print('RNC_RECOVERY_SOURCE_VERIFIED',a['id'],a['expected_bytes'],flush=True)
        active=None;validate_checkpoint(cp,assets[0]);new(root/('handoff-'+job+'.json'),dict(completed_utc=now(),source='officialRNAcentral28.0',license='CC0-1.0',verified_source_assets=done,original_partial_control_unchanged=True,native_sequence_QC_pending=True,source_rows_not_counted_until_fullscan=True,task_admitted_records=0,reader_support=False));status('finite source recovery completed; fullnativeFASTAQCpending')
    except BaseException as e:
        child_state=None
        if child is not None and child.poll() is None:
            # Stop ONLY this new owned child; never signal old sources or model.
            pid=child.pid;p=pathlib.Path('/proc/'+str(pid)+'/stat')
            if p.exists() and int(p.read_text().split()[21])==child_start:
                child.terminate()
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=10)
                child_state=dict(pid=pid,start_ticks=child_start,stopped_new_owned_child=True,exit_code=child.returncode)
        status('stopped; all protected originals and new partials preserved',error=type(e).__name__+':'+str(e),new_child_stop=child_state);raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--approval',required=True);a=p.parse_args();run(a.plan,a.approval)
