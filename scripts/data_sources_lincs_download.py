"""Finite two-original public LINCS download. No auth/config reads, source writes exclusive."""
import argparse,concurrent.futures,datetime,email.utils,fcntl,hashlib,json,os,pathlib,signal,subprocess,threading,time,urllib.request
from data_sources_aria2 import aria2_command,final_sha256
from data_sources_aria2_run import console_progress,sockets
from data_sources_mmseqs_createdb import wait_process_identity,process_identity
from data_sources_workflow_guard import _wait_empty_session
from data_sources_mmseqs_supervisor import spawn_gated_group,release_gate

NOW=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for data in iter(lambda:f.read(8*1024**2),b''):h.update(data)
    return h.hexdigest()
def write_new(p,d):
    with pathlib.Path(p).open('x') as f:json.dump(d,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def mutable(p,d):
    p=pathlib.Path(p);tmp=p.with_name(p.name+'.writing-'+str(os.getpid()))
    with tmp.open('w') as f:json.dump(d,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def command(a):
    asset=dict(url=a['source_url'],expected_etag=a['etag'],expected_bytes=a['remote_file_bytes'])
    connections=8 if a['new_asset_id'].endswith('-CP') else 4
    cmd=aria2_command(asset,a['destination'],connections=connections,min_split='32M',max_tries=4)
    cmd[-1:-1]=['--no-netrc=true','--netrc-path=/dev/null','--check-certificate=true','--human-readable=false']
    return cmd
def check_entity(headers,a):
    stamp=email.utils.parsedate_to_datetime(headers['Last-Modified'])
    expected=datetime.datetime.fromisoformat(a['last_modified'].replace('Z','+00:00'))
    if headers['ETag']!=a['etag'] or a['etag'].startswith('W/') or int(headers['Content-Length'])!=a['remote_file_bytes'] or stamp!=expected:raise ValueError('exact source URL/entity/length/date drift')
def verify_footer(path,a):
    p=pathlib.Path(path);footer=a['captured_footer']
    if p.stat().st_size!=a['remote_file_bytes']:raise ValueError('source length mismatch')
    with p.open('rb') as f:
        if f.read(4)!=b'PAR1':raise ValueError('source is not native Parquet')
        f.seek(footer['range_start']);data=f.read(footer['range_stop']-footer['range_start'])
    if hashlib.sha256(data).hexdigest()!=footer['range_sha256']:raise ValueError('captured original footer mismatch')
def stop_only_owned(proc,identity):
    actual=process_identity(proc.pid)
    if actual is None:return
    if actual!=identity or actual['pgid']!=proc.pid:raise ValueError('owned downloader identity changed; no signals')
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if process_identity(proc.pid)!=identity:raise ValueError('changed downloader identity; refuse kill')
        os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
    _wait_empty_session(proc.pid,timeout=5)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);args=parser.parse_args();pp=pathlib.Path(args.plan).resolve();B=pp.parent;p=json.loads(pp.read_text());assets=p['assets'];budget=json.loads(pathlib.Path(p['budget_receipt']).read_text())
    if digest(p['budget_receipt'])!=p['budget_receipt_sha256'] or p['raw_cap_bytes']!=512*1024**2 or p['derived_cap_bytes']!=2*1024**3 or len(assets)!=2:raise ValueError('exact two-asset registered scope required')
    if sum(a['remote_file_bytes'] for a in assets)!=474854042 or {a['new_asset_id'] for a in assets}!={'DSE-LINCS-Pilot1-CP','DSE-LINCS-Pilot1-L1000'}:raise ValueError('different release scope refused')
    rights=json.loads(pathlib.Path(p['rights_and_entity_preflight']).read_text())
    if rights['data_license']!='CC0-1.0':raise ValueError('substantive original data rights required')
    if any(pathlib.Path(a['destination']).exists() or pathlib.Path(a['destination']+'.aria2').exists() for a in assets):raise FileExistsError('new immutable namespace only; prior partials must not reset')
    for a in assets:
        if not pathlib.Path(a['destination']).resolve().is_relative_to((B/'raw').resolve()) or not a['source_url'].startswith('https://cellpainting-gallery.s3.amazonaws.com/cpg0003-rosetta/broad/workspace/curated_preprocessed_data/LINCS-Pilot1/'):raise ValueError('unapproved source or output')
    lockfile=B/'audits/two-original-assets.lease'
    with lockfile.open('x+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);json.dump(dict(wrapper_pid=os.getpid(),plan_sha256=digest(pp),created_utc=NOW()),lease);lease.flush();os.fsync(lease.fileno())
        mutex=threading.Lock();active={};done=[];failures=[];start=time.monotonic()
        def status(phase):
            mutable(B/'live-acquisition.json',dict(updated_utc=NOW(),wrapper_pid=os.getpid(),phase=phase,active=active,raw_verified=done,failures=failures,elapsed_seconds=time.monotonic()-start,source_payload_bytes=474854042,sparse_file_size_is_not_transfer_progress=True,segment_bounds=dict(CP=8,L1000=4,host_max=16,global_max=32),full_row_QC=False,task_admitted_records=0))
        def raw_used():
            return sum(f.stat().st_size for directory in ['raw','research','logs','audits'] for f in (B/directory).rglob('*') if f.is_file() and not f.is_symlink())
        def transfer(a):
            identity=None;proc=None;state_path=None
            try:
                with (B/'audits'/(a['new_asset_id']+'.lease')).open('x') as f:f.write(str(os.getpid())+'\n')
                with urllib.request.urlopen(urllib.request.Request(a['source_url'],method='HEAD',headers={'If-Match':a['etag'],'User-Agent':'Mora3-source-acquisition/1.0'}),timeout=30) as response:check_entity(response.headers,a)
                fs=os.statvfs(B)
                if fs.f_bavail*fs.f_frsize<p['minimum_free_bytes']+p['raw_cap_bytes']+p['derived_cap_bytes']:raise ValueError('free reserve unavailable')
                cmd=command(a);log=B/'logs'/('aria2-'+a['new_asset_id']+'.log');begin=time.monotonic()
                control=B/'audits'/('transfer-control-'+a['new_asset_id']);control.mkdir(exist_ok=False);state_path=control/'supervisor-state.json'
                spec=dict(command=cmd,cwd=str(control),state_path=str(state_path),gate_timeout=5,guard=dict(stage_cap_bytes=p['raw_cap_bytes'],stop_margin_bytes=16*1024**2,minimum_free_bytes=p['minimum_free_bytes'],max_seconds=p['finite_download_seconds']),started_monotonic=start,address_space_limit_bytes=1024**3,threads=1)
                with log.open('xb') as stream:proc,identity,owned_command=spawn_gated_group(spec,stdout=stream)
                release_gate(proc)
                write_new(B/'audits'/(a['new_asset_id']+'-launch.json'),dict(at_utc=NOW(),identity=identity,supervisor_command=owned_command,native_command=cmd,source_URL=a['source_url'],expected_entity=a['etag'],log=str(log)))
                while proc.poll() is None:
                    fs=os.statvfs(B)
                    if time.monotonic()-start>p['finite_download_seconds'] or fs.f_bavail*fs.f_frsize<p['minimum_free_bytes'] or raw_used()>p['raw_cap_bytes']:
                        stop_only_owned(proc,identity);raise ValueError('finite/raw/free source guard; all partials retained')
                    with mutex:
                        state=json.loads(state_path.read_text()) if state_path.exists() else {};native_pid=state.get('native_pid')
                        active[a['new_asset_id']]=dict(supervisor_identity=identity,native_pid=native_pid,log=str(log),console=console_progress(log),established_connections=sockets(native_pid) if native_pid else None);status('parallel original source transfer')
                    time.sleep(1)
                elapsed=time.monotonic()-begin
                if proc.returncode:raise ValueError('aria2 finite exit '+str(proc.returncode)+'; no reset or wrapper retry')
                proc.stdin.close();_wait_empty_session(proc.pid,timeout=5);verify_footer(a['destination'],a)
                sha=final_sha256(a['destination'],dict(expected_bytes=a['remote_file_bytes']))
                with urllib.request.urlopen(urllib.request.Request(a['source_url'],method='HEAD',headers={'If-Match':a['etag']}),timeout=30) as response:check_entity(response.headers,a)
                rec=dict(source_asset=a,path=a['destination'],bytes=a['remote_file_bytes'],sha256=sha,opaque_multipart_ETag_not_assumed_whole_MD5=True,captured_footer_exact=True,source_entity_before_after_pass=True,transfer_seconds_including_startup=elapsed,mean_payload_bytes_per_second=a['remote_file_bytes']/elapsed,full_row_QC=False,task_admitted_records=0,completed_utc=NOW())
                write_new(B/'audits'/(a['new_asset_id']+'-verified-raw.json'),rec)
                with mutex:done.append(rec);active.pop(a['new_asset_id'],None);status('original raw verified; full native QC pending')
                print('LINCS_RAW_VERIFIED',a['new_asset_id'],a['remote_file_bytes'],sha,flush=True)
            except BaseException as exc:
                if proc is not None and proc.poll() is None and identity is not None:stop_only_owned(proc,identity)
                failure=dict(asset=a['new_asset_id'],error=type(exc).__name__+':'+str(exc),all_source_partials_retained=True,at_utc=NOW())
                write_new(B/'audits'/(a['new_asset_id']+'-failure.json'),failure)
                with mutex:failures.append(failure);active.pop(a['new_asset_id'],None);status('source failure quarantined; no automatic restart')
                raise
        status('registered exact two-source acquisition starting')
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(transfer,a) for a in assets]
            for future in futures:future.result()
        write_new(B/'raw-acquisition-manifest.json',dict(schema='mora3.LINCS_Pilot1_two_original_raw_sources.v1',completed_utc=NOW(),plan=str(pp),plan_sha256=digest(pp),data_license='CC0-1.0',rights_evidence=p['rights_and_entity_preflight'],assets=done,raw_bytes=sum(r['bytes'] for r in done),available_metadata_rows=80060,full_row_QC=False,task_admitted_records=0))
        status('both raw original assets verified; full native QC pending')
        print('LINCS_TWO_ORIGINAL_RAW_COMPLETE',flush=True)
if __name__=='__main__':main()
