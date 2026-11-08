"""Finite 1M source resource pilot; exact review required, no full MINT reconstruction."""
import argparse,datetime,fcntl,gzip,hashlib,importlib.metadata,json,os,pathlib,platform,resource,subprocess,sys,time
from data_sources_mmseqs_createdb import guard_decision,verify_native_database
from data_sources_workflow_guard import spawn_workflow,release_workflow,stop_workflow,workflow_identity,guarded_tree_bytes,_command,_active_group,_session_writers,_wait_empty_session
from data_sources_mint_pilot import write_sample,audit_sample,audit_cluster_tsv,pilot_commands,_refs

ROOT=pathlib.Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/data-sources/20261031-legacy-mint-v1'
CODES=('scripts/data_sources_mmseqs_createdb.py','scripts/data_sources_workflow_guard.py','scripts/data_sources_mint_pilot.py','scripts/data_sources_mint_pilot_run.py')
NOW=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()

def digest(path,on_progress=None):
    h=hashlib.sha256();n=last=0
    with pathlib.Path(path).open('rb') as f:
        for data in iter(lambda:f.read(8*1024**2),b''):
            h.update(data);n+=len(data)
            if on_progress and n-last>=1024**3:on_progress(n);last=n
    return h.hexdigest()

def new_json(p,obj):
    with pathlib.Path(p).open('x') as f:json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())

def mutable_json(p,obj):
    p=pathlib.Path(p);q=p.with_name(p.name+'.writing-'+str(os.getpid()))
    with q.open('w') as f:json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(q,p)

def verify_approval(a,plan_sha,codes):
    if a.get('approved') is not True or a.get('action')!='MINT_1M_native_resource_pilot_v1' or a.get('plan_sha256')!=plan_sha or a.get('code_sha256')!=codes:raise ValueError('exact 1M pilot plan/code execution review required')

def validate_limits(p):
    expected=dict(source_rows=59309604,selected_rows=1000000,stage_cap_bytes=8*1024**3,stop_margin_bytes=1024**3,minimum_free_bytes=200*1024**3,max_seconds=7200,threads=2,address_space_limit_bytes=16*1024**3,maximum_selected_expanded_bytes=1024**3)
    if any(p.get(k)!=v for k,v in expected.items()):raise ValueError('exact reserved 1M/8GiB/2CPU/2h/16GiBAS source scope required')

def require_guarded_worker(plan):
    parent=os.getppid();group=os.getpgrp()
    if parent!=group or os.getsid(0)!=group:raise ValueError('worker requires a new verified isolated supervisor session')
    identity=workflow_identity(parent)
    if identity is None or identity.get('pid')!=group or identity.get('pgid')!=group or identity.get('sid')!=group or not identity['argv'] or identity['argv'][-2]!='supervisor':raise ValueError('worker is outside source guard')
    job=json.loads(identity['argv'][-1])
    if identity['argv']!=_command('supervisor',job) or job['cwd']!=plan['new_output']:raise ValueError('exact owned source guard identity required')
    if job['guard']!={k:plan[k] for k in ['stage_cap_bytes','stop_margin_bytes','minimum_free_bytes','max_seconds']} or job['address_space_limit_bytes']!=plan['address_space_limit_bytes']:raise ValueError('guard resource envelope differs from plan')
    if job['source_symlink_targets']:raise ValueError('pilot uses self-contained selected DB; no external source links needed')
    if job['command']!=[sys.executable,'-B','-u',str(pathlib.Path(__file__).resolve()),'--worker','--plan',plan['_plan_path'],'--approval',plan['_approval_path']]:raise ValueError('worker command changed')
    return job

def budget_reason(used,free,elapsed,incoming,p):
    return guard_decision(used+incoming,free-incoming,elapsed,p)

def available_memory():
    return int(next(l.split()[1] for l in pathlib.Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024

def checked(args):
    pp=pathlib.Path(args.plan).resolve();ap=pathlib.Path(args.approval).resolve();p=json.loads(pp.read_text());a=json.loads(ap.read_text());codes={s:digest(ROOT/s) for s in CODES};verify_approval(a,digest(pp),codes)
    if p['code_sha256']!=codes:raise ValueError('pilot code changed')
    validate_limits(p)
    if not sys.dont_write_bytecode or platform.python_version()!=p['python_version'] or importlib.metadata.version('pyarrow')!=p['pyarrow_version']:raise ValueError('exact existing runtime -B required')
    if digest(p['budget_receipt'])!=p['budget_receipt_sha256']:raise ValueError('pilot budget receipt changed')
    budget=json.loads(pathlib.Path(p['budget_receipt']).read_text())
    if budget['STRING_family_allocation_after_bytes']!=64*1024**3 or budget['aggregate_derived_registered_after_bytes']!=154*1024**3:raise ValueError('reserved ledger differs')
    if p['source_sha256']!='210382afd5136897cde16464f76f41857c82123707dd1452236e3d4f9d0177c4' or pathlib.Path(p['source_path']).stat().st_size!=13003020999:raise ValueError('exact full provider FASTA required')
    if pathlib.Path(p['new_output']).resolve()!=BASE/'curated/MINT-1M-native-resource-pilot-v1':raise ValueError('fresh exact source-pilot namespace required')
    if digest(p['binary'])!=p['binary_sha256'] or pathlib.Path(p['binary']).stat().st_mode&0o6000:raise ValueError('exact nonprivileged native binary required')
    if subprocess.run([p['binary'],'version'],capture_output=True,text=True,check=True,timeout=15).stdout.strip()!=p['binary_version']:raise ValueError('native binary version changed')
    p.update(_plan_path=str(pp),_approval_path=str(ap));return p,codes

def _worker(p,codes):
    job=require_guarded_worker(p);D=pathlib.Path(p['new_output']);source=pathlib.Path(p['source_path']);progress={};pending=0
    def live(phase,**more):mutable_json(D/'worker-live.json',dict(updated_utc=NOW(),pipeline_pid=os.getpid(),phase=phase,progress=progress,**more))
    def check(extra=0):
        used=guarded_tree_bytes(D,[]);s=os.statvfs(D);reason=budget_reason(used,s.f_bavail*s.f_frsize,time.monotonic()-job['started_monotonic'],extra,p)
        if reason:raise ValueError('finite source-pilot '+reason+'; all partials preserved')
        if available_memory()<16*1024**3:raise ValueError('available RAM reserve reached; source pilot stops')
    def prewrite(n):
        nonlocal pending
        if n>2**20:raise ValueError('unexpected oversized source record; no truncation')
        pending+=n
        if pending>=2**20:check(pending);pending=0
    def emit(name,obj):
        data=(json.dumps(obj,indent=2)+'\n').encode();check(len(data));new_json(D/name,obj)
    live('full source SHA preflight')
    if digest(source,lambda n:(progress.update(raw_bytes_SHA_checked=n),live('full source SHA preflight')))!=p['source_sha256']:raise ValueError('immutable source SHA mismatch')
    live('stream full original source; retain deterministic 1M full-axis source sample')
    sample=write_sample(source,D,p['source_rows'],p['selected_rows'],prewrite)
    with gzip.open(D/'selected-source.fasta.gz','rb') as f:
        expanded=0
        for chunk in iter(lambda:f.read(8*1024**2),b''):expanded+=len(chunk);check()
    if expanded>p['maximum_selected_expanded_bytes']:raise ValueError('selected self-contained logical DB input exceeds1GiB AS rationale; no smaller cohort substituted')
    sample['expanded_FASTA_bytes']=expanded;emit('sample-source-manifest.json',sample)
    live('independent all-original-source versus selected sample replay')
    sample_proof=audit_sample(source,D,p['source_rows'],p['selected_rows']);emit('sample-full-source-replay.json',sample_proof)
    if digest(source)!=p['source_sha256']:raise ValueError('immutable source changed during selection')
    stages=[];native_db=None;cluster=None
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',RUNNER='',MPI_RUNNER='',TMPDIR=str(D/'tmp'))
    for name,argv in pilot_commands(p['binary'],D):
        check();live('native '+name,requested_native_command=argv);start=time.monotonic()
        with (D/(name+'.log')).open('xb') as log:
            child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,cwd=D,env=env)
            live('native '+name,requested_native_command=argv,requested_native_pid=child.pid);code=child.wait()
        rec=dict(stage=name,argv=argv,seconds=time.monotonic()-start,exit_code=code,stage_bytes=guarded_tree_bytes(D,[]),max_children_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss);stages.append(rec);emit(name+'-stage-receipt.json',rec)
        if code:raise ValueError('finite native '+name+' failed; no reset/retry')
        if name=='createdb':
            live('independent selected native DB full-header/AA/lookup proof')
            native_db=verify_native_database(D/'selected-source.fasta.gz',D/'DB100',p['selected_rows'],32,on_progress=lambda n,t,phase:(check(),progress.update(native_proof_phase=phase,native_records=n,total=t),live('selected native DB full proof')))
            emit('selected-native-DB-proof.json',native_db)
    live('whole native cluster member/representative census')
    ids={r['provider_id'] for r in _refs(D/'selected-source-occurrence-refs.jsonl')}
    if len(ids)!=p['selected_rows']:raise ValueError('source membership IDs ambiguous')
    cluster=audit_cluster_tsv(D/'clu50.tsv',ids);emit('cluster-full-member-census.json',cluster)
    if digest(source)!=p['source_sha256']:raise ValueError('source changed after pilot')
    emit('pilot-native-result.json',dict(schema='mora3.MINT_1M_native_resource_pilot_result.v1',completed_utc=NOW(),source_total=p['source_rows'],sample=sample,sample_full_source_proof=sample_proof,native_DB_proof=native_db,cluster=cluster,stages=stages,raw_source_SHA_before_after_pass=True,code_sha256=codes,native_address_space_limit_bytes=p['address_space_limit_bytes'],selected_DB_self_contained_full_provider_DB_not_mapped=True,split_memory_setting_not_peak_RSS_guarantee=True,source_original_ordinals_and_full_DB_keys_preserved=True,full59M_clustering=False,full_MINT_reconstructed=False,pair_filter_shuffle_split_performed=False,protein_annotations_inherited=False,task_admitted_records=0))
    live('pipeline complete; outer wrapper final hashes pending')

def cleanup_owned_workflow(child,owners,checkpoint,settle_seconds=5):
    error=None
    if owners is not None:
        try:stop_workflow(owners,checkpoint)
        except BaseException as exc:error=exc
    if child is not None and child.poll() is None:
        try:child.wait(timeout=15)
        except subprocess.TimeoutExpired as exc:error=exc
    if owners is not None:
        _wait_empty_session(owners['leader']['pid'],timeout=settle_seconds)
    return error

def _outer(p,codes):
    D=pathlib.Path(p['new_output']);lease=BASE/'audits/MINT-1M-native-resource-pilot-v1.lease'
    if D.exists() or lease.exists():raise ValueError('exclusive fresh namespace/lease required; no resume/reset')
    fs=os.statvfs(BASE)
    if fs.f_bavail*fs.f_frsize<p['minimum_free_bytes']+p['stage_cap_bytes'] or available_memory()<32*1024**3:raise ValueError('reserved free disk/RAM unavailable')
    with lease.open('x+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);json.dump(dict(created_utc=NOW(),wrapper=workflow_identity(os.getpid()),plan_sha256=digest(p['_plan_path'])),lock);lock.flush();os.fsync(lock.fileno())
        D.mkdir(exist_ok=False);(D/'tmp').mkdir();start=time.monotonic();peak=0;child=None;owners=None
        def live(phase,**more):mutable_json(D/'outer-live.json',dict(updated_utc=NOW(),phase=phase,wrapper_pid=os.getpid(),owners=owners,elapsed_seconds=time.monotonic()-start,peak_stage_bytes_observed=peak,source_connections=0,**more))
        def checkpoint(reason):
            if owners is None:
                path=D/'startup-failure-unverified.json'
                if not path.exists():new_json(path,dict(at_utc=NOW(),reason=reason,native_START_never_authorized_by_wrapper=True,stable_partial_hashes=False,task_admitted_records=0))
                return str(path)
            session=owners['leader']['pid'];before=_wait_empty_session(session,timeout=5)
            target=D/'stopped-partial-checkpoint.json'
            if target.exists():
                previous=json.loads(target.read_text())
                if previous.get('stable_partial_hashes') is not True or previous.get('session_id')!=session:raise ValueError('existing partial receipt is not a verified empty-session checkpoint')
                return str(target)
            files=[]
            for f in sorted(D.rglob('*')):
                if f.is_symlink():
                    files.append(dict(path=str(f),symlink_raw_target=str(f.readlink()),link_bytes=f.lstat().st_size));continue
                if f.is_file() and f.name not in ['outer-live.json','worker-live.json']:
                    with f.open('rb') as stream:os.fsync(stream.fileno())
                    files.append(dict(path=str(f),bytes=f.stat().st_size,sha256=digest(f)))
            after=_wait_empty_session(session,timeout=5)
            new_json(target,dict(at_utc=NOW(),reason=reason,source_original_not_written=True,all_new_partials_retained=True,
                stable_partial_hashes=True,session_id=session,empty_session_before_hashes=before,empty_session_after_hashes=after,
                no_pre_kill_hash_snapshot_promoted=True,files=files))
            return str(target)
        try:
            native=[sys.executable,'-B','-u',str(pathlib.Path(__file__).resolve()),'--worker','--plan',p['_plan_path'],'--approval',p['_approval_path']]
            spec=dict(command=native,cwd=str(D),gate_timeout=5,guard={k:p[k] for k in ['stage_cap_bytes','stop_margin_bytes','minimum_free_bytes','max_seconds']},started_monotonic=start,address_space_limit_bytes=p['address_space_limit_bytes'],source_symlink_targets=[])
            with (D/'workflow.log').open('xb') as log:child,owners=spawn_workflow(spec,stdout=log)
            new_json(D/'verified-owners-before-native-release.json',dict(at_utc=NOW(),owners=owners,native_started=False,plan_sha256=digest(p['_plan_path'])))
            release_workflow(child);last=0
            while child.poll() is None:
                used=guarded_tree_bytes(D,[]);peak=max(peak,used);fs=os.statvfs(D);reason=guard_decision(used,fs.f_bavail*fs.f_frsize,time.monotonic()-start,p)
                if available_memory()<16*1024**3:reason='available_RAM_reserve'
                if reason:
                    live('guard freezing only verified isolated source workflow',reason=reason)
                    stop_workflow(owners,lambda:checkpoint(reason));child.wait(timeout=15);raise ValueError('finite pilot stopped; all partials retained')
                if time.monotonic()-last>=2:live('finite guarded 1M source/native pilot active',stage_bytes=used);last=time.monotonic()
                time.sleep(.05)
            if child.returncode:raise ValueError('source workflow returned '+str(child.returncode)+'; no retry/reset')
            child.stdin.close();peak=max(peak,guarded_tree_bytes(D,[]))
            result=json.loads((D/'pilot-native-result.json').read_text())
            if result['sample_full_source_proof']['selected_source_records']!=1000000:raise ValueError('full planned source cohort proof missing')
            files=[]
            for f in sorted(D.rglob('*')):
                if f.is_symlink():
                    target=f.resolve(strict=False)
                    if not target.is_relative_to(D):raise ValueError('unexpected external link at completion')
                    files.append(dict(path=str(f),symlink_target=str(target),symlink_raw_target=str(f.readlink()),link_bytes=f.lstat().st_size))
                elif f.is_file() and f.name not in ['outer-live.json','worker-live.json']:files.append(dict(path=str(f),bytes=f.stat().st_size,sha256=digest(f)))
            if guarded_tree_bytes(D,[])>=p['stage_cap_bytes']:raise ValueError('final stage cap exceeded')
            new_json(D/'manifest.json',dict(schema='mora3.MINT_bounded_native_resource_pilot_verified.v1',completed_utc=NOW(),plan=p['_plan_path'],plan_sha256=digest(p['_plan_path']),approval=p['_approval_path'],approval_sha256=digest(p['_approval_path']),code_sha256=codes,source_SHA256=p['source_sha256'],selected_records=1000000,pilot_result=result,all_retained_files=files,peak_stage_bytes_observed=peak,retained_bytes_before_manifest=guarded_tree_bytes(D,[]),max_child_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,elapsed_seconds=time.monotonic()-start,all_source_data_unchanged=True,old_StageB_32GiB_reservation_not_refunded=True,full59M_clustering_or_fullMINT_reconstruction=False,pair_shuffle_filter_split=False,task_admitted_records=0))
            live('complete 1M resource pilot with exact source/native/cluster member proof; full source clustering unperformed')
            print('MINT_1M_RESOURCE_PILOT_COMPLETE',flush=True)
        except BaseException as exc:
            if child is not None and child.stdin and not child.stdin.closed:
                try:child.stdin.close()
                except OSError:pass
            try:cleanup_error=cleanup_owned_workflow(child,owners,lambda:checkpoint('wrapper exception'))
            except BaseException as cleanup_exc:
                live('failed; group liveness unresolved; no stable checkpoint promoted',error=type(exc).__name__+':'+str(exc),cleanup_error=type(cleanup_exc).__name__+':'+str(cleanup_exc));raise
            if not (D/'stopped-partial-checkpoint.json').exists():checkpoint('failure; verified session empty; source and partials retained')
            live('stopped; no reset or native retry; all source/partials preserved',error=type(exc).__name__+':'+str(exc),cleanup_error=str(cleanup_error) if cleanup_error else None);raise

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--approval',required=True);parser.add_argument('--worker',action='store_true');args=parser.parse_args();os.chdir(ROOT)
    plan,codes=checked(args)
    if args.worker:_worker(plan,codes)
    else:_outer(plan,codes)

if __name__=='__main__':main()
