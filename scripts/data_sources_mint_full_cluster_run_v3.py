"""Full-provider cluster-only hardware-policyv3 runner; exact root execution approval required.
No pair/filter/permutation/split implementation; original inputs and old jobs untouched.
"""
import argparse,datetime,fcntl,hashlib,json,os,pathlib,platform,resource,subprocess,sys,time
from data_sources_hardware_storage import disk_reason,hardware_snapshot,load_policy
from data_sources_mint_hardware_inventory import stream_inventory
from data_sources_mint_full_guard_v2 import spawn_workflow,release_workflow,stop_workflow,workflow_identity,guarded_tree_bytes,_command,_wait_empty_session
from data_sources_mint_full_resources import pressure_snapshot,pressure_reason,check_affinity
from data_sources_mint_full_cluster import build_source_keys,audit_membership,cluster_commands
from data_sources_mint_saved_keys import copy_verified_source_keys
ROOT=pathlib.Path(__file__).resolve().parents[1];BASE=ROOT/'outputs/data-sources/20261031-legacy-mint-v1';G=1024**3
CODES=('scripts/data_sources_mmseqs_createdb.py','scripts/data_sources_mint_full_resources.py','scripts/data_sources_hardware_storage.py','scripts/data_sources_mint_hardware_inventory.py','scripts/data_sources_mint_full_guard_v2.py','scripts/data_sources_mint_full_cluster.py','scripts/data_sources_mint_saved_keys.py','scripts/data_sources_mint_full_cluster_run_v3.py')
NOW=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
NAMES={'DB100','DB100.dbtype','DB100.index','DB100.lookup','DB100.source','DB100_h','DB100_h.dbtype','DB100_h.index'}
def digest(p,on_progress=None,deadline=None):
    h=hashlib.sha256();n=last=0
    with pathlib.Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):
            if deadline is not None and time.monotonic()>deadline:raise ValueError('finite whole-wrapper deadline during stable hash; no receipt promotion')
            h.update(b);n+=len(b)
            if on_progress and n-last>=G:on_progress(n);last=n
    return h.hexdigest()
def new_json(p,d):
    p=pathlib.Path(p);data=(json.dumps(d,indent=2)+'\n').encode()
    if disk_reason(hardware_snapshot(p.parent)['available_bytes'],0,dict(minimum_free_bytes=64*G,max_seconds=172800),incoming_bytes=len(data)):raise ValueError('hardware free-space buffer before immutable JSON creation')
    with p.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
def mutable_json(p,d):
    p=pathlib.Path(p);t=p.with_name(p.name+'.writing-'+str(os.getpid()))
    with t.open('w') as f:json.dump(d,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(t,p)
def verify_approval(a,plan_sha,codes,policy_sha):
    if a.get('approved') is not True or a.get('action')!='MINT_full_provider_cluster_only_hardware_v3' or a.get('plan_sha256')!=plan_sha or a.get('code_sha256')!=codes or a.get('policy_sha256')!=policy_sha:raise ValueError('NEW exact full-cluster-only execution approval required')
def validate_limits(p):
    if any(k in p for k in ['stage_cap_bytes','stop_margin_bytes','raw_cap_bytes','derived_cap_bytes']):raise ValueError('superseded artificial storage allocation in new plan')
    exact=dict(source_rows=59309604,minimum_free_bytes=64*G,max_seconds=172800,threads=4,address_space_limit_bytes=64*G,available_RAM_start_minimum_bytes=40*G,checkpoint_reserve_seconds=1800)
    memory=dict(max_session_pss_bytes=32*G,session_pss_stop_margin_bytes=8*G,min_host_available_bytes=16*G,max_session_swap_bytes=256*1024**2)
    if any(p.get(k)!=v for k,v in exact.items()) or p.get('physical_memory_policy')!=memory:raise ValueError('exact proposed cluster-only source/resource envelope required')
    if p.get('cpu_affinity')!=[28,29,30,31]:raise ValueError('exact four-CPU proposal required; no other-job affinity changes')
def validate_component_map(p,m):
    components=p['input_components'];prefix=pathlib.Path(p['source_db_prefix']);expected={x['path']:x for x in m['all_stage_files'] if pathlib.Path(x['path']).name in NAMES}
    if m['source_rows']!=59309604 or m['source_original_whole_SHA_before_after_pass'] is not True or len(components)!=8 or {pathlib.Path(x['path']).name for x in components}!=NAMES:raise ValueError('exact whole verified native source component allowlist required')
    if len({x['path'] for x in components})!=8 or any(pathlib.Path(x['path']).parent!=prefix.parent or expected.get(x['path'])!=x for x in components):raise ValueError('input component/path/hash differs from immutable whole-source proof')
def make_aliases(p,D):
    inputdir=pathlib.Path(D)/'input';inputdir.mkdir(exist_ok=False)
    for a in p['input_components']:
        source=pathlib.Path(a['path'])
        if source.is_symlink() or not source.is_file() or source.stat().st_size!=a['bytes']:raise ValueError('input source component is not original regular file/length')
        (inputdir/source.name).symlink_to(source)

def file_refs(D,allowed,deadline=None,inventory_name='retained-file-inventory.jsonl'):
    # Native SID is empty before finalization; wrapper memory and host pressure still apply.
    policy=dict(max_session_pss_bytes=32*G,session_pss_stop_margin_bytes=8*G,min_host_available_bytes=16*G,max_session_swap_bytes=256*1024**2);last=[None]
    def poll():
        now=time.monotonic()
        if last[0] is None or now-last[0]>=0.5:
            reason=pressure_reason(pressure_snapshot(os.getsid(0)),policy);last[0]=now
            if reason:raise ValueError('final inventory physical memory guard '+reason+'; no stable receipt promotion')
    poll()
    return stream_inventory(D,allowed,pathlib.Path(D)/inventory_name,deadline,minimum_free_bytes=64*G,on_poll=poll)

def stable_checkpoint(D,owners,allowed,reason,deadline=None):
    D=pathlib.Path(D)
    if owners is None:
        p=D/'startup-failure-unverified.json'
        if not p.exists():new_json(p,dict(created_utc=NOW(),reason=reason,native_START_not_authorized=True,stable_partial_hashes=False))
        return str(p)
    sid=owners['leader']['pid'];before=_wait_empty_session(sid,timeout=5);target=D/'stopped-partial-checkpoint.json'
    if target.exists():
        old=json.loads(target.read_text())
        if old.get('stable_partial_hashes') is not True or old.get('session_id')!=sid:raise ValueError('existing receipt is not exact empty-session stable checkpoint')
        return str(target)
    refs=file_refs(D,allowed,deadline,inventory_name='stopped-partial-file-inventory.jsonl');after=_wait_empty_session(sid,timeout=5)
    new_json(target,dict(created_utc=NOW(),reason=reason,stable_partial_hashes=True,session_id=sid,empty_before=before,empty_after=after,retained_file_inventory=refs,wrapper_never_writes_original_sources=True,all_new_partials_retained=True,no_pre_kill_hash_snapshot_promoted=True));return str(target)

def checked(args):
    pp=pathlib.Path(args.plan).resolve();ap=pathlib.Path(args.approval).resolve();p=json.loads(pp.read_text());a=json.loads(ap.read_text());codes={n:digest(ROOT/n) for n in CODES};verify_approval(a,digest(pp),codes,p['execution_policy_sha256'])
    if p['code_sha256']!=codes:raise ValueError('reviewed code changed')
    validate_limits(p);check_affinity(p['cpu_affinity'])
    if not sys.dont_write_bytecode or platform.python_version()!=p['python_version']:raise ValueError('existing exact interpreter with -B required')
    if p['execution_policy_sha256']!='38b41a0eaf2c87b216795c43127237a954628894ef8825c1f6e99813bc4baa76':raise ValueError('exact latestUSERpolicy required; new policy needs reviewed continuation')
    load_policy(p['execution_policy'],p['execution_policy_sha256'])
    mp=pathlib.Path(p['source_native_proof']['path'])
    if digest(mp)!=p['source_native_proof']['sha256']:raise ValueError('prior full source proof changed')
    validate_component_map(p,json.loads(mp.read_text()))
    if pathlib.Path(p['source_db_prefix']).resolve()!=BASE/'curated/MINT-full-provider-createdb-compressed-v2/DB100':raise ValueError('exact immutable native provider DB required')
    if pathlib.Path(p['new_output']).resolve()!=BASE/'curated/MINT-full-provider-cluster-only-hardware-v3' or pathlib.Path(p['lease_path']).resolve()!=BASE/'audits/MINT-full-provider-cluster-only-hardware-v3.lease':raise ValueError('fresh exact future output/lease required')
    if p['source_sha256']!='210382afd5136897cde16464f76f41857c82123707dd1452236e3d4f9d0177c4' or pathlib.Path(p['source_path']).stat().st_size!=13003020999:raise ValueError('exact original full provider FASTA required')
    if digest(p['binary'])!=p['binary_sha256'] or pathlib.Path(p['binary']).stat().st_mode&0o6000:raise ValueError('pinned nonprivileged native binary required')
    if subprocess.run([p['binary'],'version'],capture_output=True,text=True,check=True,timeout=15).stdout.strip()!=p['binary_version']:raise ValueError('native version changed')
    p.update(_plan_path=str(pp),_approval_path=str(ap));return p,codes

def guard_limits(p):
    g={k:p[k] for k in ['minimum_free_bytes','max_seconds']};g['max_seconds']-=p['checkpoint_reserve_seconds'];return g

def require_guard(p):
    group=os.getpgrp();parent=os.getppid();identity=workflow_identity(parent)
    if parent!=group or os.getsid(0)!=group or identity is None or identity['pgid']!=group or identity['sid']!=group or identity['argv'][-2]!='supervisor':raise ValueError('exact isolated guard required')
    job=json.loads(identity['argv'][-1]);expected=[sys.executable,'-B','-u',str(pathlib.Path(__file__).resolve()),'--worker','--plan',p['_plan_path'],'--approval',p['_approval_path']]
    if identity['argv']!=_command('supervisor',job) or job['command']!=expected or job['cwd']!=p['new_output'] or job['source_symlink_targets']!=[x['path'] for x in p['input_components']] or job['guard']!=guard_limits(p) or job['physical_memory_policy']!=p['physical_memory_policy'] or job['cpu_affinity']!=p['cpu_affinity'] or job['address_space_limit_bytes']!=p['address_space_limit_bytes'] or job.get('pressure_sample_seconds')!=0.5 or job.get('gate_timeout')!=5:raise ValueError('guard/source/resource policy differs')
    if os.sched_getaffinity(0)!=set(p['cpu_affinity']) or resource.getrlimit(resource.RLIMIT_AS)!=(p['address_space_limit_bytes'],p['address_space_limit_bytes']) or os.getpriority(os.PRIO_PROCESS,0)<10:raise ValueError('worker actual affinity/AS/nice differs')
    return job

def native_environment(p,base=None):
    D=pathlib.Path(p['new_output']);env=dict(os.environ if base is None else base)
    # Pinned MMseqs refuses pre-set OMP_PROC_BIND; kernel CPU affinity remains the guard.
    for key in ('OMP_PROC_BIND','OMP_PLACES'):env.pop(key,None)
    env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',ARROW_NUM_THREADS='1',RUNNER='',MPI_RUNNER='',TMPDIR=str(D/'tmp'),SQLITE_TMPDIR=str(D/'tmp'));return env

def _worker(p,codes):
    job=require_guard(p);D=pathlib.Path(p['new_output']);allowed=[a['path'] for a in p['input_components']];progress={}
    def live(phase,**kw):mutable_json(D/'worker-live.json',dict(updated_utc=NOW(),pipeline_pid=os.getpid(),phase=phase,progress=progress,**kw))
    def check():
        fs=os.statvfs(D);reason=disk_reason(fs.f_bavail*fs.f_frsize,time.monotonic()-job['started_monotonic'],guard_limits(p)) or pressure_reason(pressure_snapshot(os.getsid(0)),p['physical_memory_policy'])
        if reason:raise ValueError('finite cluster guard '+reason+'; all partials retained')
        guarded_tree_bytes(D,allowed)  # exact path allowlist/accounting only; no artificial stage cap
    def emit(n,v):
        check();data=(json.dumps(v,indent=2)+'\n').encode()
        if disk_reason(hardware_snapshot(D)['available_bytes'],time.monotonic()-job['started_monotonic'],guard_limits(p),incoming_bytes=len(data)):raise ValueError('physical hardware buffer before proof write')
        new_json(D/n,v)
    def inputs(phase):
        live(phase)
        for item in p['input_components']:
            f=pathlib.Path(item['path'])
            if f.is_symlink() or f.stat().st_size!=item['bytes'] or digest(f,lambda n:(check(),progress.update(input=str(f),SHA_bytes=n),live(phase)))!=item['sha256']:raise ValueError('original native input changed')
            check()
        if digest(p['source_path'],lambda n:(check(),progress.update(raw_source_SHA_bytes=n),live(phase)))!=p['source_sha256']:raise ValueError('original FASTA changed')
    inputs('whole immutable native input/source preflight');make_aliases(p,D)
    live('full provider uniqueness/native source ordinal proof')
    keys=copy_verified_source_keys(p['reuse_source_key_proof'],p,D/'source-keys-and-membership.sqlite',check);emit('source-key-proof.json',keys)
    stages=[];env=native_environment(p)
    for name,argv in cluster_commands(p['binary'],D):
        check();live('native '+name,requested_native_command=argv);start=time.monotonic()
        with (D/(name+'.log')).open('xb') as log:
            child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,cwd=D,env=env);live('native '+name,requested_native_command=argv,requested_native_pid=child.pid);code=child.wait()
        rec=dict(stage=name,argv=argv,seconds=time.monotonic()-start,exit_code=code,stage_bytes=guarded_tree_bytes(D,allowed),max_children_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss);stages.append(rec);emit(name+'-stage-receipt.json',rec)
        if code:raise ValueError('native '+name+' failed; no retry/reset or smaller-source substitution')
    live('whole membership, representative self-membership and source-key proof');members=audit_membership(D/'clu50.tsv',D/'source-keys-and-membership.sqlite',p['source_rows'],check);emit('whole-membership-proof.json',members)
    inputs('whole immutable input/source final hashes')
    emit('full-cluster-native-result.json',dict(schema='mora3.MINT_full_provider_cluster_only_result.v1',completed_utc=NOW(),source_rows=p['source_rows'],source_keys=keys,membership=members,stages=stages,original_native_DB_and_FASTA_whole_hashes_before_after_pass=True,source_native_proof=p['source_native_proof'],code_sha256=codes,per_process_virtual_AS_limit_bytes=p['address_space_limit_bytes'],physical_memory_policy=p['physical_memory_policy'],aggregate_memory_not_guaranteed_by_RLIMIT_AS=True,native_split_memory_setting_not_peak_RSS_guarantee=True,computed_clusters_not_original_MINT_artifact=True,pair_filter_shuffle_split_performed=False,protein_annotations_inherited=False,task_admitted_records=0));live('pipeline complete; outer empty-session/final-file hashes pending')

def _outer(p,codes):
    D=pathlib.Path(p['new_output']);lease=pathlib.Path(p['lease_path']);allowed=[a['path'] for a in p['input_components']]
    if D.exists() or lease.exists():raise ValueError('fresh exclusive namespace/lease required; no resume/reset')
    fs=os.statvfs(BASE);available=int(next(v.split()[1] for v in pathlib.Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:')))*1024
    if fs.f_bavail*fs.f_frsize<=p['minimum_free_bytes'] or available<p['available_RAM_start_minimum_bytes']:raise ValueError('measured hardware free buffer/host RAM unavailable')
    if os.getsid(0)!=os.getpid() or os.getpgrp()!=os.getpid():raise ValueError('outer launcher must also be its own isolated session/group')
    check_affinity(p['cpu_affinity']);os.sched_setaffinity(0,set(p['cpu_affinity']));os.nice(max(0,10-os.getpriority(os.PRIO_PROCESS,0)));resource.setrlimit(resource.RLIMIT_AS,(p['address_space_limit_bytes'],p['address_space_limit_bytes']));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    with lease.open('x+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);json.dump(dict(created_utc=NOW(),wrapper=workflow_identity(os.getpid()),plan_sha256=digest(p['_plan_path'])),lock);lock.flush();os.fsync(lock.fileno());D.mkdir(exist_ok=False);(D/'tmp').mkdir();new_json(D/'launch-hardware-capacity.json',hardware_snapshot(D,p['minimum_free_bytes']));start=time.monotonic();peak=0;max_pss=max_rss=0;child=None;owners=None
        def live(phase,**kw):mutable_json(D/'outer-live.json',dict(updated_utc=NOW(),phase=phase,wrapper_pid=os.getpid(),owners=owners,elapsed_seconds=time.monotonic()-start,peak_stage_bytes_observed=peak,max_owned_session_PSS_bytes_observed=max_pss,max_owned_session_RSS_bytes_observed=max_rss,source_connections=0,**kw))
        def checkpoint(reason):return stable_checkpoint(D,owners,allowed,reason,deadline=start+p['max_seconds'])
        try:
            native=[sys.executable,'-B','-u',str(pathlib.Path(__file__).resolve()),'--worker','--plan',p['_plan_path'],'--approval',p['_approval_path']]
            spec=dict(command=native,cwd=str(D),gate_timeout=5,guard=guard_limits(p),started_monotonic=start,address_space_limit_bytes=p['address_space_limit_bytes'],source_symlink_targets=allowed,physical_memory_policy=p['physical_memory_policy'],pressure_sample_seconds=0.5,cpu_affinity=p['cpu_affinity'])
            env=native_environment(p);env['PYTHONDONTWRITEBYTECODE']='1'
            with (D/'workflow.log').open('xb') as log:child,owners=spawn_workflow(spec,stdout=log,env=env)
            new_json(D/'verified-owners-before-native-release.json',dict(created_utc=NOW(),owners=owners,native_started=False,plan_sha256=digest(p['_plan_path'])));release_workflow(child)
            while child.poll() is None:
                used=guarded_tree_bytes(D,allowed);peak=max(peak,used);fs=os.statvfs(D);snapshot=pressure_snapshot(owners['leader']['pid']);max_pss=max(max_pss,snapshot['owned_session_pss_bytes']);max_rss=max(max_rss,snapshot['owned_session_rss_bytes']);reason=disk_reason(fs.f_bavail*fs.f_frsize,time.monotonic()-start,guard_limits(p)) or pressure_reason(snapshot,p['physical_memory_policy'])
                if reason:live('verified owned source workflow guard stop',reason=reason);stop_workflow(owners,lambda:checkpoint(reason));child.wait(timeout=15);raise ValueError('finite full cluster stopped; partials retained')
                live('finite full-provider cluster-only workflow active',stage_bytes=used,physical_memory=snapshot);time.sleep(0.5)
            if child.returncode:raise ValueError('full cluster source workflow returned '+str(child.returncode)+'; no retry/reset')
            child.stdin.close();_wait_empty_session(owners['leader']['pid'],timeout=5);peak=max(peak,guarded_tree_bytes(D,allowed));result=json.loads((D/'full-cluster-native-result.json').read_text())
            if result['membership']['members_once']!=59309604 or result['source_keys']['source_records']!=59309604:raise ValueError('whole source/member proof missing')
            refs=file_refs(D,allowed,deadline=start+p['max_seconds']);_wait_empty_session(owners['leader']['pid'],timeout=5)
            if hardware_snapshot(D)['available_bytes']<=p['minimum_free_bytes']:raise ValueError('hardware free buffer reached before final receipt')
            new_json(D/'manifest.json',dict(schema='mora3.MINT_full_provider_cluster_only_hardware_verified.v3',completed_utc=NOW(),plan=p['_plan_path'],plan_sha256=digest(p['_plan_path']),approval=p['_approval_path'],approval_sha256=digest(p['_approval_path']),code_sha256=codes,native_result=result,retained_file_inventory=refs,stable_hashes_after_verified_empty_session=True,peak_stage_bytes_observed=peak,retained_bytes_before_manifest=guarded_tree_bytes(D,allowed),max_owned_session_PSS_bytes_observed=max_pss,max_owned_session_RSS_bytes_observed=max_rss,per_process_virtual_AS_bytes=p['address_space_limit_bytes'],elapsed_seconds=time.monotonic()-start,historical_receipts_and_all_prior_artifacts_preserved=True,artificial_storage_caps_superseded=True,governing_execution_policy=p['execution_policy'],policy_sha256=p['execution_policy_sha256'],final_hardware_snapshot=hardware_snapshot(D),training_performed=False,original_MINT_clu50_identity_unproven=True,pair_filter_permutation_split_performed=False,task_admitted_records=0));live('full source cluster-only proof complete; original-MINT equivalence/pair stage/admission not asserted')
        except BaseException as exc:
            if child is not None and child.stdin and not child.stdin.closed:
                try:child.stdin.close()
                except OSError:pass
            try:
                if owners is not None:stop_workflow(owners,lambda:checkpoint('wrapper failure'))
                if child is not None and child.poll() is None:child.wait(timeout=15)
                if owners is not None:_wait_empty_session(owners['leader']['pid'],timeout=5)
                checkpoint('failure; owned session empty or pre-native unverified startup')
            except BaseException as cleanup:
                live('failed checkpoint or ownership settlement; no stable hash promotion',error=type(exc).__name__+':'+str(exc),cleanup_error=str(cleanup));raise
            live('stopped; source and all new partials retained; no retry/reset',error=type(exc).__name__+':'+str(exc));raise

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--approval',required=True);parser.add_argument('--worker',action='store_true');args=parser.parse_args();os.chdir(ROOT);p,codes=checked(args)
    if args.worker:_worker(p,codes)
    else:_outer(p,codes)
if __name__=='__main__':main()
