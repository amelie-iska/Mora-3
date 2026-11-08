"""NEW full-cluster guard; prior completed-pilot guard remains immutable.
Two independent physical-memory/space/deadline controllers, exact surviving owners.
"""
import ctypes,json,os,pathlib,resource,select,signal,subprocess,sys,time
from data_sources_hardware_storage import disk_reason,hardware_snapshot
from data_sources_mint_full_resources import pressure_snapshot,pressure_reason,check_affinity

_HERE=pathlib.Path(__file__).resolve().parent
_ENTRY="import sys;sys.path.insert(0,"+repr(str(_HERE))+");from data_sources_mint_full_guard_v2 import worker_main;worker_main()"

def workflow_identity(pid):
    try:
        root=pathlib.Path('/proc')/str(pid)
        a=root.joinpath('stat').read_text().rsplit(')',1)[1].split()
        if a[0]=='Z':return None
        argv=[v.decode() for v in root.joinpath('cmdline').read_bytes().split(b'\0') if v]
        b=root.joinpath('stat').read_text().rsplit(')',1)[1].split()
        if a[19]!=b[19] or b[0]=='Z':return None
        return dict(pid=pid,start_ticks=int(a[19]),pgid=int(a[2]),sid=int(a[3]),argv=argv)
    except (FileNotFoundError,ProcessLookupError):return None

def _command(role,payload):
    return [sys.executable,'-B','-u','-c',_ENTRY,role,json.dumps(payload,sort_keys=True,separators=(',',':'))]

def _wait_identity(pid,argv,group,timeout):
    end=time.monotonic()+timeout;previous=None;start=None
    while time.monotonic()<end:
        actual=workflow_identity(pid)
        if actual is None:raise ProcessLookupError('owned bootstrap exited before ready')
        if start is None:start=actual['start_ticks']
        if actual['start_ticks']!=start:raise ValueError('bootstrap PID/start reused')
        if actual['argv']==argv and actual['pgid']==group and actual['sid']==group:
            if previous==actual:return actual
            previous=actual
        else:previous=None
        time.sleep(.005)
    raise ValueError('bounded exact PID/start/argv/session/group readiness failed')

def guarded_tree_bytes(root,allowed_sources):
    root=pathlib.Path(root).resolve();allowed={pathlib.Path(p).resolve(strict=True) for p in allowed_sources};total=0
    for entry_number,p in enumerate(root.rglob('*'),1):
        try:
            if p.is_symlink():
                target=p.resolve(strict=False)
                if not target.exists() and not target.is_relative_to(root):raise ValueError('dangling external source symlink forbidden')
                if target not in allowed and not target.is_relative_to(root):raise ValueError('source symlink target outside explicit allowlist')
                total+=p.lstat().st_size
            elif p.is_file():total+=p.stat().st_size
        except FileNotFoundError:continue
    return total

def _validate_owners(owners):
    leader=owners['leader'];anchor=owners['anchor'];g=leader['pid']
    if not leader['argv'] or not anchor['argv'] or leader['pgid']!=g or leader['sid']!=g or anchor['pid']==g or anchor['pgid']!=g or anchor['sid']!=g:raise ValueError('invalid exact isolated-session ownership proof')
    live=[]
    for owner in [leader,anchor]:
        actual=workflow_identity(owner['pid'])
        if actual is not None:
            if actual!=owner:raise ValueError('owner PID/start/argv/session/group changed; no signals')
            live.append(actual)
    return g,live

def _session_writers(session,proc_root='/proc'):
    """Census every live thread in this SID, including zombie leaders/other PGIDs."""
    rows=[]
    for p in pathlib.Path(proc_root).iterdir():
        if not p.name.isdigit():continue
        try:
            leader=p.joinpath('stat').read_text().rsplit(')',1)[1].split()
            if int(leader[3])!=session:continue
            tasks=list(p.joinpath('task').iterdir())
            for t in tasks:
                if not t.name.isdigit():continue
                try:fields=t.joinpath('stat').read_text().rsplit(')',1)[1].split()
                except (FileNotFoundError,ProcessLookupError):continue
                if int(fields[3])==session and fields[0] not in ('Z','X'):
                    rows.append(dict(pid=int(p.name),tid=int(t.name),state=fields[0],pgid=int(fields[2]),sid=int(fields[3])))
        except (FileNotFoundError,ProcessLookupError):continue
        except PermissionError:
            if p.stat().st_uid!=os.getuid():continue
            raise
    return rows

def _wait_empty_session(session,timeout=5):
    if not 0<=timeout<=15:raise ValueError('bounded empty-session proof required')
    end=time.monotonic()+timeout;empty=observations=maximum_live=0
    while True:
        writers=_session_writers(session);observations+=1;maximum_live=max(maximum_live,len(writers))
        empty=empty+1 if not writers else 0
        if empty>=2:return dict(session_id=session,consecutive_empty_live_thread_censuses=empty,observations=observations,maximum_live_threads_observed=maximum_live)
        if time.monotonic()>=end:raise ValueError('isolated session still has live/unsettled threads; no stable checkpoint: '+str(writers[:8]))
        time.sleep(.01)

def stop_workflow(owners,after_empty,settle_timeout=5):
    """No stable callback before exact-owner kill AND verified empty whole session."""
    group,live=_validate_owners(owners);signals=[]
    if live:
        os.killpg(group,signal.SIGSTOP);signals.append('SIGSTOP')
        # No file hashes while merely stopping: other tasks may still be R/D.
        _,current=_validate_owners(owners)
        if current:
            os.killpg(group,signal.SIGKILL);signals.append('SIGKILL')
    proof=_wait_empty_session(group,timeout=settle_timeout)
    checkpoint=after_empty()
    return dict(owners_already_exited=not live,signals=signals,checkpoint=checkpoint,
        empty_session_verified=True,stable_callback_after_empty_session=True,empty_session_proof=proof)

def _write(path,state):
    p=pathlib.Path(path);q=p.with_name(p.name+'.writing-'+str(os.getpid()))
    with q.open('w') as f:json.dump(state,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(q,p)

def spawn_workflow(spec,stdout=None,env=None,_waiter=None):
    if sys.platform!='linux' or not 0<spec['gate_timeout']<=10 or not spec['command']:raise ValueError('bounded Linux explicit workflow required')
    command=_command('supervisor',spec)
    p=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=stdout,stderr=subprocess.STDOUT,start_new_session=True,cwd=spec['cwd'],env=env,close_fds=True)
    try:
        leader=(_waiter or _wait_identity)(p.pid,command,p.pid,spec['gate_timeout'])
        anchor_command=_command('anchor',dict(job=spec,creator=leader))
        end=time.monotonic()+spec['gate_timeout'];state_path=pathlib.Path(spec['cwd'])/'workflow-state.json'
        while time.monotonic()<end:
            if p.poll() is not None:raise ValueError('supervisor exited before anchor verification')
            if state_path.exists():
                state=json.loads(state_path.read_text())
                if state.get('phase')=='native_gate_waiting' and state.get('anchor_pid'):
                    anchor=_wait_identity(state['anchor_pid'],anchor_command,p.pid,spec['gate_timeout'])
                    owners=dict(leader=leader,anchor=anchor);_validate_owners(owners);return p,owners
            time.sleep(.005)
        raise ValueError('bounded verified second anchor readiness failed')
    except BaseException:
        p.stdin.close()  # no native START was ever authorized
        try:p.wait(timeout=spec['gate_timeout']+2)
        except subprocess.TimeoutExpired:pass
        raise

def release_workflow(p):
    if p.poll() is not None:raise ValueError('owner exited before release')
    p.stdin.write(b'START\n');p.stdin.flush()

def _own_stop(initial,reason,state_path):
    try:_write(state_path,dict(phase='owned_group_stopping',reason=reason,owner=initial))
    except OSError:pass  # a full/unwritable status filesystem must not prevent owned cleanup
    actual=workflow_identity(os.getpid())
    if actual!=initial or actual['sid']!=actual['pgid']:os._exit(77)
    os.killpg(actual['pgid'],signal.SIGKILL)

_LAST_GUARD={}
def _budget(job):
    # Each owner samples independently; no future-slot booking or blind sleep.
    now=time.monotonic();key=os.getpid();prior=_LAST_GUARD.get(key)
    if prior is not None and now-prior[0]<job.get('pressure_sample_seconds',0.5):return prior[1]
    space=hardware_snapshot(job['cwd'],job['guard']['minimum_free_bytes'])
    reason=disk_reason(space['available_bytes'],now-job['started_monotonic'],job['guard'])
    if job.get('physical_memory_policy') is not None:
        try:
            snapshot=pressure_snapshot(os.getsid(0));memory_reason=pressure_reason(snapshot,job['physical_memory_policy'])
            _write(pathlib.Path(job['cwd'])/('physical-memory-'+str(key)+'.json'),dict(sampled_monotonic=now,**snapshot))
            reason=reason or memory_reason
        except (OSError,ValueError,StopIteration):reason='unresolved_owned_physical_memory'
    if not reason:guarded_tree_bytes(job['cwd'],job['source_symlink_targets'])  # preserve exact external-source allowlist; no file/byte allocation cap
    _LAST_GUARD[key]=(now,reason);return reason

def _anchor_main(payload):
    job=payload['job'];creator=payload['creator'];group=creator['pid'];self_id=workflow_identity(os.getpid());state=pathlib.Path(job['cwd'])/'anchor-state.json'
    if not creator['argv'] or creator['sid']!=group or creator['pgid']!=group or workflow_identity(group)!=creator or self_id['pgid']!=group or self_id['sid']!=group or self_id['argv']!=_command('anchor',payload):raise SystemExit(71)
    _write(state,dict(phase='anchor_ready',identity=self_id,creator=creator))
    while True:
        ready,_,_=select.select([0],[],[],.05)
        if ready:
            data=os.read(0,64)
            if data==b'DONE\n':_write(state,dict(phase='anchor_completed',identity=self_id));return
            _own_stop(self_id,'supervisor_control_EOF' if not data else 'bad_control',state)
        if workflow_identity(group)!=creator:_own_stop(self_id,'supervisor_identity_lost',state)
        reason=_budget(job)
        if reason:_own_stop(self_id,reason,state)

def _active_group(group):
    found=[]
    for p in pathlib.Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:fields=p.joinpath('stat').read_text().rsplit(')',1)[1].split()
        except (FileNotFoundError,ProcessLookupError):continue
        except PermissionError:
            if p.stat().st_uid!=os.getuid():continue
            raise
        if fields[0]!='Z' and int(fields[2])==group and int(fields[3])==group:found.append(int(p.name))
    return found

def _supervisor_main(job):
    pid=os.getpid();initial=workflow_identity(pid);expected=_command('supervisor',job);state=pathlib.Path(job['cwd'])/'workflow-state.json';native=None;anchor=None;anchor_id=None
    if initial['argv']!=expected or initial['sid']!=pid or initial['pgid']!=pid:raise SystemExit(72)
    if job.get('cpu_affinity') is not None:
        os.sched_setaffinity(0,check_affinity(job['cpu_affinity']));os.nice(max(0,10-os.getpriority(os.PRIO_PROCESS,0)))
    def status(phase,**more):_write(state,dict(phase=phase,leader=initial,anchor_pid=anchor.pid if anchor else None,native_pid=native.pid if native else None,**more))
    def native_limits():
        libc=ctypes.CDLL(None,use_errno=True)
        if libc.prctl(1,signal.SIGKILL,0,0,0)!=0 or os.getppid()!=pid:os._exit(78)
        resource.setrlimit(resource.RLIMIT_AS,(job['address_space_limit_bytes'],job['address_space_limit_bytes']));resource.setrlimit(resource.RLIMIT_CORE,(0,0));os.nice(max(0,10-os.getpriority(os.PRIO_PROCESS,0)))
        if job.get('cpu_affinity') is not None:os.sched_setaffinity(0,check_affinity(job['cpu_affinity']))
    try:
        payload=dict(job=job,creator=initial);anchor_cmd=_command('anchor',payload)
        anchor=subprocess.Popen(anchor_cmd,stdin=subprocess.PIPE,cwd=job['cwd'],stdout=None,stderr=None,close_fds=True)
        anchor_id=_wait_identity(anchor.pid,anchor_cmd,pid,job['gate_timeout']);status('native_gate_waiting')
        ready,_,_=select.select([0],[],[],job['gate_timeout'])
        if not ready or os.read(0,64)!=b'START\n':return
        if workflow_identity(pid)!=initial or workflow_identity(anchor.pid)!=anchor_id:raise ValueError('owners changed before native gate')
        ready,_,_=select.select([0],[],[],0)
        if ready and os.read(0,64)==b'':return  # parent lost before native spawn
        native=subprocess.Popen(job['command'],stdin=subprocess.DEVNULL,cwd=job['cwd'],stdout=None,stderr=None,preexec_fn=native_limits,close_fds=True);status('native_active')
        while native.poll() is None:
            ready,_,_=select.select([0],[],[],.05)
            if ready:_own_stop(initial,'wrapper_control_EOF' if not os.read(0,64) else 'bad_wrapper_control',state)
            if workflow_identity(anchor.pid)!=anchor_id:_own_stop(initial,'anchor_identity_lost',state)
            reason=_budget(job)
            if reason:_own_stop(initial,reason,state)
        code=native.wait()
        if {r['pid'] for r in _session_writers(pid)}-{pid,anchor.pid}:_own_stop(initial,'native_root_exited_with_live_descendants',state)
        anchor.stdin.write(b'DONE\n');anchor.stdin.flush();anchor.stdin.close();anchor.wait(timeout=5)
        if anchor.returncode!=0:raise ValueError('anchor did not exit cleanly')
        status('workflow_completed',native_exit_code=code);raise SystemExit(code)
    finally:
        if (native is not None and native.poll() is None) or (anchor is not None and anchor.poll() is None):_own_stop(initial,'supervisor_exit_with_owned_work',state)

def worker_main():
    role=sys.argv[1];payload=json.loads(sys.argv[2])
    if role=='anchor':_anchor_main(payload)
    elif role=='supervisor':_supervisor_main(payload)
    else:raise SystemExit(70)
