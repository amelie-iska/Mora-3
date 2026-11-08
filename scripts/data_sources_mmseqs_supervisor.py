"""Finite gated source worker: stable ownership before native work, no network."""
import ctypes
import hashlib
import json
import os
import pathlib
import resource
import select
import signal
import subprocess
import sys
import time
from data_sources_mmseqs_createdb import guard_decision,process_identity,wait_process_identity

_HERE=pathlib.Path(__file__).resolve().parent
_ENTRY="import sys;sys.path.insert(0,"+repr(str(_HERE))+");from data_sources_mmseqs_supervisor import gated_worker_main;gated_worker_main()"


def stage_tree_bytes(root):
    total=0
    for p in pathlib.Path(root).rglob('*'):
        try:
            if p.is_symlink():raise ValueError('newstage external/source symlink forbidden')
            if p.is_file():total+=p.stat().st_size
        except FileNotFoundError:continue  # only this new tool's concurrent internal merge files
    return total


def supervisor_command(spec):
    return [sys.executable,'-B','-u','-c',_ENTRY,json.dumps(spec,sort_keys=True,separators=(',',':'))]


def spawn_gated_group(spec,stdout=None,env=None,_waiter=None):
    """Return a ready exclusive owner; native work still waits for release_gate."""
    if sys.platform!='linux':raise ValueError('reviewed Linux parent-death/process-group semantics required')
    if not 0<spec['gate_timeout']<=10 or not spec['command'] or any(not isinstance(x,str) for x in spec['command']):raise ValueError('bounded gate and explicit command argv required')
    command=supervisor_command(spec)
    p=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=stdout,stderr=subprocess.STDOUT,start_new_session=True,cwd=spec['cwd'],env=env,close_fds=True)
    try:
        identity=(_waiter or wait_process_identity)(p.pid,command,timeout=spec['gate_timeout'])
        if identity['argv']!=command or identity['pgid']!=p.pid or identity['pid']!=p.pid:raise ValueError('stable gated owner identity mismatch')
    except BaseException:
        p.stdin.close()  # never sent START; bootstrap cannot spawn native work
        try:p.wait(timeout=spec['gate_timeout']+1)
        except subprocess.TimeoutExpired:pass  # still no native-start authorization exists
        raise
    return p,identity,command


def release_gate(p):
    if p.poll() is not None:raise ValueError('gated owner exited before native work release')
    p.stdin.write(b'START\n');p.stdin.flush()  # keep open as parent-liveness channel


def _write_state(path,state):
    """Atomic update of this execution's explicitly mutable current-status file."""
    path=pathlib.Path(path);tmp=path.with_name(path.name+'.writing-'+str(os.getpid()))
    with tmp.open('w') as f:json.dump(state,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def gated_worker_main():
    spec=json.loads(sys.argv[1]);pid=os.getpid();expected_argv=supervisor_command(spec);native=None;initial=None
    state=dict(supervisor_pid=pid,native_work_started=False,native_pid=None,phase='gate_waiting',source_connections=0)
    def status(phase,**extra):
        state.update(updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),phase=phase,**extra);_write_state(spec['state_path'],state)
    def kill_this_owned_group(reason):
        status('owned_group_stopping',guard_reason=reason)
        actual=process_identity(pid)
        if initial is None or actual!=initial or actual['argv']!=expected_argv or actual['pgid']!=pid:os._exit(77)  # native PDEATHSIG still applies; no unverifiable group signal
        os.killpg(pid,signal.SIGKILL)
    def native_limits_and_parent_death():
        libc=ctypes.CDLL(None,use_errno=True);libc.prctl.argtypes=[ctypes.c_int,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_ulong];libc.prctl.restype=ctypes.c_int
        if libc.prctl(1,signal.SIGKILL,0,0,0)!=0 or os.getppid()!=pid:os._exit(78)
        resource.setrlimit(resource.RLIMIT_AS,(spec['address_space_limit_bytes'],spec['address_space_limit_bytes']));resource.setrlimit(resource.RLIMIT_CORE,(0,0));os.nice(10)
    status('gate_waiting')
    ready,_,_=select.select([0],[],[],spec['gate_timeout'])
    if not ready:status('gate_timeout_no_native_work');raise SystemExit(72)
    gate=os.read(0,64)
    if gate!=b'START\n':status('gate_EOF_no_native_work' if not gate else 'malformed_gate_no_native_work');raise SystemExit(73)
    initial=process_identity(pid)
    if initial is None or initial['argv']!=expected_argv or initial['pgid']!=pid:status('unverified_self_no_native_work');raise SystemExit(74)
    # Parent lost between START and spawn: never start the native operation.
    ready,_,_=select.select([0],[],[],0)
    if ready and os.read(0,64)==b'':status('parent_EOF_before_native_work');raise SystemExit(75)
    try:
        native=subprocess.Popen(spec['command'],stdin=subprocess.DEVNULL,stdout=None,stderr=None,start_new_session=False,cwd=spec['cwd'],preexec_fn=native_limits_and_parent_death,close_fds=True)
        state.update(native_work_started=True,native_pid=native.pid);status('native_active',native_requested_command=spec['command'])
        last=0
        while native.poll() is None:
            ready,_,_=select.select([0],[],[],.05)
            if ready:
                data=os.read(0,64)
                kill_this_owned_group('parent_control_EOF' if not data else 'unexpected_parent_control_data')
            used=stage_tree_bytes(spec['cwd']);free=os.statvfs(spec['cwd']).f_bavail*os.statvfs(spec['cwd']).f_frsize;reason=guard_decision(used,free,time.monotonic()-spec['started_monotonic'],spec['guard'])
            if reason:kill_this_owned_group(reason)
            if time.monotonic()-last>=2:status('native_active',all_stage_bytes=used,native_current_snapshot=process_identity(native.pid));last=time.monotonic()
        code=native.wait();status('native_completed',native_exit_code=code);raise SystemExit(code)
    finally:
        if native is not None and native.poll() is None:kill_this_owned_group('supervisor_exception_with_native_live')
