"""Read-only aggregate owned-session physical-memory and exact CPU policy gates.
RLIMIT_AS is per-process virtual address space; it is not a total-RAM limit.
"""
import os,pathlib

def _stat(p):
    try:
        v=(p/'stat').read_text().rsplit(')',1)[1].split()
        return dict(state=v[0],sid=int(v[3]),start=int(v[19]))
    except (FileNotFoundError,ProcessLookupError):return None

def pressure_snapshot(session,proc_root='/proc'):
    root=pathlib.Path(proc_root);members=[];pss=rss=swap=0
    for p in root.iterdir():
        if not p.name.isdigit():continue
        before=_stat(p)
        if before is None or before['sid']!=session:continue
        if before['state'] in ('Z','X'):
            try:
                for task in (p/'task').iterdir():
                    t=_stat(task)
                    if t is not None and t['sid']==session and t['state'] not in ('Z','X'):raise ValueError('unresolved resident memory of zombie leader with live owned threads')
            except (FileNotFoundError,ProcessLookupError):
                if _stat(p) is not None:raise ValueError('unresolved zombie owned-process thread census')
            continue
        try:
            lines=(p/'smaps_rollup').read_text().splitlines();values={line.split(':',1)[0]:int(line.split()[1])*1024 for line in lines if line.startswith(('Rss:','Pss:','Swap:'))}
            after=_stat(p)
            if after is None:continue
            if after['start']!=before['start'] or after['sid']!=session:raise ValueError('owned memory PID/start/session changed')
            if after['state'] in ('Z','X'):continue
            if set(values)!={'Pss','Rss','Swap'}:raise ValueError('incomplete owned physical memory metrics')
        except (FileNotFoundError,ProcessLookupError,PermissionError) as exc:
            if _stat(p) is not None:raise ValueError('unresolved live owned-process physical memory') from exc
            continue
        members.append(dict(pid=int(p.name),start_ticks=before['start'],pss_bytes=values['Pss'],rss_bytes=values['Rss'],swap_bytes=values['Swap']));pss+=values['Pss'];rss+=values['Rss'];swap+=values['Swap']
    available=int(next(line.split()[1] for line in (root/'meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    return dict(session_id=session,owned_processes=len(members),owned_session_pss_bytes=pss,owned_session_rss_bytes=rss,owned_session_swap_bytes=swap,host_mem_available_bytes=available,members=members,physical_metric='sum of per-process proportional resident set; aggregate RSS also reported and may double-count shared maps')

def pressure_reason(snapshot,policy):
    if snapshot['owned_session_pss_bytes']>=policy['max_session_pss_bytes']-policy['session_pss_stop_margin_bytes']:return 'aggregate_owned_session_PSS_margin'
    if snapshot['host_mem_available_bytes']<policy['min_host_available_bytes']:return 'host_available_RAM_reserve'
    if snapshot['owned_session_swap_bytes']>=policy['max_session_swap_bytes']:return 'owned_session_swap_limit'
    return None

def check_affinity(cpus,available=None):
    available=os.sched_getaffinity(0) if available is None else available
    if len(cpus)!=4 or len(set(cpus))!=4 or any(not isinstance(v,int) or isinstance(v,bool) or v<0 for v in cpus) or not set(cpus)<=set(available):raise ValueError('exact four allowed CPUs required; no expansion or silent replacement')
    return set(cpus)
