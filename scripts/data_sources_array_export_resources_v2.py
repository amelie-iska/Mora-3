"""Read-only aggregate owned-session physical-memory and exact CPU policy gates.
RLIMIT_AS is per-process virtual address space; it is not a total-RAM limit.
"""
import os,pathlib,time
EXIT_RECHECK_SECONDS=.1

def _stat(p):
    try:
        v=(p/'stat').read_text().rsplit(')',1)[1].split()
        return dict(state=v[0],sid=int(v[3]),start=int(v[19]))
    except (FileNotFoundError,ProcessLookupError):return None

def _same_scope(before,after,session):
    if after["start"]!=before["start"] or after["sid"]!=session:
        raise ValueError("owned memory PID/start/session identity changed")

def _dead_threads_proven(p,before,session):
    after=_stat(p)
    if after is None:return True
    _same_scope(before,after,session)
    if after["state"] not in ("Z","X"):return False
    try:
        tasks=list((p/"task").iterdir())
        for task in tasks:
            t=_stat(task)
            if t is None:continue
            if t["sid"]!=session:raise ValueError("owned thread session scope changed")
            if t["state"] not in ("Z","X"):
                raise ValueError("unresolved resident memory of dead leader with live owned threads")
    except (FileNotFoundError,ProcessLookupError) as exc:
        if _stat(p) is not None:raise ValueError("unresolved dead-owned thread census") from exc
    final=_stat(p)
    if final is not None:
        _same_scope(before,final,session)
        if final["state"] not in ("Z","X"):raise ValueError("dead owned process became unresolved live")
    return True

def _missing_memory_exit_proven(p,before,session):
    end=time.monotonic()+EXIT_RECHECK_SECONDS
    while True:
        after=_stat(p)
        if after is None:return True
        _same_scope(before,after,session)
        if after["state"] in ("Z","X"):return _dead_threads_proven(p,before,session)
        if time.monotonic()>=end:raise ValueError("unresolved live owned-process physical memory")
        time.sleep(.002)

def pressure_snapshot(session,proc_root="/proc"):
    root=pathlib.Path(proc_root);members=[];pss=rss=swap=0
    for p in root.iterdir():
        if not p.name.isdigit():continue
        before=_stat(p)
        if before is None or before["sid"]!=session:continue
        if before["state"] in ("Z","X"):
            _dead_threads_proven(p,before,session)
            continue
        try:
            lines=(p/"smaps_rollup").read_text().splitlines()
            values={line.split(":",1)[0]:int(line.split()[1])*1024 for line in lines if line.startswith(("Rss:","Pss:","Swap:"))}
            after=_stat(p)
            if after is None:continue
            _same_scope(before,after,session)
            if after["state"] in ("Z","X"):
                _dead_threads_proven(p,before,session)
                continue
            if set(values)!={"Pss","Rss","Swap"}:raise ValueError("incomplete owned physical memory metrics")
        except (FileNotFoundError,ProcessLookupError,PermissionError) as exc:
            # Never zero an unreadable live member. Only same-identity proved
            # exit and no remaining live threads can remove it from the sum.
            if not _missing_memory_exit_proven(p,before,session):
                raise ValueError("unresolved owned physical memory") from exc
            continue
        members.append(dict(pid=int(p.name),start_ticks=before["start"],pss_bytes=values["Pss"],rss_bytes=values["Rss"],swap_bytes=values["Swap"]))
        pss+=values["Pss"];rss+=values["Rss"];swap+=values["Swap"]
    available=int(next(line.split()[1] for line in (root/"meminfo").read_text().splitlines() if line.startswith("MemAvailable:")))*1024
    return dict(session_id=session,owned_processes=len(members),owned_session_pss_bytes=pss,owned_session_rss_bytes=rss,
        owned_session_swap_bytes=swap,host_mem_available_bytes=available,members=members,
        physical_metric="sum of per-process proportional resident set; exit omission requires no-live-thread proof")

def pressure_reason(snapshot,policy):
    if snapshot['owned_session_pss_bytes']>=policy['max_session_pss_bytes']-policy['session_pss_stop_margin_bytes']:return 'aggregate_owned_session_PSS_margin'
    if snapshot['host_mem_available_bytes']<policy['min_host_available_bytes']:return 'host_available_RAM_reserve'
    if snapshot['owned_session_swap_bytes']>=policy['max_session_swap_bytes']:return 'owned_session_swap_limit'
    return None

def check_affinity(cpus,available=None):
    available=os.sched_getaffinity(0) if available is None else available
    if len(cpus)!=4 or len(set(cpus))!=4 or any(not isinstance(v,int) or isinstance(v,bool) or v<0 for v in cpus) or not set(cpus)<=set(available):raise ValueError('exact four allowed CPUs required; no expansion or silent replacement')
    return set(cpus)
