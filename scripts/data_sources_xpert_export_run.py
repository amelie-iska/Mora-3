"""Reviewed one-shot controller for immutable full native X-Pert numeric export.

No source cleanup/retry/reset. Existing sources, MINT code and model code stay untouched.
"""
from __future__ import annotations
import argparse,fcntl,hashlib,json,os,resource,stat,time
from pathlib import Path
from datetime import datetime,timezone
import data_sources_array_export_guard as guard
from data_sources_hardware_storage import load_policy,hardware_snapshot,disk_reason
from data_sources_mint_full_resources import pressure_snapshot,pressure_reason

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()

def write_new(path,obj):
    with Path(path).open("x") as f:json.dump(obj,f,indent=2,sort_keys=True);f.write("\n");f.flush();os.fsync(f.fileno())

def assert_bindings(bindings):
    for path,expected in bindings.items():
        if Path(path).is_symlink() or sha(path)!=expected:raise ValueError("code/evidence SHA mismatch: "+path)

def assert_no_writers(scripts,proc_root="/proc"):
    wanted={str(p).encode() for p in scripts}
    for d in Path(proc_root).iterdir():
        if not d.name.isdigit() or int(d.name)==os.getpid():continue
        try:args=set(x for x in (d/"cmdline").read_bytes().split(b"\0") if x)
        except (FileNotFoundError,ProcessLookupError):continue
        except PermissionError:
            if d.stat().st_uid!=os.getuid():continue
            raise
        if wanted&args:raise ValueError("competing source writer/controller already live")

def claim_destination(stage,lease,record):
    stage=Path(stage);lease=Path(lease)
    if stage.exists() or stage.is_symlink() or lease.exists() or lease.is_symlink():raise FileExistsError("fresh exclusive output and source-family lease required")
    lease.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(lease,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        os.write(fd,(json.dumps(record,sort_keys=True)+"\n").encode());os.fsync(fd)
        stage.mkdir(exist_ok=False)
    except BaseException:
        os.close(fd)  # retained lease/file evidence; never unlink or reset
        raise
    return fd

def combine_pressure(owned,outer):
    ids=[(x["pid"],x["start_ticks"]) for x in owned["members"]+outer["members"]]
    if len(ids)!=len(set(ids)):raise ValueError("overlapping owned/wrapper pressure scope")
    return {"owned_session_pss_bytes":owned["owned_session_pss_bytes"]+outer["owned_session_pss_bytes"],
"owned_session_rss_bytes":owned["owned_session_rss_bytes"]+outer["owned_session_rss_bytes"],
"owned_session_swap_bytes":owned["owned_session_swap_bytes"]+outer["owned_session_swap_bytes"],
"host_mem_available_bytes":min(owned["host_mem_available_bytes"],outer["host_mem_available_bytes"]),
"members":owned["members"]+outer["members"],"owned_session_id":owned["session_id"],"wrapper_session_id":outer["session_id"],
"physical_metric":"aggregate proportional resident set of exact owned workflow SID plus isolated outer wrapper SID"}

def validate_approval(path,expected_sha,plan_path,plan):
    if sha(path)!=expected_sha:raise ValueError("exact root execution approval SHA mismatch")
    a=json.loads(Path(path).read_text())
    if (a.get("approved") is not True or a.get("action")!=plan["action"]
        or a.get("plan_sha256")!=sha(plan_path) or a.get("code_sha256")!=plan["code_sha256"]):
        raise ValueError("execution approval action/plan/complete code map mismatch")
    return a

def stable_payload_checkpoint(stage,reason,proof=None):
    records=[];root=(Path(stage)/"payloads")
    if root.exists():
        for p in sorted(root.iterdir()):
            if p.is_symlink() or not p.is_file():raise ValueError("unresolved partial payload type; no stable proof")
            with p.open("rb") as f:os.fsync(f.fileno())
            records.append({"path":str(p),"bytes":p.stat().st_size,"sha256":sha(p)})
    result={"utc":datetime.now(timezone.utc).isoformat(),"reason":reason,"numeric_partials":records,
            "all_original_sources_unchanged_by_controller":True,"not_complete_or_admitted":True,
            "stable_only_after_verified_empty_owned_session":True,"empty_session_proof":proof}
    return result

def required_code_paths():
    here=Path(__file__).resolve().parent
    return {str(here/name) for name in [
        "data_sources_xpert_arrays.py","data_sources_xpert_hdf.py","data_sources_array_export_guard.py",
        "data_sources_xpert_export_phase.py","data_sources_xpert_export_run.py",
        "data_sources_hardware_storage.py","data_sources_mint_full_resources.py"]}

def validate_code_map(plan):
    required=required_code_paths()
    if set(plan["code_sha256"])!=required or not required<=set(plan["bindings"]):
        raise ValueError("exact complete required source/imported code map required")
    if any(plan["code_sha256"][k]!=plan["bindings"][k] for k in required):
        raise ValueError("code and evidence bindings differ")

def require_isolated_wrapper():
    if os.getsid(0)!=os.getpid() or os.getpgid(0)!=os.getpid():
        raise ValueError("isolated outer wrapper session required")

def validate_resource_envelope(r):
    if (r["cpu_affinity"]!=[29] or r["address_space_limit_bytes"]!=6*1024**3 or r["minimum_free_bytes"]!=64*1024**3
        or r["max_seconds"]!=1800 or r["checkpoint_reserve_seconds"]!=60 or r["pressure_sample_seconds"]!=.1
        or r["physical_memory_policy"]!={"max_session_pss_bytes":2*1024**3,"session_pss_stop_margin_bytes":512*1024**2,
        "min_host_available_bytes":16*1024**3,"max_session_swap_bytes":256*1024**2}):
        raise ValueError("reviewed source resource envelope changed")

def execute_phases(plan,plan_path,stage):
    """Same bounded scientific pipeline for full approved execution or tiny fixtures."""
    stage=Path(stage);r=plan["resources"];started=time.monotonic();runs=[];peak_pss=peak_swap=0
    require_isolated_wrapper()
    cpus=guard.check_affinity(r["cpu_affinity"]);os.sched_setaffinity(0,cpus);os.nice(max(0,10-os.getpriority(os.PRIO_PROCESS,0)))
    resource.setrlimit(resource.RLIMIT_AS,(r["address_space_limit_bytes"],r["address_space_limit_bytes"]));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    env=os.environ.copy()
    for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]:env[k]="1"
    env.pop("OMP_PROC_BIND",None)
    try:
        for mode in ["writer","verifier"]:
            control=stage/(mode+"-control");control.mkdir(exist_ok=False)
            command=[plan["interpreters"][mode],"-B","-u",plan["phase_script"],"--mode",mode,"--plan",str(plan_path),"--stage",str(stage)]
            if mode=="verifier":command+=["--writer-sha",sha(stage/"writer-receipt.json")]
            spec={"cwd":str(control),"command":command,"gate_timeout":5,
"started_monotonic":started,"source_symlink_targets":[],"address_space_limit_bytes":r["address_space_limit_bytes"],
"cpu_affinity":r["cpu_affinity"],"physical_memory_policy":r["physical_memory_policy"],
"pressure_sample_seconds":r["pressure_sample_seconds"],"guard":{"minimum_free_bytes":r["minimum_free_bytes"],"max_seconds":r["max_seconds"]-r["checkpoint_reserve_seconds"]}}
            p=None;owners=None;released=False
            with (control/"native.log").open("xb") as log:
                try:
                    p,owners=guard.spawn_workflow(spec,stdout=log,env=env)
                    write_new(control/"actual-owners-before-START.json",{"owners":owners,"command":command,"spec":spec,"plan_sha256":sha(plan_path)})
                    assert_bindings(plan["bindings"])
                    guard.release_workflow(p);released=True
                    while p.poll() is None:
                        owned=pressure_snapshot(owners["leader"]["sid"]);outer=pressure_snapshot(os.getsid(0))
                        memory=combine_pressure(owned,outer);peak_pss=max(peak_pss,memory["owned_session_pss_bytes"]);peak_swap=max(peak_swap,memory["owned_session_swap_bytes"])
                        s=hardware_snapshot(stage,r["minimum_free_bytes"])
                        reason=pressure_reason(memory,r["physical_memory_policy"]) or disk_reason(s["available_bytes"],time.monotonic()-started,spec["guard"])
                        guard._write(stage/"live-progress.json",{"utc":datetime.now(timezone.utc).isoformat(),"phase":mode,"wrapper_pid":os.getpid(),"owners":owners,
                                    "source_native_PID_state":json.loads((control/"workflow-state.json").read_text()),"memory":memory,"hardware":s,
                                    "peak_aggregate_wrapper_plus_workflow_PSS_bytes":peak_pss,"source_export_complete":False,"task_admitted":0})
                        if reason:
                            proof=guard.stop_workflow(owners,lambda:stable_payload_checkpoint(stage,reason))
                            write_new(control/"guarded-stop-proof.json",proof);raise RuntimeError(reason)
                        time.sleep(r["pressure_sample_seconds"])
                    p.wait(timeout=5)
                    empty=guard._wait_empty_session(owners["leader"]["sid"],timeout=5)
                    state=json.loads((control/"workflow-state.json").read_text())
                    if p.returncode!=0 or state.get("phase")!="workflow_completed" or state.get("native_exit_code")!=0:
                        raise RuntimeError("owned source phase failed; originals/partials retained")
                    if p.stdin and not p.stdin.closed:p.stdin.close()
                    runs.append({"phase":mode,"owners":owners,"native_state":state,"native_exit_code":p.returncode,
                                 "empty_session_verified":True,"empty_session_proof":empty})
                    write_new(control/"completed-owned-phase-receipt.json",runs[-1])
                except BaseException as exc:
                    if owners is not None:
                        # Even exited roots need the full-SID/thread proof before
                        # any partial data hashes can be called stable.
                        stopped=guard.stop_workflow(owners,lambda:stable_payload_checkpoint(stage,str(exc)))
                        if not (control/"exception-stop-proof.json").exists():write_new(control/"exception-stop-proof.json",stopped)
                    # If readiness failed, START was never sent. The guard owns
                    # bootstrap EOF cleanup; no unverifiable PID is signalled here.
                    if p is not None and p.stdin and not p.stdin.closed:p.stdin.close()
                    raise
        manifest=stage/"numeric-store-manifest.json"
        result={"utc":datetime.now(timezone.utc).isoformat(),"phase_runs":runs,"numeric_store_manifest":str(manifest),"numeric_store_manifest_sha256":sha(manifest),
                "peak_aggregate_wrapper_plus_workflow_PSS_bytes":peak_pss,"peak_owned_swap_bytes":peak_swap,
                "elapsed_seconds":time.monotonic()-started,"task_admitted":0,"model_use_allowed":False}
        write_new(stage/"completed-controller-proof.json",result);return result
    except BaseException as exc:
        if not (stage/"controller-failure.json").exists():
            write_new(stage/"controller-failure.json",{"utc":datetime.now(timezone.utc).isoformat(),"error":type(exc).__name__+":"+str(exc),
"completed_phases":runs,"all_partials_originals_logs_and_leases_retained":True,"no_retry_reset_cleanup":True})
        raise

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--plan",required=True);ap.add_argument("--approval",required=True);ap.add_argument("--approval-sha256",required=True)
    a=ap.parse_args();require_isolated_wrapper();pp=Path(a.plan).resolve();plan=json.loads(pp.read_text())
    validate_approval(a.approval,a.approval_sha256,pp,plan)
    if plan.get("fixture_mode") is not False:raise ValueError("production launcher refuses fixture plans")
    if plan["action"]!="XPERT_full_native_array_export_v1":raise ValueError("exact source execution action required")
    load_policy(plan["execution_policy"],plan["execution_policy_sha256"]);assert_bindings(plan["bindings"])
    validate_code_map(plan)
    r=plan["resources"];validate_resource_envelope(r)
    assert_no_writers([Path(__file__).resolve(),plan["phase_script"]])
    for s in plan["source_specs"]:
        if Path(s["source_path"]).is_symlink() or Path(s["source_path"]).stat().st_size!=s["source_bytes"] or sha(s["source_path"])!=s["source_sha256"]:
            raise ValueError("original source whole SHA/size changed before source launch")
    space=hardware_snapshot(Path(plan["new_output"]).parent,r["minimum_free_bytes"])
    if space["available_bytes"]-plan["expected_numeric_and_validity_bytes"]-4*1024**2<=r["minimum_free_bytes"]:
        raise ValueError("hardware capacity cannot retain full exact source-derived payload plus free-space buffer")
    mem=pressure_snapshot(os.getsid(0))
    if mem["host_mem_available_bytes"]<16*1024**3:raise ValueError("host available RAM floor not met")
    leasefd=claim_destination(plan["new_output"],plan["lease_path"],{"utc":datetime.now(timezone.utc).isoformat(),"pid":os.getpid(),
             "identity":guard.workflow_identity(os.getpid()),"plan_path":str(pp),"plan_sha256":sha(pp),"approval_sha256":a.approval_sha256})
    try:
        write_new(Path(plan["new_output"])/"approved-launch.json",{"utc":datetime.now(timezone.utc).isoformat(),"wrapper":guard.workflow_identity(os.getpid()),
"plan":str(pp),"plan_sha256":sha(pp),"root_execution_approval":str(Path(a.approval).resolve()),"approval_sha256":a.approval_sha256,
"hardware_at_launch":space,"memory_at_launch":mem,"substantive_training_performed":False})
        execute_phases(plan,pp,Path(plan["new_output"]))
    finally:os.close(leasefd)

if __name__=="__main__":main()
