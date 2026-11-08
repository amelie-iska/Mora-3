"""Finite reviewed full-provider createdb only. No clustering, cleanup or networking."""
import argparse
import datetime
import fcntl
import hashlib
import json
import importlib.metadata
import platform
import sys
import os
import pathlib
import resource
import subprocess
import time
from data_sources_mmseqs_createdb import createdb_command,guard_decision,process_identity,stop_owned_group,tree_bytes,verify_approval,verify_native_database,wait_process_identity

CODE_PATHS=('scripts/data_sources_mmseqs_createdb.py','scripts/data_sources_mmseqs_createdb_run.py')
ROOT=pathlib.Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/data-sources/20261031-legacy-mint-v1'
NOW=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()

def digest(path,on_progress=None):
    h=hashlib.sha256();read=last=0
    with pathlib.Path(path).open('rb') as f:
        for data in iter(lambda:f.read(8*1024**2),b''):
            h.update(data);read+=len(data)
            if on_progress and read-last>=1024**3:on_progress(read);last=read
    return h.hexdigest()

def write_new(path,obj):
    with pathlib.Path(path).open('x') as f:json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--approval',required=True);args=parser.parse_args();os.chdir(ROOT)
    pp=pathlib.Path(args.plan).resolve();ap=pathlib.Path(args.approval).resolve();plan=json.loads(pp.read_text());approval=json.loads(ap.read_text());codes={p:digest(ROOT/p) for p in CODE_PATHS};verify_approval(approval,digest(pp),codes)
    if plan.get('code_sha256')!=codes:raise ValueError('plan code bindings changed')
    if not 0<plan.get('identity_ready_timeout_seconds',0)<=10:raise ValueError('bounded stable identity readiness required before spawn')
    if not sys.dont_write_bytecode or platform.python_version()!=plan['python_version'] or importlib.metadata.version('pyarrow')!=plan['pyarrow_version']:raise ValueError('reviewed existing interpreter-B/pyarrow version required')
    if plan['stage_cap_bytes']!=32*1024**3 or plan['stop_margin_bytes']<1024**3 or plan['minimum_free_bytes']<200*1024**3 or plan['max_seconds']>7200:raise ValueError('reviewed cap/reserve/deadline envelope changed')
    if plan['family_allocation_bytes']!=56*1024**3 or plan['other_family_reserved_bytes']+plan['stage_cap_bytes']>plan['family_allocation_bytes']:raise ValueError('family cumulative reservations exceed56GiB')
    if plan['source_sha256']!='210382afd5136897cde16464f76f41857c82123707dd1452236e3d4f9d0177c4' or plan['source_rows']!=59309604 or plan['source_bytes']!=13003020999:raise ValueError('exact fullSTRING source identity required')
    if digest(plan['budget_receipt'])!=plan['budget_receipt_sha256']:raise ValueError('coordinated budget receipt changed')
    if digest(plan['binary'])!=plan['binary_sha256']:raise ValueError('read-only installed binary changed')
    version=subprocess.run([plan['binary'],'version'],capture_output=True,text=True,check=True,timeout=15).stdout.strip()
    if version!=plan['binary_version']:raise ValueError('installed binary version changed')
    source=pathlib.Path(plan['source_path']);D=pathlib.Path(plan['new_output']);expected=BASE/'curated/MINT-full-provider-createdb-compressed-v1'
    if D.resolve()!=expected or D.exists():raise ValueError('fresh exclusive reviewed destination required; no resume/reset fallback')
    if source.stat().st_size!=plan['source_bytes']:raise ValueError('source size changed')
    if os.statvfs(BASE).f_bavail*os.statvfs(BASE).f_frsize<plan['minimum_free_bytes']+plan['stage_cap_bytes']:raise ValueError('free reserve plus fullstagecommit not available')
    lease=pathlib.Path(plan['lease_path'])
    if lease.resolve()!=BASE/'audits/MINT-full-provider-createdb-compressed-v1.lease':raise ValueError('unexpected stage lease')
    with lease.open('x+') as leasefile:
        fcntl.flock(leasefile,fcntl.LOCK_EX|fcntl.LOCK_NB);json.dump(dict(owner_at_creation=process_identity(os.getpid()),plan_sha256=digest(pp),approval_sha256=digest(ap),no_reentry=True,created_utc=NOW()),leasefile);leasefile.flush();os.fsync(leasefile.fileno());D.mkdir(exist_ok=False);(D/'tmp').mkdir();start=time.monotonic();peak=0;child=None;identity=None;progress={}
        def live(phase,**extra):
            (D/'live.json').write_text(json.dumps(dict(updated_utc=NOW(),wrapper_pid=os.getpid(),phase=phase,child_identity=identity,child_currently_alive=child is not None and child.poll() is None,elapsed_seconds=time.monotonic()-start,peak_all_stage_bytes_observed=peak,stage_cap_bytes=plan['stage_cap_bytes'],source_connections=0,progress=progress,**extra),indent=2)+'\n')
        def free():return os.statvfs(D).f_bavail*os.statvfs(D).f_frsize
        def checkpoint(reason):
            files=[]
            for path in sorted(D.rglob('*')):
                if path.is_symlink():raise ValueError('unexpected newstage symlink')
                if path.is_file() and path.name!='live.json':
                    with path.open('rb') as f:os.fsync(f.fileno())
                    files.append(dict(path=str(path),bytes=path.stat().st_size,sha256=digest(path)))
            rec=dict(created_utc=NOW(),reason=reason,all_written_new_stage_bytes_preserved=True,source_original_not_changed=True,files=files,stage_bytes=tree_bytes(D));write_new(D/'stopped-byte-checkpoint.json',rec);return dict(path=str(D/'stopped-byte-checkpoint.json'),sha256=digest(D/'stopped-byte-checkpoint.json'),stage_bytes=rec['stage_bytes'])
        try:
            live('preflight full immutable compressed sourceSHA advancing')
            def hashprogress(n):progress.update(compressed_source_bytes_SHA_checked=n);live('preflight full immutable compressed sourceSHA advancing')
            if digest(source,hashprogress)!=plan['source_sha256']:raise ValueError('full source SHA changed')
            def restrict():
                resource.setrlimit(resource.RLIMIT_AS,(plan['child_address_space_limit_bytes'],plan['child_address_space_limit_bytes']));resource.setrlimit(resource.RLIMIT_CORE,(0,0));os.nice(10)
            command=createdb_command(plan);env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TMPDIR=str(D/'tmp'))
            with (D/'createdb.log').open('xb') as log:child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=D,preexec_fn=restrict,env=env)
            identity=wait_process_identity(child.pid,command,timeout=plan['identity_ready_timeout_seconds'])
            if identity is not None and (identity['argv']!=command or identity['pgid']!=child.pid):raise ValueError('newchild identity mismatch')
            write_new(D/'child-launch.json',dict(launched_utc=NOW(),identity=identity,command=command,threads=2,lossless_compressed=True,shuffle_splits=32));last=0
            while child.poll() is None:
                used=tree_bytes(D);peak=max(peak,used);reason=guard_decision(used,free(),time.monotonic()-start,plan)
                if reason:
                    if identity is None:raise ValueError('child lacks verified identity; refuse signal')
                    live('guard freezing only exact newcreatedb owner',guard_reason=reason,all_stage_bytes=used)
                    stopped=stop_owned_group(child.pid,identity,lambda:checkpoint(reason));child.wait(timeout=15);write_new(D/'owned-child-guard-stop.json',dict(reason=reason,stopped=stopped,exit_code=child.returncode));raise ValueError('finite stageguard stopped newownedchild; all newpartials retained')
                if time.monotonic()-last>=2:live('full-source createdb active; allDB/header/index/lookup/tmp files counted',all_stage_bytes=used,free_bytes=free());last=time.monotonic()
                time.sleep(.05)
            if child.returncode!=0:raise ValueError('owned createdb returned '+str(child.returncode)+'; allpartialspreserved')
            peak=max(peak,tree_bytes(D));live('createdb completed; independent all-native-index and pointwise sequence/header proof starts')
            def proofprogress(n,total,phase):
                progress.update(proof_phase=phase,records=n,total=total);reason=guard_decision(tree_bytes(D),free(),time.monotonic()-start,plan)
                if reason:raise ValueError('finite proofguard '+reason+'; completedDBretained')
                live('independent full native DB proof advancing')
            native=verify_native_database(source,D/'DB100',plan['source_rows'],32,on_progress=proofprogress)
            if digest(source)!=plan['source_sha256']:raise ValueError('immutable source changed after full proof')
            files=[]
            for p in sorted(D.rglob('*')):
                if p.is_file() and p.name!='live.json':files.append(dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)))
            if tree_bytes(D)>=plan['stage_cap_bytes']:raise ValueError('final cumulative stagecap exceeded; DBretained')
            manifest=dict(schema='mora3.MINT_full_provider_createdb_native_source_candidate.v1',completed_utc=NOW(),plan=str(pp),plan_sha256=digest(pp),approval=str(ap),approval_sha256=digest(ap),code_sha256=codes,binary_version=version,binary_sha256=plan['binary_sha256'],source_path=str(source),source_sha256=plan['source_sha256'],source_rows=plan['source_rows'],source_original_whole_SHA_before_after_pass=True,independent_every_native_header_fullAA_sequence_providerID_originalordinal_DBkey_proof=native,source_indices_not_replaced_by_new_shuffled_DBkeys=True,all_stage_files=files,stage_bytes_before_manifest=tree_bytes(D),peak_stage_bytes_observed=peak,max_child_RSS_KiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,no_network_or_GPU=True,threads=2,raw_source_not_mutated=True,preexisting_artifacts_removed_or_refunded=False,new_internal_merge_chunks_managed_only_by_pinned_tool=True,source_qualification_not_inferred_from_provider_DB=True,full_source_clustering_performed=False,published_MINT_curated_collection_reconstructed=False,model_adapter_supported=False,task_admitted_records=0)
            write_new(D/'manifest.json',manifest);live('complete native full providerDB and exact all-record source proof; clustering/admission not performed');print('MMSEQS_STAGEB_COMPLETE',plan['source_rows'],tree_bytes(D),flush=True)
        except BaseException as exc:
            if child is not None and child.poll() is None and identity is not None:
                stopped=stop_owned_group(child.pid,identity,lambda:checkpoint('wrapper exception'));child.wait(timeout=15)
            live('stopped; originalsource and allnewDBpartials/checkpoints preserved',error=type(exc).__name__+':'+str(exc));raise

if __name__=='__main__':main()
