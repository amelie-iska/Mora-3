"""Reviewed one-shot guard around the existing RNA pointer publisher.

This wrapper does not construct or admit new tasks, activate a mixture, invoke a
model, or infer neural support from a source-code inventory. Every prior byte is
retained. A failed or interrupted publication is retained for explicit review;
there is no automatic retry, cleanup, rollback, or pointer repair.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import resource
import runpy
import shutil
import signal
import sys
import time

ROOT = Path('/home/iska/Documents/amelie/bio/Mora-3')
SCOPE = 'whole_RNAcentral_pointer_publication_v1'
COUNTS = dict(tasks=81054784, shards=82141, task_bytes=37333973228,
              splits=dict(train=81034656, validation=10076, test=10052))
PROGRESS_ALLOWED = set('updated_utc status running_jobs current_master_corpus_manifest current_corpus_manifest current_corpus_manifest_sha256 final_corpus_manifest current_corpus_counts reader_readiness executable_model_handoff current_handoff model_handoff RNAcentral_whole_verified_corpus RNAcentral_whole_native_counts final_full_training_dataset_complete current_owned_task next_job full_dataset_readiness_signature'.split())
QUEUE_ALLOWED = set('updated_utc current_corpus_manifest current_versioned_model_handoff full_dataset_ready substantive_training_before_full_dataset_ready entries native_consumer'.split())
ENTRY_ALLOWED = set('resolution_status next_action full_native_corpus evidence_paths'.split())
NATIVE_ALLOWED = set('manifest readiness_table executable_handoff'.split())
MUTABLE = ['planning/data_intake_progress.json', 'configs/dataset_acquisition_queue.json']


def reference(path, root=ROOT):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return dict(path=str(path.relative_to(root)), bytes=path.stat().st_size,
                sha256=digest.hexdigest())


def validate_pin(pin, root=ROOT):
    p = Path(pin['path'])
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError('invalid relative input path')
    if reference(root / p, root) != pin:
        raise ValueError('pinned file changed: ' + str(p))


def backup_exact(source, dest, pin, root=ROOT):
    validate_pin(pin, root)
    raw = source.read_bytes()
    if len(raw) != pin['bytes'] or hashlib.sha256(raw).hexdigest() != pin['sha256']:
        raise ValueError('mutable input changed while copying')
    with dest.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    if dest.read_bytes() != raw:
        raise ValueError('backup byte readback failed')
    validate_pin(pin, root)


def validate_approval(a, plan_sha, script_sha):
    checks = dict(approved=True, scope=SCOPE, plan_sha256=plan_sha,
                  script_sha256=script_sha, pointer_publication=True,
                  new_scientific_admission=False, model_use=False,
                  training=False, full_dataset_ready=False)
    for key, value in checks.items():
        actual = a.get(key)
        if isinstance(value, bool):
            good = actual is value
        else:
            good = actual == value
        if not good:
            raise ValueError('approval missing or mismatched: ' + key)


def unchanged_except(before, after, allowed, label):
    keys = set(before) | set(after)
    if any(before.get(k) != after.get(k) or (k in before) != (k in after)
           for k in keys - set(allowed)):
        raise ValueError('undeclared mutation: ' + label)


def verify_mutations(p, pn, q, qn, expected):
    unchanged_except(p, pn, PROGRESS_ALLOWED, 'progress')
    unchanged_except(q, qn, QUEUE_ALLOWED, 'queue')
    for key in ['current_corpus_manifest', 'current_master_corpus_manifest', 'final_corpus_manifest']:
        if pn.get(key) != expected['master']:
            raise ValueError('progress master pointer mismatch')
    if pn.get('final_full_training_dataset_complete') is not False or pn.get('running_jobs') != []:
        raise ValueError('progress readiness/live-job gate changed')
    if pn.get('current_corpus_counts') != COUNTS:
        raise ValueError('task/count/split conservation failed')
    if qn.get('full_dataset_ready') is not False or qn.get('substantive_training_before_full_dataset_ready') is not False:
        raise ValueError('queue scientific/training gate changed')
    if qn.get('current_corpus_manifest') != expected['master']:
        raise ValueError('queue master pointer mismatch')
    hd = expected['handoff_dir']
    if qn.get('current_versioned_model_handoff') != hd + '/executable-model-handoff.json':
        raise ValueError('queue handoff pointer mismatch')
    unchanged_except(q['native_consumer'], qn['native_consumer'], NATIVE_ALLOWED, 'native consumer')
    for key, value in dict(manifest=expected['master'], readiness_table=hd+'/reader-readiness.json', executable_handoff=hd+'/executable-model-handoff.json').items():
        if qn['native_consumer'].get(key) != value:
            raise ValueError('native-consumer pointer mismatch')
    if len(q['entries']) != len(qn['entries']):
        raise ValueError('inventory entries changed')
    for old, new in zip(q['entries'], qn['entries']):
        if old['inventory_id'] != new['inventory_id']:
            raise ValueError('inventory ID/order changed')
        unchanged_except(old, new, ENTRY_ALLOWED if old['inventory_id']=='INV-004' else set(), 'inventory entry')
        if old['inventory_id']=='INV-004' and new.get('evidence_paths', [])[:len(old.get('evidence_paths', []))] != old.get('evidence_paths', []):
            raise ValueError('old source evidence was not preserved')


def verify_handoff(rd, h, master):
    if rd.get('master') != master or h.get('master') != master:
        raise ValueError('handoff master mismatch')
    if rd['counts']['tasks'] != COUNTS['tasks'] or rd['counts']['shards'] != COUNTS['shards'] or rd['counts']['task_data_bytes'] != COUNTS['task_bytes']:
        raise ValueError('handoff counts mismatch')
    if sum(f['serialized_tasks'] for f in rd['families']) != COUNTS['tasks']:
        raise ValueError('primary-family counts do not conserve task views')
    for d, nulls, falses in [
        (rd, ['actual_whole_model_token_totals','weights','full_dataset_readiness_signature'], ['final_full_training_dataset_complete','substantive_training_authorized','global_leakage_safe']),
        (h, ['full_dataset_readiness_signature','actual_whole_model_tokens','actual_optimizer_exposure'], ['current_full_dataset_complete','substantive_training_authorized'])]:
        if any(key not in d or d[key] is not None for key in nulls) or any(d.get(key) is not False for key in falses):
            raise ValueError('unready/unmeasured/model/training gate changed')
    if h.get('optimizer_steps_from_publisher') != 0:
        raise ValueError('publisher cannot claim optimizer use')


def file_stat(p):
    s=p.stat()
    return dict(bytes=s.st_size,inode=s.st_ino,device=s.st_dev,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)


def verify_index_stats(index_evidence, root=ROOT):
    for row in index_evidence:
        p = root / row['expected_full_SHA_reference']['path']
        if file_stat(p) != row['current_stat']:
            raise ValueError('reviewed closing-index state changed')
        if any(Path(str(p)+suffix).exists() for suffix in ['-journal','-wal','-shm']):
            raise ValueError('source index has an unexpected journal')


def write_new(path, obj):
    with path.open('x') as f:
        f.write(json.dumps(obj, indent=2)+'\n')


def load(path):
    return json.loads(path.read_text())


def ancestor_pids():
    result={os.getpid()}; pid=os.getpid()
    while pid > 1:
        try:
            raw=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
            pid=int(raw[1]);result.add(pid)
        except (FileNotFoundError,ProcessLookupError):break
    return result


def require_no_other_RNA_owner():
    ancestors=ancestor_pids()
    for d in Path('/proc').iterdir():
        if not d.name.isdigit() or int(d.name) in ancestors:continue
        try:argv=(d/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        for arg in argv:
            name=Path(os.fsdecode(arg)).name
            if (name.startswith('data_intake_full_rnacentral_') or name.startswith('data_intake_publish_full_rnacentral_')) and name.endswith('.py'):
                raise RuntimeError('another RNA phase/publisher owns a live process: '+d.name)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--approval',required=True);a=parser.parse_args()
    if not __debug__:raise RuntimeError('publisher assertions require ordinary Python mode')
    os.chdir(ROOT);start=time.monotonic();plan_path=ROOT/a.plan;plan=load(plan_path)
    validate_approval(load(ROOT/a.approval),reference(plan_path)['sha256'],reference(Path(__file__).resolve())['sha256'])
    if plan['schema']!='mora3.RNAPointerPublicationExecutionPlan/v1':raise ValueError('wrong plan schema')
    closure=load(ROOT/plan['closure']['path'])
    for pin in plan['input_pins']:validate_pin(pin)
    verify_index_stats(closure['giant_index_open_close_hash_evidence'])
    if not closure['owners_absent'] or closure['SID_members_twice']!=[[],[]] or closure['publisher_invoked']:
        raise ValueError('C2 normal closure incomplete')
    require_no_other_RNA_owner()
    output=ROOT/plan['run_directory'];handoff=ROOT/plan['handoff_directory']
    if output.exists() or handoff.exists():raise FileExistsError('publication requires fresh output and handoff directories')
    if set(plan['mutable_inputs'])!=set(MUTABLE):raise ValueError('unexpected mutable target set')
    if len(load(ROOT/MUTABLE[1])['entries'])!=114:raise ValueError('inventory coverage changed')
    current=load(ROOT/MUTABLE[0])
    if current['current_master_corpus_manifest']!=plan['baseline_master']['path']:raise ValueError('competing master pointer changed')
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('pointer publication finite120s deadline')))
    signal.alarm(120)
    def guard():
        if time.monotonic()-start>=120:raise TimeoutError('pointer publication deadline')
        if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime.fromisoformat(plan['original_deadline_utc']):raise TimeoutError('original whole-RNA deadline')
        if sorted(os.sched_getaffinity(0))!=[27] or os.getpriority(os.PRIO_PROCESS,0)<10:raise RuntimeError('CPU/nice policy mismatch')
        free=shutil.disk_usage(ROOT).free
        if free < 65*1024**3:raise RuntimeError('hardware64GiB floor plus1GiB write reserve')
        mem={line.split(':',1)[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line}
        if mem['MemAvailable']<8*1024**3:raise RuntimeError('host available RAM guard')
        return dict(elapsed_seconds=time.monotonic()-start,free_bytes=free,host_mem_available_bytes=mem['MemAvailable'],own_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,max_address_space_bytes=3*1024**3,pid=os.getpid(),parent_pid=os.getppid(),SID=os.getsid(0),start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),CPU_affinity=sorted(os.sched_getaffinity(0)),nice=os.getpriority(os.PRIO_PROCESS,0),argv=sys.argv)
    initial=guard();output.mkdir();backups=output/'backups';backups.mkdir()
    try:
        before={f:load(ROOT/f) for f in MUTABLE};backup_refs=[]
        for f in MUTABLE:
            pin=next(p for p in plan['input_pins'] if p['path']==f)
            dest=backups/Path(f).name;backup_exact(ROOT/f,dest,pin);backup_refs.append(reference(dest))
        write_new(output/'prepublication-backup-receipt.json',dict(approval=reference(ROOT/a.approval),plan=reference(plan_path),initial_guard=initial,backups=backup_refs,policy_activation=False,training=False))
        for pin in plan['input_pins']:validate_pin(pin)
        guard();require_no_other_RNA_owner()
        publisher=runpy.run_path(str(ROOT/plan['publisher']['path']),run_name='RNA_pointer_publisher_reviewed')
        publisher['main']()
        pn=load(ROOT/MUTABLE[0]);qn=load(ROOT/MUTABLE[1])
        verify_mutations(before[MUTABLE[0]],pn,before[MUTABLE[1]],qn,dict(master=plan['staged_master']['path'],handoff_dir=plan['handoff_directory']))
        expected_names=set(plan['new_handoff_filenames'])
        if {p.name for p in handoff.iterdir()}!=expected_names:raise ValueError('unexpected handoff artifact set')
        verify_handoff(load(handoff/'reader-readiness.json'),load(handoff/'executable-model-handoff.json'),plan['staged_master'])
        for pin in plan['input_pins']:
            if pin['path'] not in MUTABLE:validate_pin(pin)
        verify_index_stats(closure['giant_index_open_close_hash_evidence'])
        for br in backup_refs:validate_pin(br)
        final=guard()
        write_new(output/'receipt.json',dict(schema='mora3.RNAPointerPublicationClosure/v1',status='normal_complete',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),approval=reference(ROOT/a.approval),plan=reference(plan_path),C2_closure=plan['closure'],master=plan['staged_master'],backups=backup_refs,mutable_after=[reference(ROOT/f) for f in MUTABLE],new_handoff=[reference(p) for p in sorted(handoff.iterdir())],counts=COUNTS,final_guard=final,giant_indexes_rehashed_here=False,model_code_or_weights_changed=False,new_task_construction=False,new_scientific_admission=False,full_dataset_ready=False,model_use=False,mixture_activated=False,training=False,actual_optimizer_exposure=None,actual_model_tokens=None))
        print('COMPLETE_FULL_RNACENTRAL_POINTER_PUBLICATION '+json.dumps(reference(output/'receipt.json')),flush=True)
    except BaseException as exc:
        try:write_new(output/'failure.json',dict(status='retained_publication_failure_no_retry_or_rollback',type=type(exc).__name__,message=str(exc),elapsed_seconds=time.monotonic()-start,mutable_current=[reference(ROOT/f) for f in MUTABLE],publisher_handoff_exists=handoff.exists(),no_automatic_cleanup=True))
        except BaseException:pass
        raise
    finally:signal.alarm(0)


if __name__=='__main__':main()
