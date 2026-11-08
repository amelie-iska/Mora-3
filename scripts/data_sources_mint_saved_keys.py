"""Reuse a frozen complete source-key proof into a NEW mutable membership workfile.
Never opens the old SQLite for writes; no cache reset, source fallback, or reindex.
"""
import hashlib,json,os,pathlib,sqlite3
from data_sources_hardware_storage import hardware_snapshot
G=1024**3

def _read(ref):
 p=pathlib.Path(ref['path'])
 if p.is_symlink() or not p.is_file() or p.stat().st_size!=ref['bytes']:raise ValueError('saved proof must be exact original regular file/length')
 b=p.read_bytes()
 if hashlib.sha256(b).hexdigest()!=ref['sha256']:raise ValueError('saved proof SHA changed')
 return json.loads(b)

def copy_verified_source_keys(spec,plan,target,check):
 target=pathlib.Path(target);source=pathlib.Path(spec['database']['path']);total=plan['source_rows']
 if target.exists() or target.is_symlink():raise FileExistsError('fresh exclusive membership workfile required')
 if spec['source_rows']!=total:raise ValueError('different whole source axis')
 oldplan=_read(spec['old_plan']);proof=_read(spec['source_key_proof']);checkpoint=_read(spec['checkpoint'])
 for key in ['source_rows','source_path','source_sha256','source_native_proof','input_components']:
  if oldplan.get(key)!=plan.get(key):raise ValueError('saved completed proof belongs to other native source/component map')
 key='scripts/data_sources_mint_full_cluster.py'
 if oldplan['code_sha256'].get(key)!=plan['code_sha256'].get(key):raise ValueError('saved source-key implementation changed')
 required=dict(source_records=total,source_provider_ids_unique=True,all_native_keys_contiguous=True,all_original_source_ordinals_bijective=True,source_file_index=0)
 if any(proof.get(k)!=v for k,v in required.items()):raise ValueError('saved whole provider/key/ordinal proof missing')
 if checkpoint.get('stable_partial_hashes') is not True or checkpoint.get('session_id')!=spec['saved_native_session_id'] or any(checkpoint.get(k,{}).get('consecutive_empty_live_thread_censuses',0)<2 for k in ['empty_before','empty_after']):raise ValueError('saved source cache not bound to proven empty whole native session')
 descriptor=checkpoint.get('retained_file_inventory',{})
 if any(descriptor.get(k)!=spec['inventory'][k] for k in ['path','bytes','sha256']):raise ValueError('frozen checkpoint inventory binding changed')
 inv=pathlib.Path(spec['inventory']['path']);h=hashlib.sha256();n=0;entries=[]
 if inv.is_symlink():raise ValueError('original inventory regular file required')
 with inv.open('rb') as f:
  for line in f:
   check();h.update(line);n+=len(line);entry=json.loads(line)
   if entry.get('path')==str(source):entries.append(entry)
 if n!=spec['inventory']['bytes'] or h.hexdigest()!=spec['inventory']['sha256'] or entries!=[spec['database']]:raise ValueError('saved database absent/ambiguous or changed in stable inventory')
 if source.is_symlink() or not source.is_file() or source.stat().st_size!=spec['database']['bytes']:raise ValueError('saved original DB regular file/length changed')
 before=source.stat();identity=(before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)
 if any(pathlib.Path(str(source)+suffix).exists() for suffix in ['-wal','-shm','-journal']):raise ValueError('unresolved saved source SQLite sidecar; no implicit replay')
 h=hashlib.sha256();n=0
 with source.open('rb') as src,target.open('xb') as out:
  for b in iter(lambda:src.read(4*1024**2),b''):
   check()
   if hardware_snapshot(target.parent)['available_bytes']-len(b)<=plan.get('minimum_free_bytes',64*G):raise ValueError('hardware free-space buffer while preserving source-key copy')
   n+=len(b)
   if n>spec['database']['bytes']:raise ValueError('saved source cache length changed during copy')
   out.write(b);h.update(b)
  out.flush();os.fsync(out.fileno())
 after=source.stat()
 if n!=spec['database']['bytes'] or h.hexdigest()!=spec['database']['sha256'] or identity!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):raise ValueError('saved source whole SHA/size/identity changed; partial NEW workfile retained, no promotion')
 check()
 conn=sqlite3.connect(target.as_uri()+'?mode=ro&immutable=1',uri=True)
 try:
  if [r[1] for r in conn.execute('PRAGMA table_info(source)')]!=['native_key','provider_id','original_source_ordinal','representative_key'] or [r[1] for r in conn.execute('PRAGMA table_info(reps)')]!=['native_key','self_member']:raise ValueError('saved exact source/member schema changed')
  if conn.execute('SELECT COUNT(*) FROM reps').fetchone()!=(0,):raise ValueError('saved key cache already has membership work; no reset')
 finally:conn.close()
 check()
 return dict(**proof,saved_progress_reused=True,saved_source_database=spec['database'],saved_empty_session_checkpoint=spec['checkpoint'],saved_plan=spec['old_plan'],new_mutable_membership_workfile=str(target),new_workfile_initial_sha256=h.hexdigest(),all_original_sources_and_saved_proofs_read_only=True,full_source_component_pre_and_post_hashes_still_required=True)
