"""Copy only a proved complete native prefilter into a fresh MMseqs resume stage.
No writes to saved stages, opaque best-effort resume, parameter changes, or kmer fallback.
"""
import hashlib,json,os,pathlib,re
from data_sources_hardware_storage import hardware_snapshot
from data_sources_mint_full_guard_v2 import _wait_empty_session
from data_sources_mint_full_cluster import cluster_commands
G=1024**3

def _read(ref):
 p=pathlib.Path(ref['path'])
 if p.is_symlink() or not p.is_file() or p.stat().st_size!=ref['bytes'] or ref['bytes']>4*1024**2:raise ValueError('bounded original regular metadata reference required')
 data=p.read_bytes()
 if hashlib.sha256(data).hexdigest()!=ref['sha256']:raise ValueError('saved stage metadata SHA changed')
 return json.loads(data)

def _identity(p):
 s=p.stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)

def _validate_spec(spec,plan):
 old=_read(spec['old_plan']);cp=_read(spec['checkpoint']);census=_read(spec['prefilter_index_census']);total=plan['source_rows']
 for k in ['source_rows','source_path','source_sha256','source_native_proof','input_components','binary','binary_sha256','binary_version','threads']:
  if old.get(k)!=plan.get(k):raise ValueError('saved completed prefilter belongs to other source/binary/command parameters')
 key='scripts/data_sources_mint_full_cluster.py'
 if old['code_sha256'].get(key)!=plan['code_sha256'].get(key):raise ValueError('saved original scientific command code changed')
 if spec['source_rows']!=total or cp.get('stable_partial_hashes') is not True or cp.get('session_id')!=spec['saved_native_session_id'] or any(cp.get(k,{}).get('consecutive_empty_live_thread_censuses',0)<2 for k in ['empty_before','empty_after']):raise ValueError('saved exact whole-session stable checkpoint missing')
 ir=spec['inventory'];descriptor=cp.get('retained_file_inventory',{})
 if any(descriptor.get(k)!=ir[k] for k in ['path','bytes','sha256']):raise ValueError('saved checkpoint inventory binding changed')
 ip=pathlib.Path(ir['path'])
 if ip.is_symlink() or not ip.is_file() or ip.stat().st_size!=ir['bytes'] or ir['bytes']>4*1024**2:raise ValueError('bounded original regular inventory required')
 data=ip.read_bytes()
 if hashlib.sha256(data).hexdigest()!=ir['sha256']:raise ValueError('saved inventory SHA changed')
 inventory=[json.loads(line) for line in data.splitlines()];components=spec['completed_pref_components'];names=[pathlib.Path(x['path']).name for x in components]
 if len(names)!=len(set(names)) or names.count('pref.index')!=1 or names.count('pref.dbtype')!=1:raise ValueError('exact complete native prefilter component list required')
 datanames=sorted(set(names)-{'pref.index','pref.dbtype'})
 if datanames!=['pref']:
  if not datanames or any(not re.fullmatch(r'pref\.[0-9]+',x) for x in datanames) or sorted(int(x.split('.')[1]) for x in datanames)!=list(range(len(datanames))):raise ValueError('missing/noncontiguous native prefilter data components')
 for k in ['top_tmp_hash','linclust_tmp_hash']:
  if not re.fullmatch(r'[0-9]{1,20}',spec[k]):raise ValueError('unsafe or unknown exact native temporary hash')
 oldroot=pathlib.Path(old['new_output']);leaf=oldroot/'tmp'/spec['top_tmp_hash']/'linclust'/spec['linclust_tmp_hash']
 if not oldroot.is_absolute() or oldroot.resolve()!=oldroot:raise ValueError('saved original stage path/alias changed')
 for item in components:
  p=pathlib.Path(item['path'])
  if p.parent!=leaf or p.is_symlink() or not p.is_file() or p.stat().st_size!=item['bytes'] or [x for x in inventory if x.get('path')==item['path']]!=[item]:raise ValueError('saved complete prefilter file/path/inventory changed')
 index=next(x for x in components if x['path'].endswith('/pref.index'))
 if census.get('pref_index_path')!=index['path'] or census.get('pref_index_sha256')!=index['sha256'] or census.get('pref_index_bytes')!=index['bytes'] or census.get('rows')!=total or census.get('all_native_keys_contiguous_once') is not True:raise ValueError('whole completed prefilter native key/index census missing')
 if next(x for x in components if x['path'].endswith('/pref.dbtype'))['bytes']!=4:raise ValueError('native completion dbtype marker malformed')
 return components

def copy_verified_prefilter(spec,plan,newstage,check):
 newstage=pathlib.Path(newstage);components=_validate_spec(spec,plan);sid=spec['saved_native_session_id'];_wait_empty_session(sid,timeout=5)
 if not newstage.is_absolute():newstage=newstage.resolve()
 if newstage.is_symlink() or not newstage.is_dir() or (newstage/'tmp').is_symlink() or not (newstage/'tmp').is_dir():raise ValueError('fresh regular owned stage/tmp directory required')
 top=newstage/'tmp'/spec['top_tmp_hash'];latest=newstage/'tmp/latest'
 if top.exists() or top.is_symlink() or latest.exists() or latest.is_symlink():raise FileExistsError('fresh native resume tree/latest required; no reset/overwrite')
 leaf=top/'linclust'/spec['linclust_tmp_hash'];leaf.mkdir(parents=True,exist_ok=False);copied=[]
 for item in components:
  check();source=pathlib.Path(item['path']);before=_identity(source);target=leaf/source.name;h=hashlib.sha256();count=0
  with source.open('rb') as src,target.open('xb') as out:
   for block in iter(lambda:src.read(4*1024**2),b''):
    check()
    if hardware_snapshot(newstage)['available_bytes']-len(block)<=plan.get('minimum_free_bytes',64*G):raise ValueError('hardware free-space buffer during immutable prefilter copy')
    count+=len(block)
    if count>item['bytes']:raise ValueError('saved prefilter grew during copy')
    out.write(block);h.update(block)
   out.flush();os.fsync(out.fileno())
  if count!=item['bytes'] or h.hexdigest()!=item['sha256'] or before!=_identity(source):raise ValueError('saved prefilter whole SHA/size/identity changed; new partial retained unpromoted')
  copied.append(dict(path=str(target),bytes=count,sha256=h.hexdigest(),source=item,source_initial_identity=list(before),new_initial_identity=list(_identity(target))))
 _wait_empty_session(sid,timeout=5);check()
 latest.symlink_to(spec['top_tmp_hash']);(top/'linclust/latest').symlink_to(spec['linclust_tmp_hash'])
 return dict(prefilter_records=plan['source_rows'],completed_initial_kmer_reused=True,source_checkpoint=spec['checkpoint'],source_inventory=spec['inventory'],source_index_census=spec['prefilter_index_census'],copied_components=copied,only_complete_prefilter_seeded=True,old_partial_clu_or_scripts_not_copied=True,all_saved_sources_readonly=True,provider_force_reuse_parameter_explicit=True,native_kmer_skip_depends_only_on_exact_cloned_complete_pref_dbtype=True,no_fullsource_reindex_or_fallback=True)

def verify_prefilter_unchanged(spec,newstage,proof,check):
 leaf=pathlib.Path(newstage)/'tmp'/spec['top_tmp_hash']/'linclust'/spec['linclust_tmp_hash'];checked=[]
 if proof.get('source_checkpoint')!=spec['checkpoint'] or proof.get('prefilter_records')!=spec['source_rows']:raise ValueError('exact pre-native copy identity proof required')
 identities={x['source']['path']:x for x in proof['copied_components']}
 if set(identities)!={x['path'] for x in spec['completed_pref_components']}:raise ValueError('copied prefilter identity proof incomplete')
 for item in spec['completed_pref_components']:
  for p,identity_key in [(pathlib.Path(item['path']),'source_initial_identity'),(leaf/pathlib.Path(item['path']).name,'new_initial_identity')]:
   if list(_identity(p))!=identities[item['path']][identity_key]:raise ValueError('original/copied completed prefilter identity changed or samebyte rewrite; cannot claim skipped work')
   if p.is_symlink() or not p.is_file() or p.stat().st_size!=item['bytes']:raise ValueError('original/copied completed prefilter component changed')
   before=_identity(p);h=hashlib.sha256()
   with p.open('rb') as f:
    for block in iter(lambda:f.read(8*1024**2),b''):check();h.update(block)
   if h.hexdigest()!=item['sha256'] or before!=_identity(p):raise ValueError('completed prefilter changed during native continuation')
   checked.append(dict(path=str(p),sha256=h.hexdigest(),bytes=item['bytes']))
 return dict(original_and_new_completed_prefilter_whole_SHA_after_native_pass=True,checked_components=checked)

def resumed_cluster_commands(binary,out):
 commands=cluster_commands(binary,out);commands[0][1].extend(['--force-reuse','1']);return commands
