"""Fresh MINTv5 path/environment and additive physical-resource contracts.
No dataset opens, process launches, path creation or native scientific work.
"""
import pathlib
G=1024**3
UNSET=('OMP_PROC_BIND','OMP_PLACES','PYTHONPYCACHEPREFIX')
def expected_environment(stage):
 d=pathlib.Path(stage)
 return dict(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',ARROW_NUM_THREADS='1',RUNNER='',MPI_RUNNER='',
 TMPDIR=str(d/'tmp'),SQLITE_TMPDIR=str(d/'tmp'),TMP=str(d/'tmp'),TEMP=str(d/'tmp'),XDG_CACHE_HOME=str(d/'tmp/cache'),
 PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
def validate_scope(p,stage,lease,outer_log,commands):
 stage=pathlib.Path(stage);lease=pathlib.Path(lease);outer_log=pathlib.Path(outer_log)
 if any(p.get(k)!=str(v) for k,v in [('new_output',stage),('lease_path',lease),('outer_log',outer_log)]):raise ValueError('exact fresh v5 stage/lease/outerlog write allowlist required')
 for path in [stage,lease,outer_log]:
  if not path.is_absolute() or path.resolve()!=path or path.is_symlink():raise ValueError('canonical regular new namespace without symlink parent required')
 if p.get('native_environment_overrides')!=expected_environment(stage) or p.get('native_environment_unset')!=list(UNSET):raise ValueError('all declared temporary/cache writes must target new v5 only')
 expected=[dict(stage=name,argv=argv) for name,argv in commands]
 if p.get('commands')!=expected:raise ValueError('exact pinned scientific argv and fresh path operands required')
 for _,argv in commands:
  for value in argv[1:]:
   q=pathlib.Path(value)
   if q.is_absolute() and not q.is_relative_to(stage):raise ValueError('native operand escapes new owned namespace')
 return dict(stage_tree=str(stage),lease_file=str(lease),outer_log_file=str(outer_log),temporary_and_cache_paths=expected_environment(stage),
 input_aliases='only declared eight read-only original component symlinks; wholeSHAprepost still mandatory',existing_sources_or_stages_writable=False)
def native_environment(p,base):
 env=dict(base)
 for key in UNSET:env.pop(key,None)
 declared=expected_environment(p['new_output'])
 if p.get('native_environment_overrides')!=declared or p.get('native_environment_unset')!=list(UNSET):raise ValueError('reviewed and actual native write environment differs')
 env.update(declared);return env

def combine_pressure(owned,outer):
 ids=[(m['pid'],m['start_ticks']) for m in owned['members']+outer['members']]
 if len(ids)!=len(set(ids)):raise ValueError('overlapping wrapper/owned pressure scopes')
 return dict(owned_session_pss_bytes=owned['owned_session_pss_bytes']+outer['owned_session_pss_bytes'],
 owned_session_rss_bytes=owned['owned_session_rss_bytes']+outer['owned_session_rss_bytes'],
 owned_session_swap_bytes=owned['owned_session_swap_bytes']+outer['owned_session_swap_bytes'],host_mem_available_bytes=min(owned['host_mem_available_bytes'],outer['host_mem_available_bytes']),
 members=owned['members']+outer['members'],owned_session_id=owned['session_id'],wrapper_session_id=outer['session_id'],
 physical_metric='aggregate exact owned workflow SID plus isolated outer wrapper SID; PSS not per-process AS')
def require_resource_authorization(p):
 a=p.get('existing_hardware_resource_authorization',{});m=a.get('known_pending_MINT_RAM',{})
 exact=dict(proposed_ceiling_bytes=40*G,stop_margin_bytes=8*G,stop_at_bytes=32*G,start_host_MemAvailable_floor_bytes=48*G,
 runtime_host_MemAvailable_floor_bytes=16*G,swap_ceiling_bytes=256*1024**2,per_process_AS_bytes=64*G,root_exact_new_safe_package_review_still_required=True)
 if a.get('authorized') is not True or any(m.get(k)!=v for k,v in exact.items()):raise ValueError('exact latest USER hardware authorization and separate root launch review required')
 return True

def verify_readonly_stats(bindings):
 import stat
 for name,expected in bindings.items():
  p=pathlib.Path(name);s=p.lstat()
  if not stat.S_ISREG(s.st_mode):raise ValueError('original read-only input must remain a regular file')
  got=dict(dev=s.st_dev,ino=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
  if got!=expected:raise ValueError('original input identity or same-byte rewrite detected')
 return True

def validate_readonly_inputs(p):
 paths={p['source_path'],p['reuse_source_key_proof']['database']['path']}
 paths|={x['path'] for x in p['input_components']}
 paths|={x['path'] for x in p['reuse_completed_prefilter']['completed_pref_components']}
 if len(paths)!=16 or set(p.get('source_stat_bindings',{}))!=paths:raise ValueError('exact original FASTA/eightDB/keyproof/sixpref read-only stat map required')
 return verify_readonly_stats(p['source_stat_bindings'])
