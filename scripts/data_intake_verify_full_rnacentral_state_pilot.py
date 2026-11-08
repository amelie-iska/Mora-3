"""All SAME pilot occurrences verify the new full-source state recipe; no new tasks."""
from pathlib import Path
import datetime,functools,hashlib,json,os,resource,sqlite3,sys,time
import pyarrow as pa,pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_full_rnacentral_state import NativeIdentityIndex,old_identity_resolver
B=R/'outputs/data-intake/20261101-RNAcentral-corpus-pilot-v1';O=R/'outputs/data-intake/20261101-RNAcentral-full-state-conservation-v1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while x:=f.read(4*2**20):h.update(x)
 return h.hexdigest()
def ref(p,**kw):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p),**kw)
def check(f):
 p=R/f['path'];assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256'];return p
@functools.lru_cache(maxsize=2)
def source(path,digest):
 p=R/path;assert sha(p)==digest;return pq.read_table(p).to_pylist()
def main():
 start=time.monotonic();os.sched_setaffinity(0,{30});os.nice(10);pa.set_cpu_count(1);pa.set_io_thread_count(1);assert not O.exists();O.mkdir();(O/'code').mkdir()
 for n in [Path(__file__).name,'data_intake_full_rnacentral_state.py']:(O/'code'/n).open('xb').write((R/'scripts'/n).read_bytes())
 m=json.loads((B/'manifest.json').read_text());pilot=check(m['new_identity_and_group_index']);rfam=R/'outputs/data-intake/20261101-nucleotide-corpus-v1/full-v1/task_identity_and_group_index.sqlite3';before={str(p):sha(p) for p in [pilot,rfam]};old=sqlite3.connect('file:'+str(rfam)+'?mode=ro&immutable=1',uri=True);pi=sqlite3.connect('file:'+str(pilot)+'?mode=ro&immutable=1',uri=True);resolve=old_identity_resolver(old,pi);new=NativeIdentityIndex.create(O/'compact-state.sqlite');observed=0;letters=0
 for file,ledger in enumerate(m['occurrences']):
  data=pq.read_table(check(ledger)).to_pylist()
  for r in data:
   sr=json.loads(r['source_ref_json']);native=source(sr['native_payload'],sr['native_payload_sha256'])[sr['native_row_index']];seq=native['native_sequence'];assert sha_literal(seq)==r['native_sequence_sha256'];x=new.observe(seq,native['source_identifier'],file,sr['native_row_index'],resolver=resolve);assert x['task_id']==r['task_id'] and x['first_occurrence']==r['first_occurrence_in_pilot'];assert x['uppercase_sha'].hex()==r['uppercase_sequence_sha256'] and x['rc_sha'].hex()==r['derived_RC_baseorder_sha256'];assert (x['reuse_kind']=='Rfam_existing')==r['task_already_in_Rfam_corpus'];observed+=1;letters+=len(seq)
  new.commit();st=os.statvfs(R);assert st.f_bavail*st.f_frsize>64*2**30 and time.monotonic()-start<600 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<8*2**30;print('SAME_PILOT_STATE_CHECK',file+1,'rows',observed,'seconds',round(time.monotonic()-start,2),flush=True)
 counts=new.summary();assert observed==m['counts']['quality_pass_occurrences']==446147 and counts['new_sequences']==0 and counts['reused_Rfam_sequences']==661 and counts['reused_pilot_sequences']==445486;assert not list(new.identifier_conflicts());new.close();old.close();pi.close();assert all(sha(Path(p))==digest for p,digest in before.items())
 receipt=dict(schema='mora3.FullRNAcentralStateSamePilotConservation/v1',status='verified',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_pilot=ref(B/'manifest.json'),old_readonly_indices=[ref(Path(p)) for p in before],all_old_index_bytes_unchanged=True,source_occurrences=observed,full_native_sequence_letters=letters,counts=counts,all_old_task_IDs_native_groups_first_refs_and_URS_bindings_preserved=True,case_T_U_native_axes_not_collapsed=True,new_task_views=0,master_changed=False,model_training=False,complete_full61M_source_build_launched=False,new_state_index=ref(O/'compact-state.sqlite'),elapsed_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
 (O/'receipt.json').open('x').write(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2),flush=True)
def sha_literal(s):return hashlib.sha256(s.encode()).hexdigest()
if __name__=='__main__':main()
