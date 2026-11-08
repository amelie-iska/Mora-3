"""Bounded immutable closed-prefix reader fixture; zero corpus admission."""
from pathlib import Path
import hashlib,json,os,resource,sqlite3,time,datetime,sys
import pyarrow as pa,pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
B=R/'outputs/data-intake/20261101-RNAcentral-full-native-corpus-v1';O=B/'closed-native-reader-fixture-v1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while x:=f.read(2**20):h.update(x)
 return h.hexdigest()
def ref(p,**kw):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p),**kw)
def native_json(x):return {k:(v.hex() if isinstance(v,bytes) else v) for k,v in x.items()}
def main():
 start=time.monotonic();os.sched_setaffinity(0,{26,27});os.nice(10);pa.set_cpu_count(1);assert not O.exists();O.mkdir();snapshot=json.loads((B/'materialization-live-progress.json').read_text());closed=snapshot['counts']['new_task_shards'];assert closed>300
 def guard():
  assert time.monotonic()-start<180 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<512*2**20
  st=os.statvfs(R);assert st.f_bavail*st.f_frsize>=64*2**30
 guard();cpath=B/'census/manifest.json';assert sha(cpath)=='d6240751e8ba21744bd9869ccac0c4ea74809bbaa3dca3c456144b176e1020bd';c=json.loads(cpath.read_text());dbpath=R/c['native_identity_URS_index']['path'];db=sqlite3.connect('file:'+str(dbpath)+'?mode=ro&immutable=1',uri=True);selected={};rows_by_key={};read_prefix=0
 for i in range(min(256,closed-2)):
  guard();p=B/f'materialized/shards/train/{i:06d}.parquet';before=p.stat();t=pq.read_table(p,columns=['full_axis_values']);lengths=t['full_axis_values'].to_pylist();indices=[]
  if 'short' not in selected:
   indices.extend((j,'short') for j,v in enumerate(lengths) if v<=512)
  if 'long_gt8192' not in selected:indices.extend((j,'long_gt8192') for j,v in enumerate(lengths) if v>8192)
  # At most the first 16 closed payloads are searched for ambiguity/native U.
  if i<16 or indices:
   allrows=pq.read_table(p).to_pylist()
   for j,row in enumerate(allrows):
    sequence=json.loads(row['native_payload'])['targets']['source_native_sequence']
    if i<16 and 'ambiguity' not in selected and any(x.upper() not in 'ACGTU' for x in sequence):indices.append((j,'ambiguity'))
    if i<16 and 'native_U' not in selected and 'U' in sequence.upper():indices.append((j,'native_U'))
   for j,label in indices:
    if label in selected:continue
    selected[label]=(i,j);rows_by_key[(i,j)]=allrows[j]
  assert p.stat().st_size==before.st_size and p.stat().st_mtime_ns==before.st_mtime_ns
  read_prefix=i+1
  if all(k in selected for k in ['short','long_gt8192','ambiguity']) and i>=15:break
 assert all(k in selected for k in ['short','long_gt8192','ambiguity']),selected
 native_rows=[];evidence=[];source_cache={}
 def source_file(fordinal):
  f=c['source_file_dictionary'][fordinal];p=R/f['source']['path'];assert sha(p)==f['source']['sha256'];return f,p
 def get_row(path,index):
  pf=pq.ParquetFile(path);offset=0
  for group in range(pf.num_row_groups):
   n=pf.metadata.row_group(group).num_rows
   if offset+n>index:return pf.read_row_group(group).slice(index-offset,1).to_pylist()[0]
   offset+=n
  raise ValueError('native row index out of bounds')
 for n,((si,ri),row) in enumerate(rows_by_key.items()):
  guard();p=B/f'materialized/shards/train/{si:06d}.parquet';gp=B/f'materialized/effective-groups/{si:06d}.parquet';r1=ref(p);gr1=ref(gp);group=get_row(gp,ri);assert group['task_id']==row['task_id'] and json.loads(group['effective_groups_json'])==json.loads(row['group_ids_json']);payload=json.loads(row['native_payload']);sequence=payload['targets']['source_native_sequence'];digest=hashlib.sha256(sequence.encode()).digest();assert sha_bytes(row['native_payload'])==row['native_payload_sha256'];state=db.execute('SELECT first_file,first_row,reuse_kind,occurrences,uppercase_sha,rc_sha FROM sequences WHERE native_sha=?',(digest,)).fetchone();assert state and state[2]=='new';ids=[g.split(':',1)[1] for g in json.loads(group['effective_groups_json']) if g.startswith('rnacentral_native_identifier:')];assert ids
  assert all(db.execute('SELECT 1 FROM identifiers WHERE identifier=? AND native_sha=?',(ident,digest)).fetchone() for ident in ids)
  f,sp=source_file(state[0]);sr=get_row(sp,state[1]);assert sr['native_sequence']==sequence;lp=R/f['occurrences']['path'];assert sha(lp)==f['occurrences']['sha256'];occ=get_row(lp,state[1]);assert occ['native_sha']==digest and occ['task_identity_digest'].hex()==row['task_id'];source_ref=payload['audit']['source_ref'];assert source_ref['native_row_index']==state[1] and source_ref['native_payload']==f['source']['path'];assert sr['source_identifier'] in ids
  out=O/f'case-{n:02d}.parquet';pq.write_table(pa.Table.from_pylist([row]),out,compression='zstd');assert pq.read_table(out).to_pylist()==[row];native_rows.append(sr);evidence.append(dict(case=ref(out,rows=1),labels=[k for k,v in selected.items() if v==(si,ri)],closed_original_shard=r1,source_shard_row_index=ri,closed_original_group=gr1,selected_group_row=group,native_source_file=f['source'],native_source_row_index=state[1],census_occurrence_file=f['occurrences'],selected_occurrence=native_json(occ),frozen_identity_state=dict(first_file=state[0],first_row=state[1],reuse_kind=state[2],occurrences=state[3],uppercase_sha256=state[4].hex(),rc_guard_sha256=state[5].hex()),literal_native_identifiers_from_closed_group=ids,identifier_membership_proved_by_exact_frozen_PK_lookup=True,complete_identifier_enumeration_not_independently_replayed_for_fixture=True,full_axis_values=len(sequence),biological_ACGTU_positions=sum(x.upper() in 'ACGTU' for x in sequence),ambiguity_positions=sum(x.upper() not in 'ACGTU' for x in sequence),native_T_symbols=sequence.upper().count('T'),native_U_symbols=sequence.upper().count('U'),source_reference_not_material_or_modification_identity=True));assert ref(p)==r1 and ref(gp)==gr1
 # Existing-reference witnesses: query one exact source-native identity of each role.
 reuse=[]
 for kind in ['RNAcentral_pilot_existing','Rfam_existing']:
  guard();rec=db.execute('SELECT native_sha,first_file,first_row FROM sequences WHERE reuse_kind=? LIMIT 1',(kind,)).fetchone();assert rec;digest,fo,ro=rec;f,sp=source_file(fo);sr=get_row(sp,ro);lp=R/f['occurrences']['path'];assert sha(lp)==f['occurrences']['sha256'];occ=get_row(lp,ro);assert occ['native_sha']==digest and occ['reuse_kind']==kind;tid=occ['task_identity_digest'].hex()
  if kind=='RNAcentral_pilot_existing':
   op=R/'outputs/data-intake/20261101-RNAcentral-corpus-pilot-v1/native-task-and-source-group-index.sqlite3';old=sqlite3.connect('file:'+str(op)+'?mode=ro&immutable=1',uri=True);proof=old.execute('SELECT task_id,reused_prior FROM sequences WHERE native_sha=?',(digest.hex(),)).fetchone();assert proof and proof[0]==tid and proof[1]==0
  else:
   op=R/'outputs/data-intake/20261101-nucleotide-corpus-v1/full-v1/task_identity_and_group_index.sqlite3';old=sqlite3.connect('file:'+str(op)+'?mode=ro&immutable=1',uri=True);proof=old.execute('SELECT source_family FROM tasks WHERE task_id=?',(tid,)).fetchone();assert proof and proof[0].startswith('Rfam')
  old.close();native_rows.append(sr);reuse.append(dict(reuse_kind=kind,existing_task_id=tid,new_task_serialized=False,native_source_file=f['source'],native_source_row_index=ro,census_occurrence_file=f['occurrences'],selected_occurrence=native_json(occ),existing_readonly_index=str(op.relative_to(R)),existing_index_key_result=proof,source_native_sequence_sha256=digest.hex()))
 np=O/'native-source-records.parquet';pq.write_table(pa.Table.from_pylist(native_rows),np,compression='zstd');assert pq.read_table(np).to_pylist()==native_rows;guard();receipt=dict(schema='mora3.ClosedRNAcentralNativeReaderFixture/v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='bounded reader development fixture, not global source/task proof or master admission',source_census=ref(cpath),frozen_census_index=c['native_identity_URS_index'],frozen_index_sha_evidence='completed census/root verification reference; not a new 18GB rehash for tiny fixture',closed_prefix_checkpoint=snapshot,closed_prefix_shards_inspected=read_prefix,new_task_cases=evidence,native_source_records=ref(np,rows=len(native_rows)),old_reuse_witnesses=reuse,native_U_case_available='native_U' in selected,source_T_U_case_ambiguity_preserved=True,source_conditions_have_no_native_target_length_hash_headers_URS_status=True,global_source_task_proof_complete=False,master_union_admission=False,current_authoritative_master='v22',actual_model_use_allowed=False,model_training=False,all_originals_and_writing_index_untouched=True,seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
 (O/'manifest.json').open('x').write(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(manifest=ref(O/'manifest.json'),cases=[dict(labels=x['labels'],length=x['full_axis_values']) for x in evidence],seconds=receipt['seconds'])))
def sha_bytes(x):return hashlib.sha256(x).hexdigest()
if __name__=='__main__':main()
