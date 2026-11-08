"""Storage experiment on the SAME admitted pilot; not full-source indexing or replacement."""
from pathlib import Path
import sys,json,sqlite3,hashlib,time,datetime,shutil
import pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_corpus import canonical
B=R/'outputs/data-intake/20261101-RNAcentral-corpus-pilot-v1';O=R/'outputs/data-intake/20261101-RNAcentral-compact-index-probe-v1';CAP=256*2**20;O.mkdir();start=time.perf_counter()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as stream:
  while chunk:=stream.read(4*2**20):h.update(chunk)
 return h.hexdigest()
def ref(p,**kw):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p),**kw)
def check(r):
 p=R/r['path'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'];return p
m=json.loads((B/'manifest.json').read_text());source=check(m['new_identity_and_group_index']);old=sqlite3.connect('file:'+str(source)+'?mode=ro',uri=True);expected={r[0]:(r[1],bool(r[2])) for r in old.execute('SELECT native_sha,task_id,reused_prior FROM sequences')};assert len(expected)<=500000
p=O/'compact-native-identity.sqlite3';db=sqlite3.connect(p);db.executescript('PRAGMA journal_mode=OFF;CREATE TABLE sequences(native_sha BLOB PRIMARY KEY CHECK(length(native_sha)=32),first_source_file INTEGER NOT NULL,first_native_row INTEGER NOT NULL,reused_prior INTEGER NOT NULL) WITHOUT ROWID;');files={};source_refs={};rows=0
for ledger in m['occurrences']:
 for row in pq.read_table(check(ledger)).to_pylist():
  sr=json.loads(row['source_ref_json']);file=sr['native_payload'];digest=row['native_sequence_sha256'];blob=bytes.fromhex(digest);assert blob.hex()==digest;expected_tid,reused=expected[digest];assert row['task_id']==expected_tid and bool(row['task_already_in_Rfam_corpus'])==reused;domain='RNA_family_reference_native_symbols' if reused else 'RNA_reference_native_symbols';assert hashlib.sha256(canonical([domain,None,digest])).hexdigest()==expected_tid
  if file not in files:files[file]=len(files);source_refs[file]=dict(path=file,sha256=sr['native_payload_sha256'])
  inserted=db.execute('INSERT OR IGNORE INTO sequences VALUES (?,?,?,?)',(blob,files[file],sr['native_row_index'],int(reused))).rowcount;assert inserted==int(row['first_occurrence_in_pilot']);rows+=1
 db.commit();assert p.stat().st_size<CAP and shutil.disk_usage(R).free>200*2**30
index_count=0
for blob,fileidx,nativerow,reused in db.execute('SELECT native_sha,first_source_file,first_native_row,reused_prior FROM sequences'):
 digest=bytes(blob).hex();assert len(blob)==32 and digest in expected and bool(reused)==expected[digest][1] and 0<=fileidx<len(files) and nativerow>=0;index_count+=1
assert index_count==len(expected) and rows==m['counts']['quality_pass_occurrences'];db.close();old.close();(O/'source-file-dictionary.json').open('x').write(json.dumps(dict(files=[dict(ordinal=files[path],**source_refs[path]) for path in files]),indent=2)+'\n');(O/'code.py').write_bytes(Path(__file__).read_bytes());actual=sum(f.stat().st_size for f in O.rglob('*') if f.is_file());assert actual<CAP;whole_estimate=int(p.stat().st_size/index_count*61006261)
result=dict(schema='mora3.RNAcentralCompactIdentityStorageProbe/v1',status='all_same_pilot_native_identity_and_taskID_conservation_verified;notreplacement_or_fullsourcebuild',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_pilot=ref(B/'manifest.json'),original_index=ref(source),original_index_open_mode='ro',original_index_SHA_after=sha(source),source_quality_pass_occurrences=rows,distinct_native_identity_rows=index_count,compact_index=ref(p,rows=index_count),source_file_dictionary=ref(O/'source-file-dictionary.json'),same_target_sha_hex_binary_roundtrip=True,same_domain_exact_taskID_recipe_all_rows=True,all_complete_source_occurrence_and_group_ledgers_remain_authoritative=True,first_source_reference_only_not_all_URS_or_status_multiplicity=True,old_group_SQLite_and_all_old_outputs_preserved=True,new_global_URS_or_homology_proof=False,new_corpus_master_or_task_shards=0,new_independent_experiments=0,full61M_index_or_materialization_launched=False,pilot_resource_cap_bytes=CAP,actual_probe_generated_bytes=actual,compact_index_full61006261_candidate_size_extrapolation_bytes=whole_estimate,extrapolation_scope='indexonly;notfulltasks/occurrences/effectivegroups;quantilepilotandSQLitebalancingnotfullpopulationguarantee',remaining_whole_materialization_gate='wholebuildrequiresseparatereservationaftertask/group/occurrencecosts;currentreallocationproposaldoesnotauthorizeit',seconds=time.perf_counter()-start)
assert result['original_index']['sha256']==result['original_index_SHA_after'];(O/'manifest.json').open('x').write(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['source_quality_pass_occurrences','distinct_native_identity_rows','compact_index','actual_probe_generated_bytes','compact_index_full61006261_candidate_size_extrapolation_bytes','seconds']},indent=2))
