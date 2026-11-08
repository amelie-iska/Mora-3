"""Read-only source-phenotype target exposure audit, not model-use evidence."""
from pathlib import Path
import argparse,collections,datetime,hashlib,json,os,resource,sqlite3,sys,time
import pyarrow as pa,pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_procyon_GOA_exposure import canonical_accession,compatible_positive_role
B=R/'outputs/data-intake/20261101-ProCyon-native-corpus-v1';P=B/'GOA-target-exposure-audit-plan-v1.json'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while c:=f.read(2**20):h.update(c)
 return h.hexdigest()
def ref(p,**kw):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p),**kw)
def checked(v):
 p=Path(v['path']);p=p if p.is_absolute() else R/p
 if p.stat().st_size!=v['bytes'] or sha(p)!=v['sha256']:raise ValueError('immutable exposure input changed: '+str(p))
 return p
def approved(a,plan_ref,script_ref):
 if a.get('approved') is not True or a.get('scope')!='ProCyon-qualified-canonical-GOA-corpus-target-exposure-audit-only':raise ValueError('exact exposure audit scope approval required')
 if a.get('plan_sha256')!=plan_ref['sha256'] or a.get('script_sha256')!=script_ref['sha256']:raise ValueError('exact exposure plan/script binding required')
 if any(a.get(k) is not False for k in ['task_admission','master_publication','model_use','model_training']):raise ValueError('exposure evidence authorizes no admission/publication/model operation')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--approval',required=True);args=ap.parse_args();start=time.monotonic();started=time.time();os.sched_setaffinity(0,{26,27});os.nice(10);pa.set_cpu_count(1);pa.set_io_thread_count(1);plan=json.loads(P.read_text());approval=R/args.approval;approved(json.loads(approval.read_text()),ref(P),ref(Path(__file__).resolve()))
 for v in list(plan['inputs'].values())+plan['dependencies']:checked(v)
 O=R/plan['output_namespace'];assert not O.exists();O.mkdir();stage='initial';count=collections.Counter();samples=[]
 def guard():
  peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;rn=Path('/proc/3501176/statm');rna=int(rn.read_text().split()[1])*os.sysconf('SC_PAGE_SIZE') if rn.exists() else 0;st=os.statvfs(R);s=dict(stage=stage,elapsed_seconds=time.monotonic()-start,deadline_utc=datetime.datetime.fromtimestamp(started+900,datetime.timezone.utc).isoformat(),own_peak_RSS_bytes=peak,own_current_RSS_bytes=int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE'),known_RNA_RSS_bytes=rna,expected_RNA_owner_present=rn.exists(),aggregate_with_RNA_and_wrapper_reserve_bytes=peak+rna+128*2**20,free_hardware_bytes=st.f_bavail*st.f_frsize);viol=[]
  if s['elapsed_seconds']>900:viol.append('elapsed_seconds')
  if peak>512*2**20:viol.append('own_peak_RSS_bytes')
  if s['aggregate_with_RNA_and_wrapper_reserve_bytes']>8*2**30:viol.append('aggregate_RSS_bytes')
  if s['free_hardware_bytes']<64*2**30:viol.append('free_hardware_bytes')
  if not rn.exists():viol.append('expected_RNA_resource_owner_missing_recoordination_required')
  s['threshold_failures']=viol;samples.append(s)
  if viol:(O/'guard-stop-diagnostics.json').open('x').write(json.dumps(s,indent=2)+'\n');raise RuntimeError('exposure audit guard: '+','.join(viol))
 guard();master=json.loads(checked(plan['inputs']['master']).read_text());goa=json.loads(checked(plan['inputs']['GOA_component']).read_text());proc=json.loads(checked(plan['inputs']['ProCyon_identity_protocol_audit']).read_text());source=json.loads(checked(proc['inputs']['source_QC_manifest']).read_text());qualified=set()
 for batch in pq.ParquetFile(checked(proc['entity_qualification'])).iter_batches(columns=['source_namespace','native_ID','qualification_json'],batch_size=2048):
  for row in batch.to_pylist():
   if row['source_namespace']=='protein' and json.loads(row['qualification_json'])['exact_accession_and_full_sequence_qualified']:qualified.add(row['native_ID'])
 assert len(qualified)==14734;db=sqlite3.connect(O/'source-phenotype-target-exposure.sqlite');db.executescript('PRAGMA cache_size=-65536;PRAGMA temp_store=FILE;CREATE TABLE procyon_rows(source_row INTEGER PRIMARY KEY,accession TEXT,GO_ID TEXT,source_split TEXT,native_relation TEXT,exact_current_protein_qualified INTEGER,broad_target_annotations INTEGER DEFAULT 0,positive_role_compatible_annotations INTEGER DEFAULT 0,NOT_target_annotations INTEGER DEFAULT 0,extension_target_annotations INTEGER DEFAULT 0);CREATE INDEX source_pair ON procyon_rows(accession,GO_ID);CREATE TABLE GOA_target_hits(annotation_id TEXT PRIMARY KEY,subject TEXT,GO_ID TEXT,negated INTEGER,relations_json TEXT,aspect TEXT,evidence_class TEXT,GO_evidence TEXT,ECO_evidence TEXT,has_extensions INTEGER,task_id TEXT,task_shard TEXT,task_shard_row INTEGER,raw_target_annotation_index INTEGER,typed_target_annotation_index INTEGER);')
 original=[f for f in source['CSV_files'] if f['source_path'].endswith('/protein_go_relations.unified.csv')];assert len(original)==1;f=original[0];p=checked(f['payload']);checked(f['source']);stage='ProCyon_original_GO_relation_rows'
 for batch in pq.ParquetFile(p).iter_batches(batch_size=4096):
  for row in batch.to_pylist():
   d=json.loads(row['native_fields_json']);assert d['seq_type']=='protein' and d['split']==row['native_split'];db.execute('INSERT INTO procyon_rows(source_row,accession,GO_ID,source_split,native_relation,exact_current_protein_qualified) VALUES(?,?,?,?,?,?)',(row['native_row_index'],d['seq_id'],d['text_id'],d['split'],d['relation'],int(d['seq_id'] in qualified)));count['source_ProCyon_GO_relation_rows']+=1
  guard()
 db.commit();assert count['source_ProCyon_GO_relation_rows']==236333;assert db.execute('SELECT count(*) FROM procyon_rows WHERE exact_current_protein_qualified=1').fetchone()[0]==221516;known={sh['path']:sh for sh in master['shards']};scanrefs=[]
 for n,f in enumerate(goa['task_shards']):
  stage='GOA_admitted_target_packet_shard:'+str(n);guard();assert f['path'] in known
  for key in ['bytes','sha256','rows']:assert known[f['path']][key]==f[key]
  p=checked(f);assert pq.ParquetFile(p).metadata.num_rows==f['rows'];scanrefs.append(f);shard_local=0
  for batch in pq.ParquetFile(p).iter_batches(batch_size=16):
   for row in batch.to_pylist():
    local=shard_local;shard_local+=1
    count['all_admitted_GOA_packets_seen']+=1;acc=canonical_accession(row['record_id'])
    if acc is None:count['form_or_noncanonical_packets_not_flattened']+=1;continue
    if acc not in qualified:continue
    assert row['split']=='train' and row['admitted_for_task'] is True and row['source_family']=='UniProt-GOA-assertion-packets';raw=json.loads(row['native_payload']);assert hashlib.sha256(row['native_payload']).hexdigest()==row['native_payload_sha256'];typed={a['source_annotation_id']:(i,a) for i,a in enumerate(raw['targets']['typed_GO_annotations'])};annotations=raw['targets']['source_GO_annotations'];assert len(typed)==len(annotations);count['qualified_canonical_GOA_packets_examined']+=1
    for ai,a in enumerate(annotations):
     assert a['geneProductId']==row['record_id'];ti,t=typed[a['id']];assert t['go_id']==a['goId'] and t['subject']==row['record_id'];matches=db.execute('SELECT source_row,native_relation FROM procyon_rows WHERE accession=? AND GO_ID=? AND exact_current_protein_qualified=1',(acc,a['goId'])).fetchall();count['qualified_canonical_target_annotations_examined']+=1
     if not matches:continue
     neg=t['negated'];ext=bool(t.get('extension_or_of_and'));db.execute('INSERT INTO GOA_target_hits VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(a['id'],row['record_id'],a['goId'],int(neg),json.dumps(t.get('relations',[])),t.get('aspect'),t.get('evidence_class'),t.get('GO_evidence'),t.get('ECO_evidence'),int(ext),row['task_id'],f['path'],local,ai,ti));count['current_GOA_target_annotations_with_source_pair_potential_exposure']+=1
     for source_row,literal in matches:db.execute('UPDATE procyon_rows SET broad_target_annotations=broad_target_annotations+1,positive_role_compatible_annotations=positive_role_compatible_annotations+?,NOT_target_annotations=NOT_target_annotations+?,extension_target_annotations=extension_target_annotations+? WHERE source_row=?',(int(compatible_positive_role(literal,t)),int(neg),int(ext),source_row))
   guard()
  assert shard_local==f['rows'];db.commit()
  if n%100==0:print('PROCYON_GOA_TARGET_EXPOSURE',n+1,'packets',count['all_admitted_GOA_packets_seen'],'hits',count['current_GOA_target_annotations_with_source_pair_potential_exposure'],flush=True)
 assert count['all_admitted_GOA_packets_seen']==goa['serialized_task_packets'];summary={}
 for split in [r[0] for r in db.execute('SELECT DISTINCT source_split FROM procyon_rows')]:
  row=db.execute('SELECT count(*),sum(exact_current_protein_qualified),sum(broad_target_annotations>0),sum(positive_role_compatible_annotations>0),sum(NOT_target_annotations>0),sum(extension_target_annotations>0) FROM procyon_rows WHERE source_split=?',(split,)).fetchone();summary[split]=dict(zip(['source_relation_rows','exact_current_whole_protein_qualified_rows','same_canonical_accession_and_GO_target_potential_exposure_rows','positive_role_compatible_potential_exposure_rows','with_NOT_target_overlap_rows','with_context_extension_target_overlap_rows'],row))
 count['distinct_GO_target_annotation_hits']=db.execute('SELECT count(*) FROM GOA_target_hits').fetchone()[0];assert count['distinct_GO_target_annotation_hits']==count['current_GOA_target_annotations_with_source_pair_potential_exposure'];db.close()
 for v in list(plan['inputs'].values())+plan['dependencies']:checked(v)
 stage='final';guard();receipt=dict(schema='mora3.ProCyonQualifiedCanonicalGOACorpusTargetExposureAudit/v1',status='complete_pinned_master_target_membership_audit;no_training_or_admission_decision',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan=ref(P),approval=ref(approval),inputs=plan['inputs'],counts=dict(count),per_original_source_split=summary,source_original_GO_table=original[0],admitted_GOA_task_shards_scanned=scanrefs,exposure_index=ref(O/'source-phenotype-target-exposure.sqlite'),comparison_scope='exact current-qualified whole canonical source accession AND literal GO_ID only; optional role/aspect compatibility is potential target exposure, not experiment or full context equivalence',canonical_subject_only_no_isoform_PRO_or_sequence_alias_label_transfer=True,NOT_evidence_withfrom_extensions_remain_native_source_roles=True,source_current_snapshot_different_from_historical_ProCyon=True,all_original_CLeval_roles_preserved=True,old_tasks_or_model_weights_modified=False,new_tasks_admitted=0,master_publication=False,actual_model_optimizer_use=None,new_negative_or_biological_assertion_created=False,full_dataset_ready=False,substantive_training=False,resource_samples=samples,seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);(O/'manifest.json').open('x').write(json.dumps(receipt,indent=2)+'\n');print('COMPLETE_PROCYON_GOA_TARGET_EXPOSURE',json.dumps(dict(counts=dict(count),summary=summary,seconds=receipt['seconds'])),flush=True)
if __name__=='__main__':main()
