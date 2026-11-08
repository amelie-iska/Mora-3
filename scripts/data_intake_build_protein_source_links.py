"""Finite independent raw-row acceptance of qualified UniRef representative links."""
from pathlib import Path
import collections,csv,datetime,gzip,hashlib,json,os,sys,time
import pyarrow as pa
import pyarrow.parquet as pq
R=Path.cwd();sys.path.insert(0,str(R/'scripts'))
from data_intake_protein_source_links import representative_links
B=Path('outputs/data-intake/20261101-qualified-protein-source-links-v1');B.mkdir(exist_ok=False);(B/'links').mkdir();(B/'quarantine').mkdir();(B/'code').mkdir();start=time.perf_counter();CAP=2*2**30
for f in ['data_intake_build_protein_source_links.py','data_intake_protein_source_links.py']:(B/'code'/f).write_bytes((R/'scripts'/f).read_bytes())
mp=Path('outputs/data-sources/20261031-legacy-mint-v1/curated/uniref-native-qc-v1/manifest-all130verified-shards-v1.json');m=json.loads(mp.read_text())
def ref(p,**kw):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest(),**kw)
def check(e):
 got=ref(e['path']);assert got['bytes']==e['bytes'] and got['sha256']==e['sha256'];return Path(e['path'])
def live(stage,**kw):
 (B/'live-build.json').write_text(json.dumps(dict(pid=os.getpid(),stage=stage,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.perf_counter()-start,**kw),indent=2)+'\n');print(stage,kw,round(time.perf_counter()-start,2),flush=True)
def reserve():
 st=os.statvfs('.');assert st.f_bavail*st.f_frsize>=200*2**30;assert sum(p.stat().st_size for p in B.rglob('*') if p.is_file())<=CAP
try:
 baseline_ref=m['qualified_baseline'];bp=Path(baseline_ref['path']);assert ref(bp)['sha256']==baseline_ref['sha256']
 with gzip.open(bp,'rt') as f:baseline={r['Entry']:r for r in csv.DictReader(f,delimiter='\t')}
 assert len(baseline)==107778
 (B/'build-plan.json').open('x').write(json.dumps(dict(schema='mora3.QualifiedSourceIdentityPlan/v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_manifest=ref(mp),baseline=baseline_ref,new_download_bytes=0,derived_cap_bytes=CAP,minimum_free_bytes=200*2**30,policy='Only explicit representative accession AND exact full native sequence;current reviewed score>=4;source biological release unknown;no member/seed/cluster annotation transfer',new_corpus_tasks=0),indent=2)+'\n')
 links=[];quars=[];linkparts=[];qparts=[];seenrows=set();clusters=collections.defaultdict(set);accepted_acc=set();seqhash=set();counts=collections.Counter();source_refs=[]
 def flush():
  for rows,name,parts in [(links,'links',linkparts),(quars,'quarantine',qparts)]:
   if not rows:continue
   p=B/name/f'{len(parts):05d}.parquet';pq.write_table(pa.Table.from_pylist(rows),p,compression='zstd');assert pq.read_table(p).to_pylist()==rows;parts.append(ref(p,rows=len(rows)));rows.clear()
  reserve()
 for ix,s in enumerate(m['shards']):
  asset=s['source_asset'];p=check(asset);source_refs.append(ref(p,source_rows=s['available_native_cluster_rows']));wanted={}
  for e in s['typed_candidate_or_quarantine_payloads']:
   for c in pq.read_table(check(e)).to_pylist():
    assert c['source_row_index'] not in wanted;wanted[c['source_row_index']]=c
  offset=0;matched=0;columns=['id','updated','seed_id','rep_member_id','rep_accessions','sequence','sequence_length','rep_is_seed','rep_organism_tax_id','member_count']
  for batch in pq.ParquetFile(p).iter_batches(batch_size=8192,columns=columns):
   indices=[i for i in range(offset,offset+batch.num_rows) if i in wanted]
   if indices:
    selected=batch.take(pa.array([i-offset for i in indices],type=pa.int64())).to_pylist()
    for index,row in zip(indices,selected):
     c=wanted[index];matched+=1;assert c['source_record_id'] not in seenrows;seenrows.add(c['source_record_id']);assert row['id']==c['cluster_id'] and row['rep_accessions']==c['representative_accessions'] and row['seed_id']==c['seed_id'] and row['rep_member_id']==c['representative_member_id'];assert len(row['sequence'])==row['sequence_length']==c['native_sequence_length'];assert hashlib.sha256(row['sequence'].encode()).hexdigest()==c['sequence_sha256']
     accepted,rejected=representative_links(row,baseline);assert {x['qualified_accession'] for x in accepted}==set(c['qualified_exact_accessions']);assert {x['qualified_accession'] for x in rejected}==set(c['qualified_mismatch_accessions']);counts['source_rows_with_exact_links']+=bool(accepted);counts['source_rows_with_mismatch_links']+=bool(rejected);counts['accepted_identity_links']+=len(accepted);counts['quarantined_accession_links']+=len(rejected);source_ref=dict(native_payload=str(p),native_payload_sha256=asset['sha256'],source_row_index=index,source_record_id=c['source_record_id'],source_repo=m['source_repo'],source_revision=m['source_revision'],source_biological_release=None)
     for a in accepted:
      accepted_acc.add(a['qualified_accession']);seqhash.add(a['source_sequence_sha256']);clusters[a['source_cluster_id']].add(a['source_sequence_sha256']);a['groups_json']=json.dumps(a.pop('groups'));a['source_ref_json']=json.dumps(source_ref,sort_keys=True);a['source_native_roles_json']=json.dumps({k:v for k,v in row.items() if k!='sequence'},sort_keys=True);links.append(a)
     for q in rejected:q['source_ref_json']=json.dumps(source_ref,sort_keys=True);quars.append(q)
     if len(links)+len(quars)>=4096:flush()
   offset+=batch.num_rows
  assert matched==len(wanted) and offset==s['available_native_cluster_rows'];counts['source_candidate_or_quarantine_rows_verified']+=matched;live('independent exact representative native-row readback',source_shards=ix+1,counts=dict(counts));reserve()
 flush();assert counts['source_rows_with_exact_links']==67412 and counts['source_rows_with_mismatch_links']==136;conflicts={k:sorted(v) for k,v in clusters.items() if len(v)>1};(B/'candidate_cluster_ID_sequence_conflicts.json').open('x').write(json.dumps(conflicts,indent=2)+'\n')
 result=dict(schema='mora3.QualifiedProteinSourceIdentityLinks/v1',status='full_candidate_native_row_readback_and_current_baseline_qualification_verified',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_manifest=ref(mp),qualified_baseline=baseline_ref,source_raw_available_cluster_rows=60315044,source_worker_native_QC_pass_cluster_rows=60315044,source_candidate_or_quarantine_rows_independently_checked=counts['source_candidate_or_quarantine_rows_verified'],counts=dict(counts),qualified_current_accessions=len(accepted_acc),exact_sequence_hashes=len(seqhash),source_cluster_reference_IDs=len(clusters),cluster_IDs_with_conflicting_representative_sequence_hashes=conflicts,link_parts=linkparts,quarantine_parts=qparts,source_native_payloads=source_refs,all_source_payload_hashes_rechecked=True,all_candidate_native_ID_role_sequence_accession_fields_independently_compared=True,all_link_and_quarantine_Parquet_roundtrips=True,new_protein_sequences_admitted=0,new_corpus_tasks=0,new_independent_experiments=0,annotation_or_score_inheritance_allowed=False,seed_member_and_cluster_labels_not_transferred=True,source_biological_release_unknown=True,current_baseline_release='2026_03',identity_scope='source_explicit_representative_accession_AND_exact_full_sequence_to_existing_qualified_baseline_entry_only',group_scope='exact_accession_sequence_and_source_representative_reference;not_homologycluster_admission_or_member_merge',homology_heldout_unresolved=True,global_leakage_safe=False,model_reader_support_for_new_links=False,optimizer_steps=0,code_refs=[ref(p) for p in (B/'code').iterdir()],seconds=time.perf_counter()-start)
 (B/'manifest.json').open('x').write(json.dumps(result,indent=2)+'\n');live('complete qualified source identity supplement',counts=dict(counts),qualified_accessions=len(accepted_acc));print('COMPLETE',json.dumps(dict(counts)),flush=True)
except BaseException as exc:
 (B/'failure.json').open('x').write(json.dumps(dict(error=repr(exc),seconds=time.perf_counter()-start),indent=2)+'\n');raise
