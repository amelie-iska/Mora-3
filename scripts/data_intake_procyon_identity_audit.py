"""Bounded read-only native qualification/split audit; no tasks or model use."""
from pathlib import Path
import collections,csv,datetime,gzip,hashlib,json,os,resource,sys,time
import pyarrow as pa,pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_procyon_identity import qualify_whole_protein,domain_prefix_span_candidate
H=R/'outputs/data-sources/20261101-ProCyon-source-v1/handoff-ProCyon-open-native-components-v1.json';BASE=R/'outputs/data-intake/20261031-subagent/raw/uniprot2026_03/reviewed-annotation4-5-sequences.tsv.gz';O=R/'outputs/data-intake/20261101-ProCyon-native-corpus-v1/exact-protein-and-source-split-audit-v1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while x:=f.read(2**20):h.update(x)
 return h.hexdigest()
def ref(p,**kw):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p),**kw)
def checked(v):
 p=Path(v['path']);p=p if p.is_absolute() else R/p
 assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256'];return p
def main():
 start=time.monotonic();os.sched_setaffinity(0,{26,27});os.nice(10);pa.set_cpu_count(1);pa.set_io_thread_count(1);assert not O.exists();O.mkdir(parents=True);assert sha(H)=='0cc11e0d7be06d59c0ffbeea4f8d5a3eb38631258af4892508c513d488cc43ef';assert sha(BASE)=='e9bd53c005179cc68664f920f833273b241afb7196806ca512866869674d5133';h=json.loads(H.read_text());mp=checked(h['references']['native_QC']);m=json.loads(mp.read_text());inputs=dict(handoff=ref(H),source_QC_manifest=ref(mp),qualified_current_UniProt=ref(BASE));baseline={}
 def guard():
  rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;other=Path('/proc/3501176/statm');other_rss=int(other.read_text().split()[1])*os.sysconf('SC_PAGE_SIZE') if other.exists() else 0;st=os.statvfs(R)
  if time.monotonic()-start>120 or rss>512*2**20 or rss+other_rss+128*2**20>8*2**30 or st.f_bavail*st.f_frsize<64*2**30:raise RuntimeError('finite read-only audit hardware guard; retain partials')
 guard()
 with gzip.open(BASE,'rt',newline='') as f:
  for n,r in enumerate(csv.DictReader(f,delimiter='\t')):
   assert r['Entry'] not in baseline and r['Reviewed']=='reviewed' and r['Annotation'] in ['4.0','5.0'];baseline[r['Entry']]=dict(annotation_score=r['Annotation'],reviewed=r['Reviewed'],sequence_SHA256=hashlib.sha256(r['Sequence'].encode()).hexdigest(),native_row=n)
 assert len(baseline)==107778;source_proteins={};qualified={};native={};out=[];counts=collections.Counter();fa_refs=[]
 for f in sorted(m['FASTA_files'],key=lambda x:0 if '/protein/' in x['source_path'] else 1):
  kind='protein' if '/protein/' in f['source_path'] else 'domain';p=checked(f['payload']);checked(f['source']);fa_refs.append(f);rows=pq.read_table(p).to_pylist();assert len(rows)==f['native_records']
  for n,r in enumerate(rows):
   assert r['native_ordinal']==n and len(r['native_sequence'])==r['sequence_length'] and hashlib.sha256(r['native_sequence'].encode()).hexdigest()==r['sequence_SHA256'];assert (kind,r['native_ID']) not in native;native[(kind,r['native_ID'])]=r['sequence_SHA256'];base=dict(source_namespace=kind,native_ID=r['native_ID'],native_ordinal=n,native_sequence_SHA256=r['sequence_SHA256'],native_length=r['sequence_length'],native_header=r['native_header'],source_payload_path=f['payload']['path'],source_payload_SHA256=f['payload']['sha256'],source_original_FASTA_SHA256=f['source']['sha256'],task_admitted=False,model_use=False)
   if kind=='protein':
    q=qualify_whole_protein(r['native_ID'],r['native_sequence'],baseline);qualified[r['native_ID']]=q;source_proteins[r['native_ID']]=r['native_sequence'];base['qualification_json']=json.dumps(q,sort_keys=True);base['domain_parent_candidate_json']=None;counts['protein:'+q['status']]+=1
   else:
    q=domain_prefix_span_candidate(r['native_ID'],r['native_sequence'],source_proteins,qualified);base['qualification_json']=None;base['domain_parent_candidate_json']=json.dumps(q,sort_keys=True);counts['domain:native_records']+=1;counts['domain:prefix_candidate_source_protein_present']+=q['prefix_candidate_exists_in_source_whole_protein_FASTA'];counts['domain:prefix_candidate_fullprotein_current_qualified']+=q['prefix_candidate_whole_protein_current_qualified'];counts['domain:unique_literal_substring_candidate']+=q['unique_exact_substring_candidate'];counts['domain:multiple_literal_substring_spans']+=len(q['derived_literal_substring_spans_1based_closed'])>1;counts['domain:no_literal_substring_candidate']+=len(q['derived_literal_substring_spans_1based_closed'])==0
   out.append(base)
  guard();print('PROCYON_NATIVE_FASTA_QUALIFICATION',kind,len(rows),flush=True)
 p=O/'native-entity-current-qualification-and-domain-prefix-candidates.parquet';pq.write_table(pa.Table.from_pylist(out),p,compression='zstd');assert pq.read_table(p).to_pylist()==out;entity_ref=ref(p,rows=len(out));del out
 relations=[];relation_refs=[]
 for f in m['CSV_files']:
  if '_relations.unified.csv' not in f['source_path']:continue
  guard();p=checked(f['payload']);checked(f['source']);splits=collections.defaultdict(lambda:dict(sequence_ID=set(),sequence_SHA=set(),text_ID=set(),exact_relation=set()));local=collections.Counter();raw_rel=collections.Counter();source_qualified=collections.Counter();row_count=0
  for batch in pq.ParquetFile(p).iter_batches(batch_size=4096):
   for r in batch.to_pylist():
    raw=json.loads(r['native_fields_json']);kind=raw['seq_type'];seqid=raw['seq_id'];text=raw['text_id'];split=raw['split'];assert split==r['native_split'] and r['native_row_index']==row_count;row_count+=1;local[split]+=1;raw_rel[raw['relation']]+=1;key=(kind,seqid);s=splits[split];s['sequence_ID'].add(seqid);s['text_ID'].add(text);s['exact_relation'].add(hashlib.sha256(json.dumps([kind,seqid,raw['text_type'],text,raw['relation']],ensure_ascii=False,separators=(',',':')).encode()).digest())
    if key in native:s['sequence_SHA'].add(native[key])
    else:source_qualified['native_sequence_ID_absent_from_source_FASTA']+=1
    if kind=='protein':source_qualified[qualified.get(seqid,{}).get('status','native_protein_ID_not_source_mapped')]+=1
    else:source_qualified['domain_parent_coordinate_qualification_not_admitted']+=1
   guard()
  assert row_count==f['rows'];train=splits.get('CL_train',{k:set() for k in ['sequence_ID','sequence_SHA','text_ID','exact_relation']});overlaps={s:{k:len(v[k]&train[k]) for k in train} for s,v in splits.items() if s!='CL_train'};relations.append(dict(source_path=f['source_path'],rows=row_count,original_source=f['source'],native_payload=f['payload'],exact_native_split_counts=dict(local),source_native_relation_literal_counts=dict(raw_rel),native_source_protein_qualification_row_counts=dict(source_qualified),source_split_distinct_entity_counts={s:{k:len(v[k]) for k in v} for s,v in splits.items()},source_nontrain_overlap_with_source_CL_train=overlaps,source_CL_train_is_not_actual_Mora_model_training=True,original_relation_strings_authoritative_not_global_indexed_code_zero=True,source_historical_predicate_not_assumed_current_experimental_truth=True));relation_refs.append(f['payload']);print('PROCYON_RELATION_SPLIT_AUDIT',f['source_path'],row_count,flush=True)
 assert sum(x['rows'] for x in relations)==433007;guard();assert ref(BASE)==inputs['qualified_current_UniProt'] and ref(H)==inputs['handoff'] and ref(mp)==inputs['source_QC_manifest'];receipt=dict(schema='mora3.ProCyonCurrentWholeSequenceAndNativeProtocolAudit/v1',status='complete_selected_open_native_entity_qualification_and_original_relation_split_audit;no_tasks_or_master_admission',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),inputs=inputs,source_dataset_revision=h['dataset_revision'],current_qualification_release='2026_03;not_historical_ProCyon_release_equivalence',native_entity_count=59660,native_proteins=18174,native_domains=41486,current_reviewed_annotation4plus_baseline_entries=107778,counts=dict(counts),entity_qualification=entity_ref,all_source_native_sequence_payloads=fa_refs,original_relation_tables=relations,original_relation_view_rows=433007,all_native_source_relation_type_and_split_strings_preserved=True,domain_prefix_or_derived_substring_is_not_native_parent_coordinate_or_label_inheritance=True,exact_current_whole_protein_match_not_automatic_relation_or_text_admission=True,source_node_overlap_not_actual_model_optimizer_use=True,global_homology_and_phenotype_parent_leakage_safe=False,opaque_pickles_not_read_or_unpickled=True,new_task_views=0,new_global_admitted_records=0,model_use=False,full_dataset_ready=False,substantive_training=False,source_immutability=True,seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);(O/'manifest.json').open('x').write(json.dumps(receipt,indent=2)+'\n');print('COMPLETE_PROCYON_NATIVE_IDENTITY_PROTOCOL_AUDIT',json.dumps(dict(counts=dict(counts),seconds=receipt['seconds'],manifest=ref(O/'manifest.json'))),flush=True)
if __name__=='__main__':main()
