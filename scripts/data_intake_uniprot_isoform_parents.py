"""Resolve missing isoform payloads by exact qualified-parent query; one connection."""
from pathlib import Path
import sys,json,gzip,csv,hashlib,collections,urllib.request,urllib.parse,datetime,time,os,shutil
import pyarrow as pa,pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'));from data_intake_uniprot_isoforms import fasta_records
B=R/'outputs/data-intake/20261031-uniprot-full-annotations-v1';OUT=B/'isoform-exact-parents-v1'

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.open('x').write(json.dumps(x,indent=2)+'\n')
def rec(p,**kw):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p),**kw)

def run():
 OUT.mkdir();(OUT/'raw').mkdir();(OUT/'native').mkdir();(OUT/'completed-batches').mkdir();(OUT/'code').mkdir();(OUT/'code'/Path(__file__).name).write_bytes(Path(__file__).read_bytes());start=time.monotonic()
 with gzip.open(R/'outputs/data-intake/20261031-subagent/raw/uniprot2026_03/reviewed-annotation4-5-sequences.tsv.gz','rt') as f:baseline={r['Entry']:r['Sequence'] for r in csv.DictReader(f,delimiter='\t')}
 old=json.loads((B/'isoform-sequences-v2/manifest.json').read_text());old_isoseq={}
 for f in old['native_shards']:
  p=R/f['path'];assert sha(p)==f['sha256']
  for r in pq.read_table(p,columns=['source_accession','sequence','is_canonical_record']).to_pylist():
   if not r['is_canonical_record']:old_isoseq[r['source_accession']]=r['sequence']
 fm=json.loads((B/'physical-form-supplement-v2/manifest.json').read_text());expected=collections.defaultdict(set);allforms=set()
 for f in fm['files']:
  if f['kind']=='forms':
   p=R/f['path'];assert sha(p)==f['sha256']
   for r in pq.read_table(p).to_pylist():
    x=json.loads(r['record_json'])
    if x['form_kind']=='alternative_product':
     allforms.add(x['form_id']);fid=x['form_id'];parent=fid.split('-',1)[0]
     if x['sequence_status']=='Described' and parent==x['accession'] and parent in baseline:expected[parent].add(fid)
 parents=sorted(p for p,ids in expected.items() if ids-set(old_isoseq));batches=[parents[i:i+100] for i in range(0,len(parents),100)];required=set().union(*(expected[p] for p in parents));plan=dict(schema='mora3.UniProtExactParentIsoformPlan/v1',registered_utc=now(),pid=os.getpid(),one_connection=True,release='2026_03',qualified_parent_entries=len(parents),batches=batches,expected_described_ids=sorted(required),missing_described_ids_before=sorted(required-set(old_isoseq)),prior_complete_stream=rec(B/'isoform-sequences-v2/manifest.json'),query_filter_policy='exactaccessionparentsonly;entryreviewed_score4/5verifiedfromfullKBbaseline;scorefilternotappliedtoforms',source_probe=rec(B/'isoform-exact-parent-probes-v1/receipt.json'),parent_score_is_not_perform_score=True,raw_cap_bytes=512*2**20,derived_cap_bytes=2*2**30,lease='sameoneRfamreleasedslot;noextra_connection')
 write(OUT/'acquisition-plan.json',plan);seen_seqs=dict(old_isoseq);manifests=[];rows_total=0;raw_total=0
 def live(stage,**kw):(OUT/'live-acquisition.json').write_text(json.dumps(dict(updated_utc=now(),pid=os.getpid(),stage=stage,connection_count=1,completed_batches=len(manifests),expected_batches=len(batches),raw_bytes=raw_total,prior_source_isoform_records=len(old_isoseq),consolidated_unique_isoforms=len(seen_seqs),elapsed_seconds=time.monotonic()-start,**kw),indent=2)+'\n')
 def fetch(url,p):
  nonlocal raw_total
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mora3-public-source-intake/1','Accept':'text/plain'}),timeout=90) as response,p.open('xb') as f:
   headers=dict(response.headers);assert headers.get('X-UniProt-Release')=='2026_03';received=0
   while True:
    block=response.read(1024*1024)
    if not block:break
    received+=len(block);raw_total+=len(block);assert raw_total<plan['raw_cap_bytes'] and received<16*2**20 and shutil.disk_usage(R).free>200*2**30;f.write(block)
  return rec(p,url=url,headers=headers,retrieved_utc=now(),source_release='2026_03',complete_HTTP_body=True)
 for i,batch in enumerate(batches):
  live('finite exact-parent FASTA acquisition',active_batch=i);query=' OR '.join('(accession:'+p+')' for p in batch);url='https://rest.uniprot.org/uniprotkb/stream?'+urllib.parse.urlencode(dict(query=query,format='fasta',includeIsoform='true',compressed='true'));receipt=None
  for attempt in range(1,4):
   p=OUT/'raw'/f'batch-{i:04d}.attempt-{attempt}.fasta.response'
   try:receipt=fetch(url,p);break
   except Exception as exc:
    write(OUT/'raw'/f'batch-{i:04d}.attempt-{attempt}.failure.json',dict(error=type(exc).__name__+':'+str(exc),path=str(p.relative_to(R)),bytes=p.stat().st_size if p.exists() else 0,url=url))
    if attempt==3:live('stopped;finite retries exhausted',active_batch=i);raise
    time.sleep(2**attempt)
  assert receipt is not None;write(OUT/'raw'/f'batch-{i:04d}.receipt.json',receipt);rows=[];seen_canonical=set();source_ids=set();actual_isoforms=set()
  with p.open('rb') as f:magic=f.read(2)
  handle=gzip.open(p,'rt') if magic==b'\x1f\x8b' else p.open()
  with handle as f:
   for row_index,(header,seq) in enumerate(fasta_records(f)):
    fields=header.split('|');assert fields[0]=='sp' and len(fields)>=3;sid=fields[1];parent=sid.split('-',1)[0];assert parent in batch and sid not in source_ids and seq and not(set(seq)-set('ACDEFGHIKLMNPQRSTVWYBXZJUO'));source_ids.add(sid);canonical=sid==parent
    if canonical:assert seq==baseline[parent];seen_canonical.add(parent)
    else:
     actual_isoforms.add(sid)
     if sid in seen_seqs:assert seen_seqs[sid]==seq,'same-release exact form sequence conflict'
     else:seen_seqs[sid]=seq
    rows.append(dict(source_accession=sid,parent_entry_accession=parent,is_canonical_record=canonical,source_header=header,sequence=seq,sequence_length=len(seq),sequence_sha256=hashlib.sha256(seq.encode()).hexdigest(),source_record_index=row_index,source_file=receipt['path'],source_sha256=receipt['sha256'],fullKB_form_ID_known=sid in allforms if not canonical else True,entry_qualification='parententryreviewedannotation4/5exactbaseline;no_performscoreclaim',canonical_labels_inherited=False,task_admitted=False))
  assert seen_canonical==set(batch);q=OUT/'native'/f'batch-{i:04d}.parquet';pq.write_table(pa.Table.from_pylist(rows),q,compression='zstd');assert pq.read_table(q).to_pylist()==rows;rows_total+=len(rows);need=set().union(*(expected[parent] for parent in batch));m=dict(schema='mora3.UniProtExactParentIsoformBatch/v1',completed_utc=now(),parent_accessions=batch,source_release='2026_03',source_receipt=receipt,native_sequences=rec(q,rows=len(rows)),canonical_sequences_all_exact=True,source_records=len(rows),actual_isoform_ids=sorted(actual_isoforms),expected_described_ids=sorted(need),expected_described_missing=sorted(need-actual_isoforms),source_new_unknown_descriptor_ids=sorted(actual_isoforms-allforms),all_source_sequences_fullaxis=True,all_Parquet_roundtrips=True,source_license='CC-BY-4.0',task_admitted=0,canonical_annotation_inheritance=False);mp=OUT/'completed-batches'/f'manifest-{i:04d}.json';write(mp,m);manifests.append(rec(mp));live('batch source/native QC verified',active_batch=i);print('ISOFORM_PARENT_BATCH',i+1,'parents',len(batch),'sourceforms',len(actual_isoforms),'total_uniqueisoforms',len(seen_seqs),'missing',len(need-actual_isoforms),'seconds',round(time.monotonic()-start,2),flush=True)
 final=dict(schema='mora3.UniProtExactParentIsoformSource/v1',status='finite_source_acquisition_and_native_QC_complete',source_release='2026_03',source_license='CC-BY-4.0',qualified_parent_entries=len(parents),complete_batches=manifests,source_records_thisexport_with_prior_overlaps=rows_total,initial_isoform_records=len(old_isoseq),consolidated_unique_isoforms=len(seen_seqs),new_unique_isoforms=len(seen_seqs)-len(old_isoseq),expected_described_ids=len(required),expected_described_missing=sorted(required-set(seen_seqs)),new_independent_experiments=0,task_admitted=0,canonical_annotations_inherited=False,raw_bytes=raw_total,all_native_fullaxis_roundtrips=True,all_parent_canonical_sequences_exact_baseline=True,entry_annotation_score_not_invented_for_isoforms=True,prior_stream_manifest=plan['prior_complete_stream'],plan=rec(OUT/'acquisition-plan.json'),isoform_feature_coordinate_supervision='pendingexactperfeaturemapping;noinheritance',seconds=time.monotonic()-start);write(OUT/'manifest.json',final);live('finite complete native source',manifest=str(OUT/'manifest.json'));print('COMPLETE',final['consolidated_unique_isoforms'],'missing',len(final['expected_described_missing']),flush=True)
if __name__=='__main__':run()
