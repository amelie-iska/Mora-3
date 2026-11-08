"""One-connection, finite, lossless official includeIsoform source supplement."""
from pathlib import Path
import urllib.request,urllib.parse,json,time,datetime,hashlib,gzip,csv,subprocess,os,collections,shutil
import pyarrow as pa,pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');B=R/'outputs/data-intake/20261031-uniprot-full-annotations-v1';OUT=B/'isoform-sequences-v2'

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.open('x').write(json.dumps(d,indent=2)+'\n')
def fasta_records(handle):
 header=None;sequence=[]
 for raw in handle:
  line=raw.strip()
  if not line:continue
  if line.startswith('>'):
   if header is not None:yield header,''.join(sequence)
   header=line[1:];sequence=[]
  else:
   if header is None:raise ValueError('sequence before FASTA header')
   sequence.append(line)
 if header is not None:yield header,''.join(sequence)

def run():
 OUT.mkdir();write(OUT/'connection-lease.json',dict(connection_count=1,release='completed Rfam five-asset job;sourceworker20261031T234807lease',owner='dataset_construction',started_utc=now(),pid=os.getpid(),forbid_range_resume=True,raw_budget_bytes=1*2**30,derived_budget_bytes=2*2**30))
 def live(stage,**kw):(OUT/'live-acquisition.json').write_text(json.dumps(dict(updated_utc=now(),pid=os.getpid(),stage=stage,source_connections=1,**kw),indent=2)+'\n')
 def fetch(url,name,cap):
  p=OUT/name;started=time.monotonic();h=hashlib.sha256();received=0;live('streaming complete source',url=url,path=str(p),bytes=0)
  try:
   with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mora3-public-source-intake/1','Accept':'text/plain'}),timeout=90) as response,p.open('xb') as f:
    headers=dict(response.headers);assert headers.get('X-UniProt-Release',headers.get('x-uniprot-release'))=='2026_03','source release drift';last=time.monotonic()
    while True:
     block=response.read(1024*1024)
     if not block:break
     received+=len(block);assert received<=cap,'bounded raw source cap exceeded';assert shutil.disk_usage(R).free>200*2**30;f.write(block);h.update(block)
     if time.monotonic()-last>2:live('streaming complete source',url=url,path=str(p),bytes=received,elapsed_seconds=time.monotonic()-started);last=time.monotonic()
   expected=next((int(v) for k,v in headers.items() if k.lower()=='content-length'),None)
   if expected is not None:assert received==expected,'incomplete response bytes'
   rec=dict(path=str(p.relative_to(R)),url=url,retrieved_utc=now(),bytes=received,sha256=h.hexdigest(),headers=headers,complete_HTTP_body=True,seconds=time.monotonic()-started);write(OUT/(name+'.receipt.json'),rec);return p,rec
  except Exception as exc:
   write(OUT/(name+'.failure.json'),dict(url=url,bytes=received,retained_path=str(p.relative_to(R)),error=type(exc).__name__+':'+str(exc),failed_utc=now()));raise
 probe,probe_receipt=fetch('https://rest.uniprot.org/uniprotkb/stream?query=accession%3AP05067&format=fasta&includeIsoform=true','P05067.includeIsoform.fasta',2**20)
 with probe.open() as f:pr=list(fasta_records(f))
 assert any(h.split('|')[1]=='P05067' for h,s in pr) and len(pr)>1 and all(s for h,s in pr);write(OUT/'probe-verification.json',dict(records=len(pr),ids=[h.split('|')[1] for h,s in pr],source_release='2026_03',complete_probe=True,receipt=probe_receipt))
 q=json.loads((B/'acquisition-manifest.json').read_text())['query'];url='https://rest.uniprot.org/uniprotkb/stream?'+urllib.parse.urlencode(dict(query=q,format='fasta',includeIsoform='true',compressed='true'));received=None
 for attempt in range(1,4):
  try:
   raw,rec=fetch(url,f'qualified107778.includeIsoform.attempt{attempt}.fasta.response',512*2**20);received=raw;break
  except Exception:
   if attempt==3:live('stopped; finite complete-stream retries exhausted');raise
   time.sleep(2**attempt)
 assert received is not None
 live('whole FASTA identity/form/sequence verification',path=str(raw))
 with gzip.open(R/'outputs/data-intake/20261031-subagent/raw/uniprot2026_03/reviewed-annotation4-5-sequences.tsv.gz','rt') as f:baseline={r['Entry']:r['Sequence'] for r in csv.DictReader(f,delimiter='\t')}
 supp=json.loads((B/'physical-form-supplement-v2/manifest.json').read_text());forms=collections.defaultdict(list)
 for ref in supp['files']:
  if ref['kind']=='forms':
   p=R/ref['path'];assert sha(p)==ref['sha256']
   for row in pq.read_table(p).to_pylist():
    r=json.loads(row['record_json'])
    if r['form_kind']=='alternative_product':forms[r['form_id']].append(r)
 outseq=OUT/'native-sequences';outseq.mkdir();rows=[];files=[];ids=set();canon=set();isoforms=set();unmapped=[];alphabet=collections.Counter();source_status=collections.Counter();totalAA=0
 def flush():
  if not rows:return
  p=outseq/f'{len(files):04d}.parquet';pq.write_table(pa.Table.from_pylist(rows),p,compression='zstd');assert pq.read_table(p).to_pylist()==rows;files.append(dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p),rows=len(rows)));rows.clear()
 with raw.open('rb') as p:magic=p.read(2)
 handle=gzip.open(raw,'rt') if magic==b'\x1f\x8b' else raw.open()
 with handle as f:
  for index,(header,seq) in enumerate(fasta_records(f)):
   parts=header.split('|');assert len(parts)>=3 and parts[0]=='sp';acc=parts[1];assert acc not in ids and seq and not(set(seq)-set('ACDEFGHIKLMNPQRSTVWYBXZJUO')),'duplicate/invalid FASTA sequence';ids.add(acc);alphabet.update(seq);totalAA+=len(seq)
   parent=acc.split('-',1)[0];assert parent in baseline;iscanon=acc==parent
   if iscanon:assert seq==baseline[parent];canon.add(parent)
   else:isoforms.add(acc)
   contexts=forms.get(acc,[])
   if not iscanon and not contexts:unmapped.append(acc)
   for form in contexts:
    if form['verified_sequence'] is not None:assert seq==form['verified_sequence'],'displayed isoform sequence differs from KB canonical'
    source_status[form['sequence_status']]+=1
   rows.append(dict(source_record_index=index,source_accession=acc,parent_entry_accession=parent,is_canonical_record=iscanon,source_header=header,sequence=seq,sequence_length=len(seq),sequence_sha256=hashlib.sha256(seq.encode()).hexdigest(),full_source_ref=rec['path'],full_source_sha256=rec['sha256'],KB_form_contexts_json=json.dumps(contexts,sort_keys=True),KB_form_ID_matched=bool(contexts) if not iscanon else True,entry_annotation_score_inherited_to_form=False,canonical_annotation_transfer=False,training_admitted=False))
   if len(rows)==2048:flush()
 flush();assert canon==set(baseline) and len(canon)==107778
 described={fid for fid,rs in forms.items() if any(r['sequence_status']=='Described' for r in rs)};displayed={fid for fid,rs in forms.items() if any(r['sequence_status']=='Displayed' for r in rs)}
 result=dict(schema='mora3.UniProtIsoformNativeSource/v1',status='complete_official_stream_canonical_exact_and_native_roundtrip_verified',source_release='2026_03',source_receipt=rec,source_license='CC-BY-4.0',canonical_entries=len(canon),isoform_sequence_records=len(isoforms),source_records=len(ids),total_residues=totalAA,alphabet_counts=alphabet,additional_form_ids_not_inKBdescriptors=unmapped,Described_form_ids=len(described),Described_with_source_sequences=len(described&isoforms),Described_missing_source_sequences=sorted(described-isoforms),Displayed_form_ids=len(displayed),displayed_form_aliases_to_canonical='sourceDisplayedstatusandcanonicalexactsequence;notduplicateindependentobservations',matched_source_status_counts=source_status,native_shards=files,source_all_rows_full_sequence_retained=True,source_all_rows_Parquet_roundtrip=True,canonical_exact107778baseline=True,task_admitted=0,annotation_inheritance=False,isoform_coordinate_tasks='pendingperfeatureexactformsequenceandfuzzyboundvalidation',GOA_snapshot='independent2026-07-28;formquerycoverageunproven',new_independent_experiments=0)
 write(OUT/'manifest.json',result);live('finite complete source and native sequence QC finished',manifest=str(OUT/'manifest.json'),source_records=len(ids));print('COMPLETE',len(canon),len(isoforms),totalAA,flush=True)
if __name__=='__main__':run()
