from pathlib import Path
import urllib.request,urllib.parse,urllib.error,json,gzip,csv,hashlib,time,datetime,collections,concurrent.futures,threading,shutil,sys
import pyarrow as pa
import pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_corpus import annotation_score
B=R/'outputs/data-intake/20261031-uniprot-full-annotations-v1';(B/'raw').mkdir(parents=True);(B/'chunk-receipts').mkdir();(B/'entry-refs').mkdir();start=time.perf_counter();raw_budget=4*2**30;assert shutil.disk_usage(R).free>200*2**30+raw_budget+32*2**30
baseline=R/'outputs/data-intake/20261031-subagent/raw/uniprot2026_03/reviewed-annotation4-5-sequences.tsv.gz';assert hashlib.sha256(baseline.read_bytes()).hexdigest()=='e9bd53c005179cc68664f920f833273b241afb7196806ca512866869674d5133'
with gzip.open(baseline,'rt') as f:rows=list(csv.DictReader(f,delimiter='\t'))
expected={r['Entry']:r for r in rows};assert len(expected)==len(rows)==107778
query='(reviewed:true) AND ((annotation_score:4) OR (annotation_score:5))'
def source_url(q,stream=True,compressed=True,size=None):
 p=dict(query=q,format='json')
 if compressed:p['compressed']='true'
 if size is not None:p['size']=size
 return 'https://rest.uniprot.org/uniprotkb/'+('stream' if stream else 'search')+'?'+urllib.parse.urlencode(p)
def header_info(headers):
 return {k:headers.get(k) for k in ['X-UniProt-Release','X-UniProt-Release-Date','X-Total-Results','Content-Length','Content-Type','Content-Encoding','ETag','Last-Modified']}
def verify_entries(entries,wanted):
 refs=[]
 for i,e in enumerate(entries):
  acc=e['primaryAccession'];r=expected[acc];assert acc in wanted;assert e['entryType']=='UniProtKB reviewed (Swiss-Prot)';assert annotation_score(e['annotationScore'])==annotation_score(r['Annotation']);assert e['sequence']['value']==r['Sequence'];assert e['sequence']['length']==len(r['Sequence'])
  md5=e['sequence'].get('md5')
  if md5:assert hashlib.md5(r['Sequence'].encode()).hexdigest().upper()==md5.upper()
  refs.append(dict(primary_accession=acc,source_entry_index=i,canonical_sequence_sha256=hashlib.sha256(r['Sequence'].encode()).hexdigest(),annotation_score=annotation_score(e['annotationScore']),reviewed=True,sequence_length=e['sequence']['length'],entry_version=e['entryAudit'].get('entryVersion'),sequence_version=e['entryAudit'].get('sequenceVersion'),raw_top_level_keys_json=json.dumps(sorted(e)),comment_assertions=len(e.get('comments',[])),feature_assertions=len(e.get('features',[])),cross_references=len(e.get('uniProtKBCrossReferences',[]))))
 assert len(refs)==len(wanted) and {r['primary_accession'] for r in refs}==set(wanted)
 return refs
# A small complete-record query establishes actual response/release before bulk.
probe_url=source_url(query,stream=False,compressed=False,size=2)
with urllib.request.urlopen(probe_url,timeout=90) as response:probe=response.read(16*2**20+1);headers=header_info(response.headers)
assert len(probe)<=16*2**20 and headers['X-UniProt-Release']=='2026_03';assert int(headers['X-Total-Results'])==107778
p=B/'raw/full-record-probe.json';p.open('xb').write(probe);entries=json.loads(probe)['results'];assert len(entries)==2;probe_refs=verify_entries(entries,[e['primaryAccession'] for e in entries]);assert any(e.get('features') for e in entries) and any(e.get('references') for e in entries)
# Primary-accession field avoids secondary-accession overlap between partitions.
parts=[]
def split(ids,prefix=''):
 if len(ids)<=3000 and prefix:parts.append(dict(prefix=prefix,accessions=sorted(ids),expected_entries=len(ids)));return
 children=collections.defaultdict(list)
 for a in ids:children[a[len(prefix)]].append(a)
 for char,group in sorted(children.items()):split(group,prefix+char)
split(list(expected));assert sum(p['expected_entries'] for p in parts)==107778
parts.sort(key=lambda p:p['expected_entries'],reverse=True)
# Validate that the documented primary field/wildcard returns the exact expected partition count.
part=parts[0];q=query+' AND (accession_id:'+part['prefix']+'*)';partition_url=source_url(q,stream=False,compressed=False,size=1)
with urllib.request.urlopen(partition_url,timeout=90) as response:data=response.read(16*2**20+1);partition_headers=header_info(response.headers)
assert partition_headers['X-UniProt-Release']=='2026_03' and int(partition_headers['X-Total-Results'])==part['expected_entries'];assert json.loads(data)['results'][0]['primaryAccession'].startswith(part['prefix'])
(B/'raw/primary-prefix-probe.json').open('xb').write(data)
probe_receipt=dict(schema='mora3.UniProt_full_annotation_complete_record_probe.v1',status='verified_complete_json_source_probe',source_release='2026_03',source_release_date=headers['X-UniProt-Release-Date'],query=query,source_url=probe_url,raw_file=str(p.relative_to(R)),bytes=len(probe),sha256=hashlib.sha256(probe).hexdigest(),headers=headers,full_records_verified=len(entries),source_expected_entries=107778,selected_field_projection=False,top_level_keys=sorted(set().union(*(set(e) for e in entries))),source_sequence_score_reviewed_match=True,raw_all_nested_fields_preserved=True,source_probe_is_not_full_corpus_quality_pass=True,primary_prefix_probe=dict(prefix=part['prefix'],expected_entries=part['expected_entries'],headers=partition_headers,url=partition_url),partitions=len(parts),raw_budget_bytes=raw_budget,derived_budget_bytes=32*2**30,backend='3parallelcompleteRESTgzipstreams;noHTTPRange;immutableattempts;finitebackoff',license='CC-BY-4.0',seconds=time.perf_counter()-start)
with (B/'probe-receipt.json').open('x') as f:json.dump(probe_receipt,f,indent=2)
with (B/'source-partitions.json').open('x') as f:json.dump(dict(query=query,field='accession_id',source_release='2026_03',partition_count=len(parts),expected_entries=107778,partitions=parts),f,indent=2)
print('PROBE_VERIFIED',json.dumps({k:probe_receipt[k] for k in ['full_records_verified','source_release','bytes','partitions','seconds']}),flush=True)
lock=threading.Lock();serial=threading.Lock();state=dict(bytes=len(probe)+len(data),verified_chunks=0,verified_entries=0,throttled=False,failures=[],started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());receipts=[]
def live():
 (B/'live-acquisition.json').write_text(json.dumps(dict(state,status='running',max_connections=1 if state['throttled'] else 3,seconds=time.perf_counter()-start),indent=2)+'\n')
def acquire(part):
 prefix=part['prefix'];wanted=set(part['accessions']);q=query+' AND (accession_id:'+prefix+'*)';url=source_url(q)
 for attempt in range(1,4):
  path=B/'raw'/f'{prefix}-attempt{attempt}.full-json.response';ownbytes=0;sha=hashlib.sha256();t=time.perf_counter();guard=serial if state['throttled'] else None
  try:
   if guard:guard.acquire()
   with urllib.request.urlopen(url,timeout=150) as response:
    h=header_info(response.headers);assert h['X-UniProt-Release']=='2026_03';assert h['X-UniProt-Release-Date']==headers['X-UniProt-Release-Date']
    with path.open('xb') as f:
     while True:
      block=response.read(2**20)
      if not block:break
      with lock:
       assert state['bytes']+len(block)<=raw_budget;state['bytes']+=len(block)
      f.write(block);sha.update(block);ownbytes+=len(block)
   if h['Content-Length'] is not None:assert ownbytes==int(h['Content-Length'])
   body=path.read_bytes();compression='gzip' if body[:2]==b'\x1f\x8b' else 'none';decoded=gzip.decompress(body) if compression=='gzip' else body;obj=json.loads(decoded);entries=obj['results'];refs=verify_entries(entries,wanted)
   rp=B/'entry-refs'/f'{prefix}.parquet'
   for r in refs:r.update(raw_source_file=str(path.relative_to(R)),raw_source_sha256=sha.hexdigest(),source_release='2026_03')
   pq.write_table(pa.Table.from_pylist(refs),rp,compression='zstd');assert pq.read_table(rp).to_pylist()==refs
   rec=dict(schema='mora3.UniProt_full_annotation_chunk.v1',status='complete_raw_and_exact_entry_identity_verified',prefix=prefix,query=q,url=url,source_release='2026_03',headers=h,path=str(path.relative_to(R)),bytes=ownbytes,sha256=sha.hexdigest(),compression=compression,decoded_bytes=len(decoded),gzip_complete=compression=='gzip',entries=len(entries),exact_accession_set=True,all_reviewed_score4_5_and_sequence_match=True,raw_full_nested_fields=True,selected_field_projection=False,raw_top_level_keys=sorted(set().union(*(set(e) for e in entries))),entry_refs=dict(path=str(rp.relative_to(R)),rows=len(refs),bytes=rp.stat().st_size,sha256=hashlib.sha256(rp.read_bytes()).hexdigest()),seconds=time.perf_counter()-t,attempt=attempt,expected_byte_length=h['Content-Length'],dynamic_length_integrity='gzipCRC+fullJSONclosure+exactexpectedentryset whenContentLengthunavailable',source_completeness_not_task_admission=True)
   with (B/'chunk-receipts'/f'{prefix}-attempt{attempt}.json').open('x') as f:json.dump(rec,f,indent=2)
   with lock:state['verified_chunks']+=1;state['verified_entries']+=len(entries);live()
   print('CHUNK_VERIFIED',prefix,len(entries),ownbytes,round(rec['seconds'],2),flush=True);return rec
  except Exception as exc:
   error=dict(prefix=prefix,attempt=attempt,path=str(path.relative_to(R)),bytes=ownbytes,error=repr(exc),sha256=sha.hexdigest(),partial_preserved=path.exists())
   with (B/'chunk-receipts'/f'{prefix}-attempt{attempt}-failure.json').open('x') as f:json.dump(error,f,indent=2)
   with lock:
    state['failures'].append(error)
    if isinstance(exc,urllib.error.HTTPError) and exc.code in (429,503):state['throttled']=True
    live()
   if attempt==3:raise
  finally:
   if guard:guard.release()
  time.sleep(min(40,10*2**(attempt-1)))
# First small partition yields an early complete receipt while two larger streams proceed.
order=[parts[-1]]+parts[:-1]
with lock:live()
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 futures=[pool.submit(acquire,p) for p in order]
 for future in concurrent.futures.as_completed(futures):receipts.append(future.result())
assert sum(r['entries'] for r in receipts)==107778
allrefs=[]
for rec in receipts:allrefs.extend(pq.read_table(R/rec['entry_refs']['path']).to_pylist())
assert len(allrefs)==len({r['primary_accession'] for r in allrefs})==107778 and {r['primary_accession'] for r in allrefs}==set(expected)
full=dict(schema='mora3.UniProt_comprehensive_full_source_acquisition.v1',status='full107778_lossless_records_verified',source_release='2026_03',source_release_date=headers['X-UniProt-Release-Date'],license='CC-BY-4.0',query=query,selected_field_projection=False,source_entries=107778,verified_accessions_sequences_scores_reviewed=107778,raw_complete_chunks=len(receipts),raw_chunk_bytes=sum(r['bytes'] for r in receipts),raw_decoded_bytes=sum(r['decoded_bytes'] for r in receipts),all_raw_nested_fields_preserved=True,chunk_receipts=receipts,probe_receipt=str((B/'probe-receipt.json').relative_to(R)),raw_does_not_mean_typed_or_model_ready=True,typed_projection='pending',model_native_objectives='pending',companion_sources='GOqualifiers/relations,fullisoformsequenceandexternalwebsiteoverlay coverageauditpending',new_independent_observations=0,optimizer_steps=0,seconds=time.perf_counter()-start)
with (B/'acquisition-manifest.json').open('x') as f:json.dump(full,f,indent=2)
with lock:state['status']='complete';(B/'live-acquisition.json').write_text(json.dumps(dict(state,status='complete',seconds=time.perf_counter()-start),indent=2)+'\n')
print('FULL_SOURCE_COMPLETE',json.dumps({k:full[k] for k in ['source_entries','raw_complete_chunks','raw_chunk_bytes','raw_decoded_bytes','seconds']}),flush=True)
