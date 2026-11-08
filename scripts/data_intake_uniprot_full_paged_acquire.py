from pathlib import Path
import urllib.request,urllib.parse,urllib.error,json,gzip,csv,hashlib,time,datetime,collections,concurrent.futures,threading,shutil,sys,re
import pyarrow as pa
import pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_corpus import annotation_score
B=R/'outputs/data-intake/20261031-uniprot-full-annotations-v1';V=B/'paged-acquisition-v2';(V/'raw').mkdir(parents=True);(V/'page-receipts').mkdir();(V/'entry-refs').mkdir();start=time.perf_counter();budget=4*2**30;assert shutil.disk_usage(R).free>200*2**30+budget+32*2**30
bp=R/'outputs/data-intake/20261031-subagent/raw/uniprot2026_03/reviewed-annotation4-5-sequences.tsv.gz';assert hashlib.sha256(bp.read_bytes()).hexdigest()=='e9bd53c005179cc68664f920f833273b241afb7196806ca512866869674d5133'
with gzip.open(bp,'rt') as f:expected={r['Entry']:r for r in csv.DictReader(f,delimiter='\t')}
assert len(expected)==107778
basequery='(reviewed:true) AND ((annotation_score:4) OR (annotation_score:5))'
def url(q,size=500,compressed=True):
 p=dict(query=q,format='json',size=size)
 if compressed:p['compressed']='true'
 return 'https://rest.uniprot.org/uniprotkb/search?'+urllib.parse.urlencode(p)
def info(h):return {k:h.get(k) for k in ['X-UniProt-Release','X-UniProt-Release-Date','X-Total-Results','Content-Length','Content-Type','Content-Encoding','ETag','Last-Modified','Link']}
def verify(entries,wanted):
 refs=[]
 for i,e in enumerate(entries):
  acc=e['primaryAccession'];assert acc in wanted;r=expected[acc];assert e['entryType']=='UniProtKB reviewed (Swiss-Prot)';assert annotation_score(e['annotationScore'])==annotation_score(r['Annotation']);assert e['sequence']['value']==r['Sequence'];assert e['sequence']['length']==len(r['Sequence'])
  if e['sequence'].get('md5'):assert e['sequence']['md5'].upper()==hashlib.md5(r['Sequence'].encode()).hexdigest().upper()
  refs.append(dict(primary_accession=acc,source_entry_index=i,canonical_sequence_sha256=hashlib.sha256(r['Sequence'].encode()).hexdigest(),annotation_score=annotation_score(e['annotationScore']),reviewed=True,sequence_length=e['sequence']['length'],entry_version=e['entryAudit'].get('entryVersion'),sequence_version=e['entryAudit'].get('sequenceVersion'),raw_top_level_keys_json=json.dumps(sorted(e)),comments=len(e.get('comments',[])),features=len(e.get('features',[])),cross_references=len(e.get('uniProtKBCrossReferences',[]))))
 assert len({r['primary_accession'] for r in refs})==len(refs)
 return refs
# Persist first verified full-record receipt before any cohort validation/bulk.
pu=url(basequery,size=2,compressed=False)
with urllib.request.urlopen(pu,timeout=90) as response:raw=response.read(16*2**20+1);headers=info(response.headers)
assert len(raw)<=16*2**20 and headers['X-UniProt-Release']=='2026_03' and int(headers['X-Total-Results'])==107778
p=V/'raw/full-record-probe.json';p.open('xb').write(raw);entries=json.loads(raw)['results'];assert len(entries)==2;verify(entries,expected);assert any(e.get('features') for e in entries) and any(e.get('references') for e in entries)
probe=dict(schema='mora3.UniProt_full_record_probe.v2',status='verified_complete_records',path=str(p.relative_to(R)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),url=pu,headers=headers,source_entries_expected=107778,full_records_verified=2,all_source_nested_fields_retained=True,selected_field_projection=False,all_exact_sequence_reviewed_score4_5=True,raw_top_level_keys=sorted(set().union(*(set(e) for e in entries))),not_whole_corpus_quality_proof=True,previous_failure='primary-accessionwildcardqueryHTTP400;partialsourceprobe preserved;nooldbulkjoblive',backend='official500recordcursorpages;3disjointlengthcohorts;noRangeorunsafeResume',license='CC-BY-4.0',raw_budget_bytes=budget,derived_budget_bytes=32*2**30)
with (B/'probe-receipt.json').open('x') as f:json.dump(probe,f,indent=2)
print('FULL_RECORD_PROBE_VERIFIED',len(raw),'2026_03',len(entries),flush=True)
lengths=sorted(len(r['Sequence']) for r in expected.values());cuts=[min(lengths),lengths[len(lengths)//3],lengths[2*len(lengths)//3],max(lengths)]
ranges=[(cuts[0],cuts[1]),(cuts[1]+1,cuts[2]),(cuts[2]+1,cuts[3])];cohorts=[]
for lo,hi in ranges:
 ids={a for a,r in expected.items() if lo<=len(r['Sequence'])<=hi};q=basequery+f' AND (length:[{lo} TO {hi}])';u=url(q,size=1,compressed=False)
 with urllib.request.urlopen(u,timeout=90) as response:body=response.read(16*2**20+1);h=info(response.headers)
 assert len(body)<=16*2**20 and h['X-UniProt-Release']=='2026_03' and h['X-UniProt-Release-Date']==headers['X-UniProt-Release-Date'] and int(h['X-Total-Results'])==len(ids)
 entry=json.loads(body)['results'];assert len(entry)==1;verify(entry,ids)
 name=f'length-{lo}-{hi}';cp=V/'raw'/f'{name}-query-probe.json';cp.open('xb').write(body);cohorts.append(dict(name=name,minimum_length=lo,maximum_length=hi,query=q,initial_url=url(q),expected_entries=len(ids),expected_accessions=sorted(ids),query_probe=dict(url=u,headers=h,bytes=len(body),sha256=hashlib.sha256(body).hexdigest())))
assert sum(c['expected_entries'] for c in cohorts)==107778
with (V/'cohort-plan.json').open('x') as f:json.dump(dict(schema='mora3.UniProt_full_record_disjoint_cohorts.v1',source_release='2026_03',cohorts=cohorts,source_total=107778,all_cohort_query_counts_schema_verified_before_bulk=True),f,indent=2)
print('COHORTS_VERIFIED',[(c['name'],c['expected_entries']) for c in cohorts],flush=True)
lock=threading.Lock();serial=threading.Lock();state=dict(status='running',verified_pages=0,verified_entries=0,bytes=sum(p.stat().st_size for p in (V/'raw').iterdir()),throttled=False,failures=[],started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());global_seen=set()
def live():
 (V/'live-acquisition.json').write_text(json.dumps(dict(state,parallel_cohorts=1 if state['throttled'] else 3,seconds=time.perf_counter()-start),indent=2)+'\n')
def run_cohort(c):
 nexturl=c['initial_url'];seen=set();receipts=[];page=0;wanted=set(c['expected_accessions'])
 while nexturl:
  assert page<300 and urllib.parse.urlparse(nexturl).netloc=='rest.uniprot.org' and 'fields' not in urllib.parse.parse_qs(urllib.parse.urlparse(nexturl).query)
  for attempt in range(1,4):
   path=V/'raw'/f'{c["name"]}-page{page:04d}-attempt{attempt}.full-json.response';sha=hashlib.sha256();n=0;t=time.perf_counter();guard=serial if state['throttled'] else None
   try:
    if guard:guard.acquire()
    with urllib.request.urlopen(nexturl,timeout=150) as response:
     h=info(response.headers);assert h['X-UniProt-Release']=='2026_03' and h['X-UniProt-Release-Date']==headers['X-UniProt-Release-Date'];assert int(h['X-Total-Results'])==c['expected_entries']
     with path.open('xb') as f:
      while True:
       block=response.read(2**20)
       if not block:break
       with lock:assert state['bytes']+len(block)<=budget;state['bytes']+=len(block)
       f.write(block);sha.update(block);n+=len(block)
    if h['Content-Length'] is not None:assert n==int(h['Content-Length'])
    body=path.read_bytes();compression='gzip' if body[:2]==b'\x1f\x8b' else 'none';decoded=gzip.decompress(body) if compression=='gzip' else body;obj=json.loads(decoded);es=obj['results'];assert 0<len(es)<=500;refs=verify(es,wanted);ids={r['primary_accession'] for r in refs};assert not ids&seen
    link=re.search(r'<([^>]+)>;\s*rel="?next"?',h['Link'] or '');following=link.group(1) if link else None
    assert len(seen)+len(ids)<=c['expected_entries'];assert bool(following)==(len(seen)+len(ids)<c['expected_entries'])
    for r in refs:r.update(raw_source_file=str(path.relative_to(R)),raw_source_sha256=sha.hexdigest(),source_release='2026_03')
    rp=V/'entry-refs'/f'{c["name"]}-page{page:04d}.parquet';pq.write_table(pa.Table.from_pylist(refs),rp,compression='zstd');assert pq.read_table(rp).to_pylist()==refs
    rec=dict(schema='mora3.UniProt_full_record_page.v1',status='complete_raw_and_exact_entry_identity_verified',cohort=c['name'],page_index=page,url=nexturl,next_link=following,headers=h,source_release='2026_03',source_release_date=h['X-UniProt-Release-Date'],path=str(path.relative_to(R)),bytes=n,sha256=sha.hexdigest(),compression=compression,decoded_bytes=len(decoded),gzip_complete=compression=='gzip',entries=len(es),all_accessions_sequences_scores_reviewed_exact=True,all_raw_nested_fields_preserved=True,selected_field_projection=False,raw_top_level_keys=sorted(set().union(*(set(e) for e in es))),entry_refs=dict(path=str(rp.relative_to(R)),rows=len(refs),bytes=rp.stat().st_size,sha256=hashlib.sha256(rp.read_bytes()).hexdigest()),seconds=time.perf_counter()-t,expected_byte_length=h['Content-Length'],length_method='providerContentLengthwhenpresent;gzipCRC+JSONclosure+exactpage/cursor/entrygatesfordynamicchunkedresponse',task_admission=False)
    with (V/'page-receipts'/f'{c["name"]}-page{page:04d}-attempt{attempt}.json').open('x') as f:json.dump(rec,f,indent=2)
    with lock:assert not ids&global_seen;global_seen.update(ids);state['verified_pages']+=1;state['verified_entries']+=len(es);live()
    seen.update(ids);receipts.append(rec);nexturl=following;page+=1;print('FULL_PAGE_VERIFIED',c['name'],page,len(es),n,round(rec['seconds'],2),flush=True);break
   except Exception as exc:
    retry=isinstance(exc,(urllib.error.URLError,TimeoutError,ConnectionError)) and not (isinstance(exc,urllib.error.HTTPError) and exc.code not in (429,500,502,503,504));error=dict(cohort=c['name'],page=page,attempt=attempt,error=repr(exc),path=str(path.relative_to(R)),bytes=n,sha256=sha.hexdigest(),partial_preserved=path.exists())
    with (V/'page-receipts'/f'{c["name"]}-page{page:04d}-attempt{attempt}-failure.json').open('x') as f:json.dump(error,f,indent=2)
    with lock:
     state['failures'].append(error)
     if isinstance(exc,urllib.error.HTTPError) and exc.code in (429,503):state['throttled']=True
     live()
    if not retry or attempt==3:raise
   finally:
    if guard:guard.release()
   time.sleep(min(60,10*2**(attempt-1)))
 assert seen==wanted;return receipts
with lock:live()
allreceipts=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 for future in concurrent.futures.as_completed([pool.submit(run_cohort,c) for c in cohorts]):allreceipts+=future.result()
assert global_seen==set(expected) and sum(r['entries'] for r in allreceipts)==107778
full=dict(schema='mora3.UniProt_comprehensive_full_source_acquisition.v2',status='full107778_lossless_records_verified',source_release='2026_03',source_release_date=headers['X-UniProt-Release-Date'],query=basequery,license='CC-BY-4.0',source_entries=107778,verified_accessions_sequences_scores_reviewed=107778,raw_complete_pages=len(allreceipts),raw_page_bytes=sum(r['bytes'] for r in allreceipts),raw_decoded_bytes=sum(r['decoded_bytes'] for r in allreceipts),all_source_nested_fields_preserved=True,selected_field_projection=False,cohort_plan=str((V/'cohort-plan.json').relative_to(R)),page_receipts=allreceipts,probe_receipt=str((B/'probe-receipt.json').relative_to(R)),typed_projection='pending',native_model_objectives='pending',fullKB_not_all_external_overlay_companions=True,GOA_companion='sourceworkerowned;separateversionrequired;notyetbound',new_independent_observations=0,optimizer_steps=0,seconds=time.perf_counter()-start)
with (B/'acquisition-manifest.json').open('x') as f:json.dump(full,f,indent=2)
with lock:state['status']='complete';(V/'live-acquisition.json').write_text(json.dumps(dict(state,seconds=time.perf_counter()-start),indent=2)+'\n')
print('FULL_SOURCE_COMPLETE',json.dumps({k:full[k] for k in ['source_entries','raw_complete_pages','raw_page_bytes','raw_decoded_bytes','seconds']}),flush=True)
