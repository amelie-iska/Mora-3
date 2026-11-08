import pathlib,json,re,time,hashlib,threading,concurrent.futures,requests,shutil,collections
R=pathlib.Path('/home/iska/Documents/amelie/bio/Mora-3');B=R/'outputs/data-intake/20261031-pathway-semantics-v2/chebi-context-v1';source=R/'outputs/data-intake/20261031-pathway-semantics-v2/source-native-pathway-models.jsonl';ids=set();byqual=collections.defaultdict(set)
with source.open() as f:
 for line in f:
  d=json.loads(line)
  for entity in d['species'].values():
   for qualifier,urls in entity['qualifiers'].items():
    if qualifier not in ('is','hasPart'):continue
    for url in urls:
     if 'chebi' in url.lower():
      found=re.search(r'CHEBI[:_](\d+)',url,re.I)
      if found:ids.add(found[1]);byqual[qualifier].add(found[1])
plan=dict(schema='mora3.public_reference_context_acquisition.v1',source_graphs=str(source.relative_to(R)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_chebi_ids=len(ids),counts_by_qualifier={k:len(v) for k,v in byqual.items()},maximum_total_new_bytes=268435456,maximum_one_response_bytes=2097152,minimum_free_bytes=214748364800,max_concurrency=3,one_attempt_per_id=True,license='CC-BY-4.0 ChEBI;official about page https://www.ebi.ac.uk/chebi/about',credential_access=False,purpose='reference structure/class/secondaryID audit;not measured biological/physical property supervision')
with (B/'acquisition-plan.json').open('x') as f:json.dump(plan,f,indent=2)
assert shutil.disk_usage(R).free>plan['minimum_free_bytes']+plan['maximum_total_new_bytes'];lock=threading.Lock();stop=threading.Event();bytes_used=0;start=time.perf_counter();existing={r['id'].split(':')[-1]:r for r in json.loads((B/'probe-receipts.json').read_text())};done=[]
def fetch(value):
 global bytes_used
 if value in existing:return dict(existing[value],reuse_existing_probe=True,status='verified_existing_probe')
 if stop.is_set():return dict(id='CHEBI:'+value,status='not_requested_after_budget_or_rate_gate')
 time.sleep(.25);session=requests.Session();session.trust_env=False;url='https://www.ebi.ac.uk/chebi/backend/api/public/compound/CHEBI:'+value+'/';p=B/'raw'/('CHEBI-'+value+'.json');rout=dict(id='CHEBI:'+value,url=url,path=str(p),license='CC-BY-4.0',attempts=1)
 try:
  with session.get(url,stream=True,timeout=(15,45)) as response:
   rout.update(http=response.status_code,url=response.url,response_headers={k:response.headers.get(k) for k in ['ETag','Last-Modified','Content-Length','Content-Type']})
   if response.status_code in (401,403,429):stop.set();rout['status']='access_or_rate_gate_no_retry';return rout
   n=0;h=hashlib.sha256()
   with p.open('xb') as f:
    for chunk in response.iter_content(65536):
     if not chunk:continue
     n+=len(chunk)
     with lock:
      bytes_used+=len(chunk)
      if n>plan['maximum_one_response_bytes'] or bytes_used>plan['maximum_total_new_bytes'] or time.perf_counter()-start>1800 or shutil.disk_usage(R).free<plan['minimum_free_bytes']:
       stop.set();raise ValueError('finite source byte/time/free-space cap reached')
     f.write(chunk);h.update(chunk)
   rout.update(bytes=n,sha256=h.hexdigest(),status='downloaded_complete' if response.status_code==200 else 'public_source_unavailable_response_retained')
   if response.status_code==200:
    data=json.loads(p.read_text());rout.update(returned_accession=data.get('chebi_accession'),modified_on=data.get('modified_on'),stars=data.get('stars'),is_released=data.get('is_released'),has_default_structure=data.get('default_structure') is not None)
 except Exception as error:rout.update(status='failed_partial_retained' if p.exists() else 'request_failed_no_data',error=type(error).__name__+': '+str(error),partial_bytes=p.stat().st_size if p.exists() else 0)
 return rout
with concurrent.futures.ThreadPoolExecutor(max_workers=plan['max_concurrency']) as pool:
 for item in pool.map(fetch,sorted(ids,key=int)):
  done.append(item)
  if len(done)%100==0:
   checkpoint=dict(completed=len(done),expected=len(ids),new_bytes=bytes_used,status_counts=dict(collections.Counter(x['status'] for x in done)),seconds=time.perf_counter()-start);print(json.dumps(checkpoint),flush=True);(B/'live-acquisition-progress.json').write_text(json.dumps(checkpoint,indent=2))
receipt=dict(schema='mora3.Chebi_reference_context_receipt.v1',source_ids=len(ids),records=done,raw_complete=sum(r.get('http')==200 and r['status'] in ('verified_existing_probe','downloaded_complete') for r in done),new_bytes=bytes_used,status_counts=dict(collections.Counter(x['status'] for x in done)),stopped_on_access_rate_or_budget=stop.is_set(),seconds=time.perf_counter()-start,source_role='metadata/structure/reference context;not experimental task admission')
with (B/'acquisition-receipt.json').open('x') as f:json.dump(receipt,f,indent=2)
print('CHEBI_REFERENCE_COMPLETE',json.dumps({k:v for k,v in receipt.items() if k!='records'}),flush=True)
