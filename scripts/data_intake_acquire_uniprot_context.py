import pathlib,json,time,hashlib,shutil,requests,gzip,csv
R=pathlib.Path('/home/iska/Documents/amelie/bio/Mora-3');B=R/'outputs/data-intake/20261031-protein-function-context-v1';probe=json.loads((B/'research/official-field-probe.json').read_text());assert probe['http']==200 and probe['release']=='2026_03';start=time.perf_counter();path=B/'raw/reviewed-annotation4-5-function-gene-context.tsv.gz';minimum=214748364800;maximum=268435456;assert shutil.disk_usage(R).free>=minimum+maximum
s=requests.Session();s.trust_env=False
receipt=dict(schema='mora3.public_source_receipt.v1',status='starting',fields=probe['fields'],query=probe['query'],license='CC-BY-4.0',expected_release='2026_03',path=str(path),maximum_bytes=maximum,credentials=False,finite_attempts=1)
try:
 with s.get('https://rest.uniprot.org/uniprotkb/stream',params=dict(query=probe['query'],format='tsv',fields=probe['fields'],compressed='true'),headers={'Accept-Encoding':'identity'},stream=True,timeout=(20,60)) as response:
  receipt.update(url=response.url,http=response.status_code,release=response.headers.get('x-uniprot-release'),release_date=response.headers.get('x-uniprot-release-date'),response_headers={k:response.headers.get(k) for k in ['Content-Length','Content-Type','ETag','Last-Modified','Content-Encoding']});response.raise_for_status();assert receipt['release']==probe['release'];n=0;h=hashlib.sha256()
  with path.open('xb') as f:
   for block in response.iter_content(2**20):
    if not block:continue
    n+=len(block);assert n<=maximum and time.perf_counter()-start<900 and shutil.disk_usage(R).free>=minimum;f.write(block);h.update(block)
    if n%(8*2**20)<len(block):print(json.dumps(dict(downloaded_bytes=n,seconds=time.perf_counter()-start)),flush=True)
  receipt.update(bytes=n,sha256=h.hexdigest(),status='downloaded_complete_hash_pinned')
 with gzip.open(path,'rt') as f:
  reader=csv.DictReader(f,delimiter='\t');columns=reader.fieldnames;rows=sum(1 for _ in reader)
 assert rows==107778
 receipt.update(source_rows=rows,headers=columns,gzip_complete=True,status='downloaded_verified_release_rows')
except Exception as error:
 receipt.update(status='failed_partial_retained' if path.exists() else 'request_failed_no_data',error=type(error).__name__+': '+str(error),partial_bytes=path.stat().st_size if path.exists() else 0)
receipt['seconds']=time.perf_counter()-start
with (B/'acquisition-receipt.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt),flush=True)
