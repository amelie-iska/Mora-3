import pathlib,json,hashlib,time,requests,gzip
R=pathlib.Path('/home/iska/Documents/amelie/bio/Mora-3');B=R/'outputs/data-intake/20261031-sciplex-original-context-v1';B.mkdir(exist_ok=False);(B/'raw').mkdir();(B/'logs').mkdir();url='https://ftp.ncbi.nlm.nih.gov/geo/series/GSE139nnn/GSE139944/soft/GSE139944_family.soft.gz';s=requests.Session();s.trust_env=False;p=B/'raw/GSE139944_family.soft.gz';start=time.perf_counter();rout=dict(schema='mora3.public_GEO_metadata_receipt.v1',url=url,maximum_bytes=16777216,source_role='original public series/sample metadata;not new response example admission',web_page_access_note='GEO HTML crawler received recaptcha;official public FTP data service used without solving challenge or accessing credentials')
try:
 with s.get(url,stream=True,timeout=(20,60)) as response:
  rout.update(http=response.status_code,response_headers={k:response.headers.get(k) for k in ['Content-Length','Last-Modified','ETag','Content-Type']});response.raise_for_status();size=0;h=hashlib.sha256()
  with p.open('xb') as f:
   for block in response.iter_content(65536):
    if not block:continue
    size+=len(block);assert size<=rout['maximum_bytes'];f.write(block);h.update(block)
  rout.update(path=str(p),bytes=size,sha256=h.hexdigest(),status='downloaded_complete')
 selected=[];series=[];insample=False;inseries=False;decompressed=0
 with gzip.open(p,'rt') as f:
  for line in f:
   decompressed+=len(line.encode());assert decompressed<=128*2**20
   if line.startswith('^'):
    insample=line.strip()=='^SAMPLE = GSM4150377';inseries=line.strip()=='^SERIES = GSE139944'
   if insample and (line.startswith('^') or line.startswith('!')):selected.append(line.rstrip())
   if inseries and line.startswith('!Series_'):series.append(line.rstrip())
 assert selected
 rout.update(gzip_complete=True,decompressed_bytes=decompressed,selected_sample='GSM4150377',selected_sample_metadata_lines=len(selected),series_metadata_lines=len(series),status='verified_original_metadata')
 with (B/'selected-sample-metadata.json').open('x') as f:json.dump(dict(source_receipt='acquisition-receipt.json',sample_lines=selected,series_lines=series),f,indent=2)
except Exception as error:rout.update(status='failed_partial_retained' if p.exists() else 'request_failed_no_data',error=type(error).__name__+': '+str(error))
rout['seconds']=time.perf_counter()-start
with (B/'acquisition-receipt.json').open('x') as f:json.dump(rout,f,indent=2)
print(json.dumps(rout),flush=True)
if rout['status']=='verified_original_metadata':print(json.dumps(dict(sample_lines=selected,series_files=[x for x in series if 'supplementary_file' in x])),flush=True)
