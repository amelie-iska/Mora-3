import pathlib,json,hashlib,gzip,time,requests
R=pathlib.Path('/home/iska/Documents/amelie/bio/Mora-3');B=R/'outputs/data-intake/20261031-sciplex-original-context-v1';url='https://ftp.ncbi.nlm.nih.gov/geo/series/GSE139nnn/GSE139944/soft/GSE139944_family.soft.gz';p=B/'raw/GSE139944_family.http-range-attempt2.soft.gz';s=requests.Session();s.trust_env=False;out=dict(schema='mora3.GEO_metadata_bounded_range_retry.v1',url=url,reason='priorregularGETreturnedheadersbutzero-bodyconnectionEOF',attempt=2,maximum_bytes=16777216)
try:
 r=s.get(url,headers={'Range':'bytes=0-4035','Accept-Encoding':'identity'},timeout=(15,25));out.update(http=r.status_code,headers={k:r.headers.get(k) for k in ['Content-Length','Content-Range','Content-Encoding','Last-Modified','Content-Type']});r.raise_for_status();data=r.content;assert len(data)<=out['maximum_bytes'];p.write_bytes(data);out.update(path=str(p),bytes=len(data),sha256=hashlib.sha256(data).hexdigest());text=gzip.decompress(data).decode();selected=[];series=[];sample=False;active_series=False
 for line in text.splitlines():
  if line.startswith('^'):sample=line=='^SAMPLE = GSM4150377';active_series=line=='^SERIES = GSE139944'
  if sample and (line.startswith('^') or line.startswith('!')):selected.append(line)
  if active_series and line.startswith('!Series_'):series.append(line)
 assert selected;out.update(status='verified_original_metadata',gzip_complete=True,selected_sample='GSM4150377')
 with (B/'selected-sample-metadata-attempt2.json').open('x') as f:json.dump(dict(sample_lines=selected,series_lines=series),f,indent=2)
except Exception as error:out.update(status='unavailable_after_two_bounded_transport_attempts',error=type(error).__name__+': '+str(error))
with (B/'acquisition-receipt-attempt2.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out),flush=True)
if out['status']=='verified_original_metadata':print(json.dumps(dict(sample_lines=selected,series_files=[x for x in series if 'supplementary_file' in x])),flush=True)
