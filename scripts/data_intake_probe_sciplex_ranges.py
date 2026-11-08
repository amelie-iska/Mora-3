"""Small public download range probes; no whole matrix acquisition or old-file mutation."""
from pathlib import Path
import datetime,hashlib,json,urllib.request,urllib.error,time
B=Path('outputs/data-intake/20261101-sciplex-lineage-resolution-v1/range-preflight-v1');B.mkdir(exist_ok=False);base='https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4150nnn/GSM4150377/suppl/';names=['GSM4150377_sciPlex2_A549_Transcription_Modulators_gene.annotations.txt.gz','GSM4150377_sciPlex2_A549_Transcription_Modulators_UMI.count.matrix.gz'];receipts=[]
for name in names:
 for suffix,kind in [('', 'canonical'),('?download=1','public_download_query')]:
  url=base+name+suffix;start=time.perf_counter();d=dict(url=url,request_range='bytes=0-1023',role='public_source_transport_probe_only;notcompleteassetverification')
  try:
   with urllib.request.urlopen(urllib.request.Request(url,headers={'Range':'bytes=0-1023','User-Agent':'Mora3-private-source-integrity-preflight/1.0'}),timeout=20) as r:body=r.read(1024);d.update(status=r.status,headers={k:v for k,v in r.headers.items() if k.lower() in ['content-length','content-range','accept-ranges','etag','last-modified','content-type']},final_url=r.geturl())
   p=B/(name+'.'+kind+'.prefix');p.open('xb').write(body);d.update(prefix=dict(path=str(p),bytes=len(body),sha256=hashlib.sha256(body).hexdigest()),gzip_magic=body[:2]==b'\x1f\x8b',range_compatible=r.status==206 and r.headers.get('Content-Range','').startswith('bytes 0-1023/'))
  except Exception as e:d.update(status='failed',error=repr(e))
  d['seconds']=time.perf_counter()-start;receipts.append(d);print(name,kind,d['status'],d.get('range_compatible'),flush=True)
(B/'manifest.json').open('x').write(json.dumps(dict(schema='mora3.SciPlexPublicRangeProbe/v1',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),receipts=receipts,whole_new_assets_acquired=0,source_original_count_values_verified=False,no_credentials_or_old_source_access=True),indent=2)+'\n')
