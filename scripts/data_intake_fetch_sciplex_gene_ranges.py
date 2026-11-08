"""Finite exact-range source fetch after whole-body EOF; all attempts retained."""
from pathlib import Path
import datetime,gzip,hashlib,json,urllib.request
B=Path('outputs/data-intake/20261101-sciplex-original-count-assets-v1/gene-ranges-v1');B.mkdir(exist_ok=False);url='https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4150nnn/GSM4150377/suppl/GSM4150377_sciPlex2_A549_Transcription_Modulators_gene.annotations.txt.gz?download=1';size=430187;parts=[];chunks=[]
try:
 for i,start in enumerate(range(0,size,128*1024)):
  end=min(size-1,start+128*1024-1)
  with urllib.request.urlopen(urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}','User-Agent':'Mora3-private-source-integrity-range/1.0'}),timeout=30) as r:
   assert r.status==206 and r.headers.get('Content-Range')==f'bytes {start}-{end}/{size}' and r.headers.get('Last-Modified')=='Tue, 05 Nov 2019 16:25:07 GMT';data=r.read(end-start+2);assert len(data)==end-start+1
  p=B/f'part-{i:02d}.raw';p.open('xb').write(data);parts.append(dict(path=str(p),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),start=start,end=end));chunks.append(data)
 blob=b''.join(chunks);assert len(blob)==size;raw=B/'GSM4150377_sciPlex2_A549_Transcription_Modulators_gene.annotations.txt.gz';raw.open('xb').write(blob);plain=gzip.decompress(blob);plain.decode('utf-8');out=B/'gene.annotations.txt';out.open('xb').write(plain)
 result=dict(schema='mora3.ExactPublicGEOHeaderBoundRangeAsset/v1',status='complete_exact206_ranges_same_LastModified_size_and_gzipCRC_verified',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),url=url,range_parts=parts,raw=dict(path=str(raw),bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest()),decoded=dict(path=str(out),bytes=len(plain),sha256=hashlib.sha256(plain).hexdigest()),prior_wholebody_EOF_preserved=True,source_models_and_old_data_changed=False,new_tasks=0,physical_dose_mapping_verified=False)
 (B/'manifest.json').open('x').write(json.dumps(result,indent=2)+'\n');print('COMPLETE',len(blob),len(plain))
except BaseException as e:
 (B/'failure.json').open('x').write(json.dumps(dict(error=repr(e)),indent=2)+'\n');raise
