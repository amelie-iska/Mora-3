"""Bounded exact-range extraction of one public native TAR member; no full archive download."""
from pathlib import Path
import datetime,gzip,hashlib,json,tarfile,time,urllib.request
B=Path('outputs/data-intake/20261101-sciplex-original-count-assets-v1/official-tar-gene-range-v1');B.mkdir(exist_ok=False);url='https://ftp.ncbi.nlm.nih.gov/geo/series/GSE139nnn/GSE139944/suppl/GSE139944_RAW.tar?download=1';target='GSM4150377_sciPlex2_A549_Transcription_Modulators_gene.annotations.txt.gz';ledger=[];fetched=0;archive_total=None;lastmod=None

def get_range(start,end,label):
 global fetched,archive_total,lastmod
 time.sleep(.35);d=dict(url=url,requested_start=start,requested_end=end,label=label);p=B/(f'{len(ledger):03d}-'+label+'.range')
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity','User-Agent':'Mora3-private-source-range-integrity/1.0'}),timeout=25) as r:
   d.update(status=r.status,headers={k:v for k,v in r.headers.items() if k.lower() in ['content-range','content-length','last-modified','etag','content-type']});body=r.read(end-start+2)
  p.open('xb').write(body);fetched+=len(body);d['received_bytes']=len(body);d['sha256']=hashlib.sha256(body).hexdigest();d['path']=str(p);ledger.append(d);(B/'range-receipts-progress.json').write_text(json.dumps(ledger,indent=2)+'\n');assert fetched<=8*2**20;assert d['status']==206 and len(body)==end-start+1;cr=d['headers'].get('Content-Range',d['headers'].get('content-range'));prefix=f'bytes {start}-{end}/';assert cr.startswith(prefix);total=int(cr[len(prefix):]);lm=d['headers'].get('Last-Modified',d['headers'].get('last-modified'))
  if archive_total is None:archive_total=total;lastmod=lm
  else:assert archive_total==total and lm==lastmod
  return body
 except Exception as e:
  if d not in ledger:d['error']=repr(e);ledger.append(d)
  raise
try:
 offset=0;members=[];pending_name=None;found=None
 for i in range(64):
  block=get_range(offset,offset+511,f'header-{i:02d}')
  if block==b'\0'*512:break
  info=tarfile.TarInfo.frombuf(block,'utf-8','strict');name=pending_name or info.name;pending_name=None;members.append(dict(name=name,size=info.size,type=info.type.decode('ascii','replace'),header_offset=offset))
  if info.type==tarfile.GNUTYPE_LONGNAME:
   assert 0<info.size<=4096;pending_name=get_range(offset+512,offset+512+info.size-1,'longname').rstrip(b'\0').decode()
  elif Path(name).name==target:
   assert info.size==430187;blob=get_range(offset+512,offset+512+info.size-1,'native-gene-member');assert blob[:2]==b'\x1f\x8b';raw=B/target;raw.open('xb').write(blob);plain=gzip.decompress(blob);plain.decode('utf-8');out=B/'gene.annotations.txt';out.open('xb').write(plain);found=dict(raw=dict(path=str(raw),bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest()),decoded=dict(path=str(out),bytes=len(plain),sha256=hashlib.sha256(plain).hexdigest()),header=members[-1]);break
  offset+=512+((info.size+511)//512)*512
 assert found is not None,'Targetnotfoundwithin64nativeheaders;noarchivebulkdownloadperformed'
 result=dict(schema='mora3.OfficialGEOTARBoundedMemberAcquisition/v1',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='complete_single_native_member_exact_ranges_headerchecksum_and_gzipCRC_verified',archive_url=url,archive_advertised_bytes=archive_total,archive_last_modified=lastmod,archive_full_acquired=False,actual_network_body_bytes=fetched,range_receipts=ledger,visited_native_headers=members,gene_member=found,new_corpus_tasks=0,physical_dose_mapping_verified=False,source_and_failed_attempts_immutable=True)
 (B/'manifest.json').open('x').write(json.dumps(result,indent=2)+'\n');print('COMPLETE',fetched,len(members),found['decoded']['bytes'])
except BaseException as e:
 (B/'failure.json').open('x').write(json.dumps(dict(error=repr(e),actual_network_body_bytes=fetched,range_receipts=ledger),indent=2)+'\n');raise
