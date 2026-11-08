"""Finite bounded official original count-assets intake for independent verification only."""
from pathlib import Path
import datetime,gzip,hashlib,json,os,subprocess,time
R=Path.cwd();B=Path('outputs/data-intake/20261101-sciplex-original-count-assets-v1');B.mkdir(exist_ok=False);(B/'raw').mkdir();(B/'logs').mkdir();start=time.perf_counter();CAP=512*2**20
P=Path('outputs/data-intake/20261101-sciplex-lineage-resolution-v1/range-preflight-v1/manifest.json');probe=json.loads(P.read_text());assets=[]
for suffix in ['gene.annotations.txt.gz','UMI.count.matrix.gz']:
 e=next(x for x in probe['receipts'] if x['url'].endswith(suffix+'?download=1') and x.get('range_compatible') and x.get('gzip_magic'));total=int(e['headers']['Content-Range'].split('/')[-1]);assets.append(dict(url=e['url'],name=e['url'].split('/')[-1].split('?')[0],expected_bytes=total,last_modified=e['headers']['Last-Modified'],range_probe=e['prefix']))
def ref(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def live(stage,**kw):
 (B/'live-acquisition.json').write_text(json.dumps(dict(pid=os.getpid(),stage=stage,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.perf_counter()-start,**kw),indent=2)+'\n');print(stage,kw,round(time.perf_counter()-start,2),flush=True)
(B/'acquisition-plan.json').open('x').write(json.dumps(dict(schema='mora3.SciPlexOriginalCountsVerificationIntakePlan/v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source='officialpublicGEO_GSM4150377',purpose='Independentoriginal5000geneexpressionvalueverification;notnewtaskadmission',source_transport='Officialpublicdownloadqueryservedexact206ranges;canonicalmatrixURLreturned403;noaccounts/challenges/credentialsused',preflight=ref(P),assets=assets,raw_cap_bytes=CAP,minimum_free_bytes=200*2**30,source_connections=4,backend='existingaria2segmentedstaticHTTP;no-conf/no-netrc;finite2tries',source_rights_and_physicaldose_gates_unchanged=True,old_sources_and_failed_prefixes_immutable=True),indent=2)+'\n');receipts=[]
try:
 for e in assets:
  st=os.statvfs('.');assert st.f_bavail*st.f_frsize>=200*2**30;name=e['name'];argv=['/usr/bin/aria2c','--no-conf=true','--no-netrc=true','--enable-rpc=false','--allow-overwrite=false','--auto-file-renaming=false','--max-concurrent-downloads=1','--split=4','--max-connection-per-server=4','--min-split-size=4M','--max-tries=2','--retry-wait=3','--connect-timeout=15','--timeout=30','--file-allocation=none','--console-log-level=warn','--summary-interval=1','--dir='+str(B/'raw'),'--out='+name,e['url']];log=B/'logs'/(name+'.log');f=log.open('x');proc=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True);live('finiteoriginal_source_countasset_download',active_asset=name,download_pid=proc.pid,expected_bytes=e['expected_bytes'])
  try:code=proc.wait(timeout=900)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,__import__('signal').SIGTERM);code='finite_timeout'
  f.close();p=B/'raw'/name;receipt=dict(asset=e,exit_code=code,log=ref(log),status='failed_or_partial_preserved')
  if p.exists():receipt['raw']=ref(p)
  if code==0:
   assert p.stat().st_size==e['expected_bytes'];assert sum(p.stat().st_size for p in (B/'raw').iterdir() if p.is_file())<=CAP;assert p.read_bytes()[:1024]==Path(e['range_probe']['path']).read_bytes();n=0
   with gzip.open(p,'rb') as gz:
    for block in iter(lambda:gz.read(8*2**20),b''):n+=len(block);assert n<=8*2**30
   receipt.update(status='complete_size_prefix_SHA_and_gzipCRC_verified',decoded_bytes=n,independent_original_matrix_alignment_done=False,upstream_MD5_not_provided_in_selected_metadata=True)
  receipts.append(receipt);(B/'receipt-progress.json').write_text(json.dumps(receipts,indent=2)+'\n');live('original_asset_attempt_complete',asset=name,status=receipt['status'])
 result=dict(schema='mora3.SciPlexOriginalCountAssetIntake/v1',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan=ref(B/'acquisition-plan.json'),receipts=receipts,verified_assets=sum(x['status']=='complete_size_prefix_SHA_and_gzipCRC_verified' for x in receipts),new_tasks=0,source_model_or_legacy_data_modified=False,physical_dose_units_unchanged=False,seconds=time.perf_counter()-start)
 (B/'manifest.json').open('x').write(json.dumps(result,indent=2)+'\n');live('finite_originalcountassets_complete',verified_assets=result['verified_assets'])
except BaseException as exc:
 (B/'failure.json').open('x').write(json.dumps(dict(error=repr(exc),seconds=time.perf_counter()-start),indent=2)+'\n');raise
