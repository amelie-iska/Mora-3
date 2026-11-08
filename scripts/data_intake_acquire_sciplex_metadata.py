"""Finite public-original metadata intake; no existing partial files or training state touched."""
from pathlib import Path
import datetime,gzip,hashlib,json,os,subprocess,time
R=Path.cwd();B=Path('outputs/data-intake/20261101-sciplex-lineage-resolution-v1');B.mkdir(exist_ok=False);(B/'raw').mkdir();(B/'decoded').mkdir();(B/'logs').mkdir();start=time.perf_counter();CAP=16*2**20
base='https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4150nnn/GSM4150377/suppl/'
names=['GSM4150377_sciPlex2_pData.txt.gz','GSM4150377_sciPlex2_A549_Transcription_Modulators_cell.annotations.txt.gz','GSM4150377_sciPlex2_A549_Transcription_Modulators_gene.annotations.txt.gz','GSM4150377_sciPlex2_A549_Transcription_Modulators_hashTable.out.txt.gz','GSM4150377_sciPlex2_hashSampleSheet.txt.gz'];assets=[('GSE139944_family.soft.gz','https://ftp.ncbi.nlm.nih.gov/geo/series/GSE139nnn/GSE139944/soft/GSE139944_family.soft.gz')]+[(n,base+n) for n in names]
plan=dict(schema='mora3.SciPlexOriginalMetadataPlan/v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source='GSE139944/GSM4150377publicofficialFTPdownloads',source_primary_page='https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4150377',source_policy='https://www.ncbi.nlm.nih.gov/geo/info/disclaimer.html',research_only_rights_and_lineage_gates_unmodified=True,raw_cap_bytes=CAP,per_file_decoded_cap_bytes=32*2**20,minimum_free_bytes=200*2**30,source_connections=1,backend='/usr/bin/aria2c;no-conf;no-netrc;singleconnection_finite2tries',performance_rationale='smallgzipmetadataandsingleNCBIhost;notsegmentedbulktransfer',existing_partial_outputs_immutable='outputs/data-intake/20261031-sciplex-original-context-v1',assets=[dict(name=n,url=u) for n,u in assets],new_training_tasks=0,new_physical_dose_or_barcode_claim=False)
(B/'acquisition-plan.json').open('x').write(json.dumps(plan,indent=2)+'\n');receipts=[]
def ref(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def live(stage,**kw):
 (B/'live-acquisition.json').write_text(json.dumps(dict(pid=os.getpid(),stage=stage,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.perf_counter()-start,completed_assets=len(receipts),**kw),indent=2)+'\n');print(stage,kw,round(time.perf_counter()-start,2),flush=True)
try:
 for n,u in assets:
  st=os.statvfs('.');assert st.f_bavail*st.f_frsize>=200*2**30;argv=['/usr/bin/aria2c','--no-conf=true','--no-netrc=true','--enable-rpc=false','--allow-overwrite=false','--auto-file-renaming=false','--max-concurrent-downloads=1','--split=1','--max-connection-per-server=1','--min-split-size=1M','--max-tries=2','--retry-wait=3','--connect-timeout=15','--timeout=30','--file-allocation=none','--console-log-level=warn','--summary-interval=0','--dir='+str(B/'raw'),'--out='+n,u]
  log=B/'logs'/(n+'.log');f=log.open('x');proc=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True);live('finiteoriginalmetadata_download',active_asset=n,download_pid=proc.pid)
  try:code=proc.wait(timeout=90)
  except subprocess.TimeoutExpired:
   os.killpg(proc.pid,__import__('signal').SIGTERM);code='finite_timeout'
  f.close();p=B/'raw'/n;entry=dict(name=n,url=u,backend_argv=argv,exit_code=code,log=ref(log),status='download_failed_preserved')
  if p.exists():entry['raw']=ref(p)
  if code==0 and p.exists():
   assert sum(p.stat().st_size for p in (B/'raw').iterdir() if p.is_file())<=CAP
   try:
    with gzip.open(p,'rb') as g:data=g.read(32*2**20+1)
    assert len(data)<=32*2**20;data.decode('utf-8');decoded=B/'decoded'/n[:-3];decoded.open('xb').write(data);entry.update(status='gzipCRC_UTF8_complete_metadata_verified',decoded=ref(decoded),decoded_lines=data.count(b'\n'),training_admission=False)
   except Exception as e:entry.update(status='source_decode_integrity_failed_retained',decode_error=repr(e))
  receipts.append(entry);(B/'acquisition-receipts-progress.json').write_text(json.dumps(receipts,indent=2)+'\n');live('metadataasset_attempt_complete',asset=n,status=entry['status'])
 result=dict(schema='mora3.SciPlexOriginalMetadataAcquisition/v1',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan=ref(B/'acquisition-plan.json'),receipts=receipts,verified_assets=sum(r['status']=='gzipCRC_UTF8_complete_metadata_verified' for r in receipts),new_training_admissions=0,original_paired_lineage_or_physical_dose_gate_lifted=False,source_rights_gate_unchanged=True,existing_sources_partialfiles_model_jobs_untouched=True,MCP_servers_enabled=0,seconds=time.perf_counter()-start)
 (B/'manifest.json').open('x').write(json.dumps(result,indent=2)+'\n');live('finiteoriginalmetadata_acquisition_complete',verified_assets=result['verified_assets'])
except BaseException as exc:
 (B/'failure.json').open('x').write(json.dumps(dict(error=repr(exc),seconds=time.perf_counter()-start),indent=2)+'\n');raise
