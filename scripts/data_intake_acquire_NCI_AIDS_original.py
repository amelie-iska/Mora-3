"""Finite public-original NCI acquisition; no authentication or source rewrites."""
from pathlib import Path
import requests,hashlib,json,datetime,time,re,html,urllib.parse,subprocess,shutil,os
from html.parser import HTMLParser
R=Path('/home/iska/Documents/amelie/bio/Mora-3');O=R/'outputs/data-intake/20261101-NCI-AIDS-original-v1';CAP=64*2**20;O.mkdir();(O/'raw').mkdir();(O/'research').mkdir();start=time.perf_counter();session=requests.Session();session.trust_env=False;session.headers['User-Agent']='Mora3-public-dataset-intake/1.0'
PAGE='https://wiki.nci.nih.gov/spaces/NCIDTPdata/pages/158204006/AIDS%2BAntiviral%2BScreen%2BData';POLICY='https://www.cancer.gov/policies/copyright-reuse'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p,**kw):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p),**kw)
def status(stage,**kw):
 (O/'live-acquisition.json').write_text(json.dumps(dict(pid=os.getpid(),status=stage,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.perf_counter()-start,**kw),indent=2)+'\n');print(stage,kw,flush=True)
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):
  if tag=='a':self.links += [v for k,v in attrs if k=='href']
try:
 assert shutil.disk_usage(R).free>200*2**30+512*2**20
 refs=[];bodies={}
 for name,url in [('source-page.html',PAGE),('NCI-reuse-policy.html',POLICY)]:
  with session.get(url,timeout=(20,45)) as resp:resp.raise_for_status();body=resp.content;assert len(body)<2*2**20
  p=O/'research'/name;p.open('xb').write(body);refs.append(ref(p,url=url,public_HTTP_status=resp.status_code));bodies[name]=body.decode(errors='replace')
 parser=Links();parser.feed(bodies['source-page.html']);wanted=['aids_conc_may04.txt','aids_ec50_may04.txt','aids_ic50_may04.txt','aido99sd.bin'];assets=[]
 for name in wanted:
  matches=[urllib.parse.urljoin(PAGE,html.unescape(url)) for url in parser.links if urllib.parse.unquote(urllib.parse.urlsplit(url).path).split('/')[-1].lower()==name];assert matches,'required public asset link not found '+name;url=matches[0];assert urllib.parse.urlsplit(url).hostname=='wiki.nci.nih.gov';head=session.head(url,allow_redirects=True,timeout=(20,45));head.raise_for_status();assert not any(x in head.url.lower() for x in ['/login','/authenticate','saml']);length=int(head.headers['Content-Length']) if head.headers.get('Content-Length') else None
  probe=session.get(url,headers={'Range':'bytes=0-1023','Accept-Encoding':'identity'},stream=True,timeout=(20,45));probe.raise_for_status();data=probe.raw.read(2048);content_range=probe.headers.get('Content-Range');total=int(content_range.rsplit('/',1)[1]) if content_range and '/' in content_range else length;probe.close();assert total is not None and 0<total<CAP;assert data and not data.lstrip().lower().startswith((b'<!doctype html',b'<html'));prefix=O/'research'/(name+'.prefix');prefix.open('xb').write(data);assets.append(dict(id='NCI-AIDS-original-'+name,filename=name,url=url,expected_bytes=total,HEAD_status=head.status_code,HEAD_headers={k:head.headers.get(k) for k in ['Content-Length','Content-Type','ETag','Last-Modified']},range_probe_HTTP_status=probe.status_code,range_probe_content_range=content_range,prefix=ref(prefix),parallel_ranges_eligible=probe.status_code==206 and content_range is not None,release='May2004' if name.endswith('.txt') else 'October1999',source_role='sourceclass_and_assay_endpoint_averages' if name.endswith('.txt') else 'chemicalconnectiontables;Corina3Dcomputedwithoutstereochemistry_notobservedgeometry'))
 assert sum(a['expected_bytes'] for a in assets)+sum(r['bytes'] for r in refs)<CAP
 (O/'acquisition-plan.json').open('x').write(json.dumps(dict(schema='mora3.NCIAIDSOriginalAcquisitionPlan/v1',source_pages=refs,assets=assets,raw_cap_bytes=CAP,derived_cap_bytes=512*2**20,minimum_free_bytes=200*2**30,conditions='publicoriginals;noauth;NSCidentityonly;nomatchedcell/exposureclaim;Corina3Dexcludedfromobservedgeometry',backend_policy='smalldatafilesboundedsequentialstream;BINsegmentedaria2onlyafterexact206probe'),indent=2)+'\n');status('originalpublicassets_preflight_complete',assets=len(assets),expected_bytes=sum(a['expected_bytes'] for a in assets))
 receipts=[]
 for a in assets:
  dest=O/'raw'/a['filename'];assert not dest.exists();status('downloading_original_asset',asset=a['filename'],expected_bytes=a['expected_bytes'])
  if a['expected_bytes']>=8*2**20 and a['parallel_ranges_eligible']:
   args=['/usr/bin/aria2c','--no-conf=true','--no-netrc=true','--allow-overwrite=false','--auto-file-renaming=false','--file-allocation=none','--max-tries=3','--retry-wait=2','--connect-timeout=15','--timeout=45','--split=4','--max-connection-per-server=4','--min-split-size=8M','--summary-interval=10','--console-log-level=warn','--dir='+str(dest.parent),'--out='+dest.name,a['url']];log=O/'research'/(a['filename']+'.aria2.log')
   with log.open('xb') as out:done=subprocess.run(args,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,timeout=900)
   assert done.returncode==0,'aria2 failed original asset'
   backend='aria2fourranges';extra=dict(backend_log=ref(log))
  else:
   with session.get(a['url'],stream=True,timeout=(20,60)) as resp:
    resp.raise_for_status();assert not any(x in resp.url.lower() for x in ['/login','/authenticate','saml']);size=0
    with dest.open('xb') as out:
     for chunk in resp.iter_content(1024*1024):
      size+=len(chunk);assert size<=a['expected_bytes'] and size<CAP;out.write(chunk)
   backend='boundedsequentialpublicHTTPstream';extra={}
  assert dest.stat().st_size==a['expected_bytes'];assert dest.read_bytes()[:min(1024,dest.stat().st_size)]==Path(R/a['prefix']['path']).read_bytes()[:min(1024,dest.stat().st_size)];receipts.append(dict(a,status='downloaded_exact_length_and_sourceprefix_selfSHA_pinned',raw=ref(dest),backend=backend,upstream_checksum_not_published=True,**extra));assert sum(r['raw']['bytes'] for r in receipts)<CAP;(O/'receipts-progress.json').write_text(json.dumps(receipts,indent=2)+'\n')
 (O/'code.py').write_bytes(Path(__file__).read_bytes());result=dict(schema='mora3.NCIAIDSOriginalAcquisition/v1',status='all_four_official_public_original_files_downloaded_exactlength_prefix_hashpinned',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_pages=refs,assets=receipts,raw_verified_bytes=sum(x['raw']['bytes'] for x in receipts),source_release_names=['May2004endpoint/classification','October1999chemicalstructures'],provider_NSC_count_claims=dict(classification=43850,protection_EC50=39365,uninfected_growth_IC50=39350,chemicalstructures=42390),counts_are_source_claims_not_parsed_yet=True,source_reuse_policy='NCIcopyrightfreeproductsunlessotherwiseindicated;creditNCI;publicnonconfidentialscreeningdata;nospecificCCSPDXinvented',geometry_evidence='Corina-computed3Dnostereochemistryoroptimization;notexperimental3Dtarget',model_adapter_supported=False,new_task_admissions=0,raw_cap_bytes=CAP,derived_cap_bytes=512*2**20,seconds=time.perf_counter()-start);(O/'manifest.json').open('x').write(json.dumps(result,indent=2)+'\n');status('completed_original_source_acquisition',bytes=result['raw_verified_bytes'])
except BaseException as exc:
 (O/'failure.json').open('x').write(json.dumps(dict(error=repr(exc),seconds=time.perf_counter()-start),indent=2)+'\n');status('failed_preserved_noautomaticretry',error=repr(exc));raise
