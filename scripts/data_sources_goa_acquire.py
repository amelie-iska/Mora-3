"""Finite, one-connection GOA acquisition through the pinned QuickGO skill client."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import time
from data_sources_goa import acquire_batch,check_snapshot,query_params,retry_delay

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def write_new(path,data):
    with path.open('x') as f:json.dump(data,f,indent=2);f.write('\n')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

class SkillClient:
    def __init__(self,root,plan):
        self.root=root;self.plan=plan;self.receipts=[];self.batch_dir=None;self.last_end=0;self.request_count=0
        skill=root/'research/skill-runtime/skills/quickgo-skill/scripts/rest_request.py'
        spec=importlib.util.spec_from_file_location('pinned_quickgo_rest_request',skill);self.skill=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.skill)
        self.last_http={};outer=self;original=self.skill.requests.Session
        class PublicSession(original):
            def __init__(self):super().__init__();self.trust_env=False
            def request(self,*args,**kwargs):
                response=super().request(*args,**kwargs)
                outer.last_http=dict(status=response.status_code,headers={k:v for k,v in response.headers.items() if k.lower() in ['date','etag','last-modified','content-length','retry-after','content-type']})
                return response
        self.skill.requests.Session=PublicSession
    def raw_bytes(self):return sum(p.stat().st_size for p in (self.root/'raw').rglob('*') if p.is_file())
    def request(self,path,params,name,accept='application/json'):
        if self.batch_dir is None:raise ValueError('batch directory required')
        for attempt in range(3):
            if shutil.disk_usage(self.root).free<self.plan['minimum_free_bytes']+self.plan['derived_cap_bytes']:raise ValueError('free-space reserve gate')
            if self.raw_bytes()>=self.plan['raw_cap_bytes']-8*1024**2:raise ValueError('raw budget reserve gate')
            time.sleep(max(0,1-(time.monotonic()-self.last_end)))
            suffix='.json' if accept=='application/json' else ''
            output=self.batch_dir/(name+suffix+('.attempt%d'%attempt if attempt else ''))
            if output.exists():raise FileExistsError('original source payload already exists; no overwrite')
            query=dict(base_url='https://www.ebi.ac.uk/QuickGO/services',path=path,params=params,headers={'Accept':accept},response_format='json' if accept=='application/json' else 'text',record_path='results' if path=='annotation/search' else None,max_items=1,timeout_sec=60,save_raw=True,raw_output_path=str(output))
            self.last_http={};result=self.skill.execute(query);self.last_end=time.monotonic();self.request_count+=1
            receipt=dict(retrieved_utc=now(),query=query,skill_result=result,http=self.last_http,source_response_path=str(output),source_bytes=output.stat().st_size if output.exists() else None,source_sha256=sha(output) if output.exists() else None)
            receipt_path=self.batch_dir/(name+'.request-%d.json'%attempt);write_new(receipt_path,receipt);self.receipts.append(str(receipt_path))
            if result.get('ok'):
                if not output.exists():raise ValueError('skill did not preserve exact raw response')
                if self.raw_bytes()>self.plan['raw_cap_bytes']:raise ValueError('received source exceedsrawbudget; retainedquarantine')
                return json.loads(output.read_text()) if accept=='application/json' else output.read_text()
            status=self.last_http.get('status');delay=retry_delay(status,attempt,next((v for k,v in self.last_http.get('headers',{}).items() if k.lower()=='retry-after'),None));time.sleep(delay)
        raise ValueError('finite request attempts exhausted')

def run(plan_path,lease_path):
    plan_path=pathlib.Path(plan_path).resolve();root=plan_path.parent;plan=json.loads(plan_path.read_text());lease=json.loads(pathlib.Path(lease_path).read_text())
    if not lease.get('one_source_connection_reserved') or lease.get('job_id')!=plan['job_id']:raise ValueError('explicit finite connection lease required')
    if plan.get('workers')!=1 or plan.get('requests_per_second_max')!=1:raise ValueError('unreviewed API concurrency')
    accessions=[x for batch in plan['batches'] for x in batch]
    if accessions!=plan['accessions'] or len(accessions)!=len(set(accessions)) or len(accessions)!=plan['canonical_scope_count']:raise ValueError('scope overlap or mismatch')
    for batch in plan['batches']:query_params(batch,1)
    scope=json.loads((root/'audits/remaining-canonical-scope-v1.json').read_text())
    if not set(accessions)<=set(scope['accessions']):raise ValueError('scope outside independently qualified baseline')
    lock=root/('lease-'+plan['job_id']+'.json');write_new(lock,dict(pid=os.getpid(),started_utc=now(),plan=str(plan_path),connection_lease=str(pathlib.Path(lease_path).resolve())))
    out=root/'raw'/plan['job_id'];out.mkdir(exist_ok=False);client=SkillClient(root,plan);manifests=[];started=time.monotonic();live=root/('live-acquisition-'+plan['job_id']+'.json')
    def status(phase,active=None,error=None):
        data=dict(updated_utc=now(),pid=os.getpid(),phase=phase,active_batch=active,completed_batches=len(manifests),expected_batches=len(plan['batches']),raw_bytes=client.raw_bytes(),requests=client.request_count,elapsed_seconds=time.monotonic()-started,manifest_paths=manifests,error=error,connection_count=1,source_snapshot=plan['snapshot'],connection_lease=lease)
        live.write_text(json.dumps(data,indent=2)+'\n')
    status('live finite exact-accession GOA acquisition')
    try:
        for i,batch in enumerate(plan['batches']):
            directory=out/('batch-%04d'%i);directory.mkdir(exist_ok=False);client.batch_dir=directory;status('live batch source validation',i)
            def inventory(d):
                if d['expected_rows']>plan['max_batch_annotations']:raise ValueError('batch volume exceedsreviewedRAMbound; split futureplan')
                d.update(canonical_accessions=batch,source_snapshot=plan['snapshot'],registered_utc=now());write_new(directory/'planned-source-assets.json',d)
            result=acquire_batch(batch,client.request,plan['snapshot'],inventory);result.update(source_raw_directory=str(directory),receipt_paths=[x for x in client.receipts if str(directory) in x],all_raw_payloads=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in directory.iterdir() if p.is_file() and '.request-' not in p.name and p.name!='planned-source-assets.json'],baseline=plan['baseline'],source_release_separate_from_UniProt=True,reader_support='native source only; corpus typed projection/admission owned by dataset_construction',license='CC-BY-4.0',completed_utc=now())
            manifest=root/'curated'/('manifest-'+plan['job_id']+'-batch-%04d.json'%i);write_new(manifest,result);manifests.append(str(manifest));status('batch native QC completed',i);print('GOA_BATCH_COMPLETE',i,result['available_annotations'],flush=True)
        handoff=dict(schema='mora3.goa_finite_handoff.v1',job_id=plan['job_id'],completed_utc=now(),source_snapshot=plan['snapshot'],baseline=plan['baseline'],canonical_scope_count=len(accessions),full107778scope=plan.get('full_baseline',False),batch_manifests=manifests,available_annotations=sum(json.loads(pathlib.Path(p).read_text())['available_annotations'] for p in manifests),quality_pass_annotations=sum(json.loads(pathlib.Path(p).read_text())['quality_pass_annotations'] for p in manifests),task_admitted_annotations=0,license='CC-BY-4.0',source_review=plan['source_review'],reader_support=False,admission_owner='dataset_construction',all_explicit_source_forms_retained=True,reused_sample_manifest=plan['reused_sample_manifest'])
        write_new(root/('handoff-manifest-'+plan['job_id']+'.json'),handoff);status('finite source acquisition and native QC completed');print('GOA_FINISHED',len(accessions),handoff['available_annotations'],flush=True)
    except Exception as exc:
        status('stopped; affected batch retained quarantined',error=type(exc).__name__+':'+str(exc));raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--connection-lease',required=True);args=parser.parse_args();run(args.plan,args.connection_lease)
