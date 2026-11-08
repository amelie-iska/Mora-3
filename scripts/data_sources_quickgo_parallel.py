"""Finite 6+2 QuickGO dispatcher. Requires reviewed code, frozen checkpoint and retirement.
No signals, no live-helper changes; only new namespace writes and public requests.
"""
import argparse,datetime,json,multiprocessing,os,pathlib,shutil,time
from data_sources_quickgo_resume import SharedLimits,sha,certify_receipt,conservation
from data_sources_quickgo_client import CachedSkillClient
from data_sources_goa_v2 import acquire_batch
from data_sources_go_ontology import check_version,validate_page,project_term

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write_new(p,j):
    with p.open('x') as f:json.dump(j,f,indent=2);f.write('\n')
def status(path,**data):path.write_text(json.dumps(dict(updated_utc=now(),pid=os.getpid(),**data),indent=2)+'\n')
def verify_review(approval_path,checkpoint_path,retirement_path):
    a=json.loads(pathlib.Path(approval_path).read_text());c=json.loads(pathlib.Path(checkpoint_path).read_text());r=json.loads(pathlib.Path(retirement_path).read_text())
    if not a.get('approved') or a.get('action')!='QuickGO_checkpoint_parallel_handoff_v1':raise ValueError('exact reviewed execution receipt required')
    required=['scripts/data_sources_quickgo_resume.py','scripts/data_sources_quickgo_client.py','scripts/data_sources_quickgo_checkpoint.py','scripts/data_sources_quickgo_parallel.py']
    for p in required:
        if a.get('reviewed_code_sha256',{}).get(p)!=sha(p):raise ValueError('reviewed helper changed or missing approvalhash')
    if not r.get('all_owned_old_writers_retired') or r.get('checkpoint_sha256')!=sha(checkpoint_path):raise ValueError('old writers not retired for exactcheckpoint')
    for old in c['owned_processes']:
        p=pathlib.Path('/proc/'+str(old['pid']))
        if p.exists():
            parts=(p/'stat').read_text().split()
            if int(parts[21])==old['start_ticks'] and parts[2]!='Z':raise ValueError('old owned source writer remains')
    return c

def worker_plan(job,category):
    return dict(job_id=job,minimum_free_bytes=200*1024**3,derived_cap_bytes=8*1024**3,raw_cap_bytes=(4*1024**3 if category=='annotation' else 512*1024**2))
def annotation_worker(cp,number,limits,run):
    b=pathlib.Path(cp['root']);job=run+'-annotation-%d'%number;out=b/'raw'/job;out.mkdir(exist_ok=False);tasks=cp['new_annotation_leases'][number];live=b/('live-acquisition-'+job+'.json');client=CachedSkillClient(b,worker_plan(job,'annotation'),limits,'annotation');results=[];started=time.monotonic()
    try:
        for i,t in enumerate(tasks):
            d=out/('batch-%04d'%i);d.mkdir();client.start_batch(d,t['cache_certificates']);status(live,phase='live strict native annotation acquisition/replay',active_batch=i,completed_batches=len(results),expected_batches=len(tasks),source_connections=1,HTTP_requests=client.request_count,reused_requests=client.cached_requests)
            def inventory(j):
                if j['expected_rows']>250000:raise ValueError('source batch exceeds reviewed250k nativeRAMbound')
                write_new(d/'planned-source-assets.json',dict(j,canonical_accessions=t['canonical_accessions'],source_snapshot=cp['annotation_snapshot'],old_job=t['old_job'],old_batch=t['old_batch'],registered_utc=now()))
            r=acquire_batch(t['canonical_accessions'],client.request,cp['annotation_snapshot'],inventory)
            refs={x['path']:dict(path=x['path'],bytes=x['bytes'],sha256=x['sha256']) for x in client.payload_refs}
            r.update(source_raw_directory=str(d),receipt_paths=[x for x in client.receipts if str(d) in x],all_raw_payloads=list(refs.values()),certified_reused_or_new_sources=client.payload_refs.copy(),baseline=dict(path=cp['qualified_baseline_path'],sha256=cp['qualified_baseline_sha256'],release='2026_03',entries=107778),source_release_separate_from_UniProt=True,reader_support=False,license='CC-BY-4.0',completed_utc=now(),original_query_lease=dict(old_job=t['old_job'],old_batch=t['old_batch'],original_part_c_batch=t['original_part_c_batch']),shared_raw_budget=True)
            p=b/'curated'/('manifest-'+job+'-batch-%04d.json'%i);write_new(p,r);results.append(dict(path=str(p),sha256=sha(p),annotations=r['available_annotations']));client.raw_bytes();status(live,phase='batch native semantic QC complete',active_batch=i,completed_batches=len(results),expected_batches=len(tasks),source_connections=1,HTTP_requests=client.request_count,reused_requests=client.cached_requests,raw_bytes=client.raw_bytes(),elapsed_seconds=time.monotonic()-started)
        write_new(b/('handoff-manifest-'+job+'.json'),dict(job_id=job,source_snapshot=cp['annotation_snapshot'],completed_utc=now(),batch_manifests=results,query_canonical_accessions=[x for t in tasks for x in t['canonical_accessions']],task_admitted_annotations=0,source_connections=0));status(live,phase='finite worker complete',completed_batches=len(results),expected_batches=len(tasks),source_connections=0)
    except Exception as e:
        if 'snapshot' in str(e).lower():limits.block('annotation snapshot drift')
        status(live,phase='stopped; affected batch source-present/quarantined',completed_batches=len(results),expected_batches=len(tasks),source_connections=0,error=type(e).__name__+':'+str(e));raise

def ontology_worker(cp,number,limits,run):
    b=pathlib.Path(cp['root']);job=run+'-ontology-%d'%number;out=b/'raw'/job;out.mkdir(exist_ok=False);pages=cp['new_ontology_page_leases'][number];live=b/('live-acquisition-'+job+'.json');client=CachedSkillClient(b,worker_plan(job,'ontology'),limits,'ontology');sources=[]
    try:
        for i,page in enumerate(pages):
            d=out/('group-%04d'%(i//50));d.mkdir(exist_ok=True);client.start_batch(d)
            if i%50==0:check_version(client.request('ontology/go/about',{},'version-before-%04d'%page),cp['ontology_timestamp'],cp['ontology_version'])
            r=client.request('ontology/go/terms',dict(page=page),'page-%04d'%page);validate_page(r,page,48392);sources.append(dict(page=page,source=client.payload_refs[-1]));client.raw_bytes();status(live,phase='native ontology page QC complete',completed_pages=len(sources),expected_pages=len(pages),active_source_page=page,source_connections=1,HTTP_requests=client.request_count)
        d=out/'version-after';d.mkdir();client.start_batch(d);check_version(client.request('ontology/go/about',{},'version-after'),cp['ontology_timestamp'],cp['ontology_version']);client.raw_bytes();write_new(b/('handoff-manifest-'+job+'.json'),dict(job_id=job,completed_utc=now(),source_pages=sources,ontology_timestamp=cp['ontology_timestamp'],ontology_version=cp['ontology_version'],source_connections=0));status(live,phase='finite worker complete',completed_pages=len(sources),expected_pages=len(pages),source_connections=0)
    except Exception as e:
        if 'snapshot' in str(e).lower() or 'version' in str(e).lower():limits.block('ontology snapshot drift')
        status(live,phase='stopped; original page bytes retained quarantined',completed_pages=len(sources),expected_pages=len(pages),source_connections=0,error=type(e).__name__+':'+str(e));raise

def version_probe(cp,limits,run,name):
    b=pathlib.Path(cp['root']);d=b/'raw'/(run+'-'+name);d.mkdir(exist_ok=False);client=CachedSkillClient(b,worker_plan(run+'-'+name,'annotation'),limits,'annotation');client.start_batch(d)
    a=client.request('annotation/about',{},'annotation-about');o=client.request('ontology/go/about',{},'ontology-about')
    if a.get('annotation',{}).get('timestamp')!=cp['annotation_snapshot']:raise ValueError('annotation snapshot drift')
    check_version(o,cp['ontology_timestamp'],cp['ontology_version']);write_new(b/'audits'/(run+'-'+name+'.json'),dict(annotation=a,ontology=o,source_certificates=client.payload_refs,independent_snapshots=True))

def publish_ontology(cp,run):
    import pyarrow as pa,pyarrow.parquet as pq
    b=pathlib.Path(cp['root']);out=b/'curated'/(run+'-ontology');out.mkdir(exist_ok=False);pages=cp['ontology_cached_pages'].copy()
    for i in range(2):pages+=json.loads((b/('handoff-manifest-'+run+'-ontology-%d.json'%i)).read_text())['source_pages']
    if len(pages)!=1936 or {r['page'] for r in pages}!=set(range(1,1937)):raise ValueError('missing/duplicate native ontology page lease')
    seen=set();counts=dict(native_terms=0,obsolete_terms=0,missing_aspect=0,current_child_edges=0,native_history_entries=0);nodes=[];edges=[];refs=[];derived=0
    ns=pa.schema([('go_id',pa.string()),('name',pa.string()),('aspect',pa.string()),('is_obsolete',pa.bool_()),('ontology_version',pa.string()),('source_raw_path',pa.string()),('source_result_index',pa.int64()),('native_term_sha256',pa.string()),('native_term_json',pa.string())]);es=pa.schema([('child_id',pa.string()),('relation',pa.string()),('parent_id',pa.string()),('ontology_version',pa.string()),('source_raw_path',pa.string()),('source_result_index',pa.int64())])
    def flush():
        nonlocal derived
        if not nodes:return
        for kind,rows,schema in [('nodes',nodes,ns),('edges',edges,es)]:
            p=out/('%s-%04d.parquet'%(kind,len(refs)//2));pq.write_table(pa.Table.from_pylist(rows,schema=schema),p,compression='zstd')
            if pq.read_table(p).to_pylist()!=rows:raise ValueError('native ontology roundtrip')
            derived+=p.stat().st_size;refs.append(dict(kind=kind,path=str(p),rows=len(rows),bytes=p.stat().st_size,sha256=sha(p)))
        nodes.clear();edges.clear()
    for p in sorted(pages,key=lambda r:r['page']):
        c=p['source'];raw=pathlib.Path(c['path'])
        if raw.stat().st_size!=c['bytes'] or sha(raw)!=c['sha256']:raise ValueError('old/new source ontology bytes changed')
        j=json.loads(raw.read_text());validate_page(j,p['page'],48392)
        for i,r in enumerate(j['results']):
            if r['id'] in seen:raise ValueError('duplicate ontology ID across exactpageunion')
            seen.add(r['id']);n,e=project_term(r,str(raw),i,cp['ontology_version']);nodes.append(n);edges+=e;counts['native_terms']+=1;counts['obsolete_terms']+=r['isObsolete'];counts['missing_aspect']+=r.get('aspect') is None;counts['current_child_edges']+=len(e);counts['native_history_entries']+=len(r.get('history') or [])
        if len(nodes)>=5000:flush()
    flush()
    if len(seen)!=48392 or derived>512*1024**2:raise ValueError('fullontology count/derivedcap')
    write_new(out/'manifest.json',dict(schema='mora3.GO_native_ontology_complete_reused_source.v1',completed_utc=now(),ontology_timestamp=cp['ontology_timestamp'],ontology_version=cp['ontology_version'],annotation_snapshot=cp['annotation_snapshot'],same_snapshot=False,counts=counts,source_pages=pages,typed_payloads=refs,derived_bytes=derived,all_native_fields_in_native_term_JSON=True,history_not_current_edges=True,annotation_goName_filled=False,task_admitted_rows=0,reader_support=False,license='CC-BY-4.0'))
    return str(out/'manifest.json')

def run(checkpoint,retirement,approval,run_id):
    cp=verify_review(approval,checkpoint,retirement);b=pathlib.Path(cp['root']);run=run_id
    if not run.startswith('quickgo-parallel-') or '/' in run:raise ValueError('new explicitnamespace required')
    # Byte-preserving handoff readback before any network worker exists.
    for r in cp['old_raw_hashes']:
        p=pathlib.Path(r['path'])
        if p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise ValueError('protected oldsource changed at handoff')
    ctx=multiprocessing.get_context('fork');limits=SharedLimits(ctx,3,{'annotation':(4*1024**3,cp['initial_shared_raw_used']['annotation']),'ontology':(512*1024**2,cp['initial_shared_raw_used']['ontology'])});live=b/('live-'+run+'.json');started=time.monotonic();lease=b/('lease-'+run+'.json');write_new(lease,dict(pid=os.getpid(),started_utc=now(),checkpoint=str(pathlib.Path(checkpoint).resolve()),checkpoint_sha256=sha(checkpoint),retirement=retirement,approval=approval,annotation_workers=6,ontology_workers=2,max_shared_request_starts_per_second=3,raw_caps_shared_with_all_preserved_originalbytes=True))
    p=ctx.Process(target=version_probe,args=(cp,limits,run,'versions-before'));p.start();p.join()
    if p.exitcode:raise ValueError('beforeversionsprobe failed')
    children=[]
    for kind,n,target in [('annotation',6,annotation_worker),('ontology',2,ontology_worker)]:
        for i in range(n):
            p=ctx.Process(target=target,args=(cp,i,limits,run));p.start();children.append(dict(kind=kind,index=i,process=p))
    ontology_manifest=None
    while any(x['process'].is_alive() for x in children):
        for x in children:x['process'].join(timeout=0)
        ont=[x['process'] for x in children if x['kind']=='ontology']
        if ontology_manifest is None and all(p.exitcode==0 for p in ont):ontology_manifest=publish_ontology(cp,run)
        status(live,phase='finite public-source workers advancing',workers=[dict(kind=x['kind'],index=x['index'],pid=x['process'].pid,alive=x['process'].is_alive(),exit_code=x['process'].exitcode) for x in children],shared_limits=limits.snapshot(),elapsed_seconds=time.monotonic()-started,ontology_manifest=ontology_manifest)
        time.sleep(5)
    for x in children:x['process'].join()
    failures=[dict(kind=x['kind'],index=x['index'],exit_code=x['process'].exitcode) for x in children if x['process'].exitcode]
    if failures:status(live,phase='finite dispatcher ended with preserved source failures',failures=failures,shared_limits=limits.snapshot());raise ValueError('finite source worker failure; successful manifests retained')
    if ontology_manifest is None:ontology_manifest=publish_ontology(cp,run)
    p=ctx.Process(target=version_probe,args=(cp,limits,run,'versions-after'));p.start();p.join()
    if p.exitcode:raise ValueError('afterversionsprobe failed')
    refs=cp['annotation_previous_completed_manifests'].copy();groups=cp['annotation_previous_completed_groups'].copy()
    for i in range(6):
        h=json.loads((b/('handoff-manifest-'+run+'-annotation-%d.json'%i)).read_text());refs+=h['batch_manifests']
        for r in h['batch_manifests']:groups.append(json.loads(pathlib.Path(r['path']).read_text())['query_canonical_accessions'])
    proof=conservation(groups,[],cp['qualified_baseline_IDs']);sample=json.loads((b/'handoff-manifest-sample-v1.json').read_text());total=sum(r['annotations'] for r in refs)+sample['available_annotations']
    handoff=dict(schema='mora3.GOA_full_qualified_canonical_query_subset.v1',completed_utc=now(),checkpoint=checkpoint,source_snapshot=cp['annotation_snapshot'],qualified_baseline_release='2026_03',canonical_query_count=107778,query_conservation=proof,complete_native_batch_manifests=refs,reused_P05067_sample=str(b/'handoff-manifest-sample-v1.json'),available_native_annotation_occurrences=total,quality_pass_native_annotation_occurrences=total,task_admitted_annotations=0,ontology_companion_manifest=ontology_manifest,ontology_snapshot_separate=True,all_forms_NOT_ECO_unknown_fields_AND_OR_extensions_retained=True,all_original_C_prefix_and_GPAD_exception_proofs_retained=True,allpossibleisoformfiltercoverage_not_asserted=True,license='CC-BY-4.0',reader_support=False,corpus_admission_owner='dataset_construction',shared_limits=limits.snapshot())
    write_new(b/('handoff-manifest-'+run+'.json'),handoff);status(live,phase='full finite canonicalquerysubset andontology native QC complete',handoff=str(b/('handoff-manifest-'+run+'.json')),shared_limits=limits.snapshot(),elapsed_seconds=time.monotonic()-started)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--retirement',required=True);p.add_argument('--approval',required=True);p.add_argument('--run-id',required=True);a=p.parse_args();run(a.checkpoint,a.retirement,a.approval,a.run_id)
