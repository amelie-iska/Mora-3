"""Build an immutable QuickGO checkpoint ONLY after owned workers are frozen/exited.
This helper sends no signals, starts no requests, and changes no existing files.
"""
import argparse,collections,csv,datetime,gzip,hashlib,json,os,pathlib
from data_sources_quickgo_resume import sha,certify_receipt,conservation
from data_sources_go_ontology import validate_page

def write_new(p,j):
    with p.open('x') as f:json.dump(j,f,indent=2);f.write('\n')
def inspect_owned(expected):
    p=pathlib.Path('/proc/'+str(expected['pid']))
    if not p.exists():return dict(expected,state='exited_before_checkpoint')
    s=(p/'stat').read_text().split();cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip()
    if int(s[21])!=expected['start_ticks'] or cmd!=expected['command'] or os.getpgid(expected['pid'])!=expected['pid']:raise ValueError('ownedprocessidentitychanged; no signal permitted')
    if s[2] not in ['T','t','Z']:raise ValueError('old source writer not frozen/exited; no live checkpoint')
    return dict(expected,state=s[2],process_group=os.getpgid(expected['pid']))
def batch_cache(directory,snapshot):
    refs=[];quarantine=[]
    if not directory.exists():return refs,quarantine
    before=directory/'snapshot-before.request-0.json'
    try:
        c=certify_receipt(before);j=json.loads(pathlib.Path(c['path']).read_text())
        if j.get('annotation',{}).get('timestamp')!=snapshot:raise ValueError('priorbatchsnapshotmismatch')
    except Exception as e:return [],[dict(path=str(directory),reason='no certified matching source snapshot: '+str(e))]
    for p in sorted(directory.glob('*.request-*.json')):
        try:refs.append(certify_receipt(p))
        except Exception as e:quarantine.append(dict(path=str(p),reason=str(e)))
    return refs,quarantine

def build(proposal_path,out):
    proposal=json.loads(pathlib.Path(proposal_path).read_text());b=pathlib.Path(proposal_path).resolve().parents[1];snap=json.loads(pathlib.Path(proposal['nonoperative_scope_snapshot']['path']).read_text());out=pathlib.Path(out).resolve();out.mkdir(exist_ok=False)
    expected=[dict(pid=j['pid'],start_ticks=j['start_ticks'],command=j['command'],job_id=j['job_id']) for j in snap['old_annotation_jobs']];o=snap['ontology'];expected.append(dict(pid=o['pid'],start_ticks=o['start_ticks'],command=o['command'],job_id='ontology-native-v1'));states=[inspect_owned(x) for x in expected]
    # Every completed query appears exactly once, including original C27 prefix,
    # its independently GPAD-validated failed batch27 and the pilot100.
    completed=[];completed_groups=[['P05067']]
    for p in sorted((b/'curated').glob('*.json')):
        m=json.loads(p.read_text())
        if 'query_canonical_accessions' in m and m.get('quality_pass_annotations')==m.get('available_annotations') and m.get('full_batch_scan'):
            completed.append(dict(path=str(p.resolve()),sha256=sha(p),annotations=m['available_annotations']));completed_groups.append(m['query_canonical_accessions'])
    tasks=[];quarantines=[]
    for job in snap['old_annotation_jobs']:
        pp=pathlib.Path(job['plan'])
        if sha(pp)!=job['plan_sha256']:raise ValueError('immutable oldplanchanged')
        p=json.loads(pp.read_text())
        for i,ids in enumerate(p['batches']):
            m=b/'curated'/('manifest-'+p['job_id']+'-batch-%04d.json'%i)
            if m.exists():continue
            directory=b/'raw'/p['job_id']/('batch-%04d'%i);certs,q=batch_cache(directory,'2026-07-28 10:08');quarantines+=q;tasks.append(dict(old_job=p['job_id'],old_batch=i,original_part_c_batch=p.get('original_batch_indices',[None]*len(p['batches']))[i],canonical_accessions=ids,cache_certificates=certs,old_raw_directory=str(directory)))
    baseline=pathlib.Path('outputs/data-intake/20261031-subagent/raw/uniprot2026_03/reviewed-annotation4-5-sequences.tsv.gz').resolve()
    if sha(baseline)!='e9bd53c005179cc68664f920f833273b241afb7196806ca512866869674d5133':raise ValueError('qualified baseline changed')
    with gzip.open(baseline,'rt') as f:baseline_ids={r['Entry'] for r in csv.DictReader(f,delimiter='\t')}
    if len(baseline_ids)!=107778:raise ValueError('baselinecount')
    proof=conservation(completed_groups,tasks,baseline_ids)
    op=json.loads((b/'acquisition-plan-ontology-native-v1.json').read_text());pages=[];pending=[];seen=set()
    for page in range(1,op['pages']+1):
        path=pathlib.Path(op['reuse_first_page']['path']) if page==1 else b/'raw/ontology-native-v1'/('group-%04d'%((page-1)//50))/('page-%04d.json'%page)
        try:
            if page==1:
                if sha(path)!=op['reuse_first_page']['sha256']:raise ValueError('first source page changed')
                cert=dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size,query=dict(path='ontology/go/terms',params={'page':1}))
            else:cert=certify_receipt(path.parent/('page-%04d.request-0.json'%page))
            if cert['query'].get('path')!='ontology/go/terms' or cert['query'].get('params')!={'page':page}:raise ValueError('wrongcachedontologypagequery')
            j=json.loads(path.read_text());validate_page(j,page,48392);ids={r['id'] for r in j['results']}
            if seen&ids:raise ValueError('duplicateontologyIDacrosscachedpages')
            seen|=ids;pages.append(dict(page=page,source=cert))
        except (FileNotFoundError,ValueError,AssertionError) as e:
            pending.append(page)
            if path.exists():quarantines.append(dict(path=str(path),reason=str(e),available_but_not_cache_eligible=True))
    # Freeze-safe whole original source-file fsync/hash. Mutable live-state copies
    # are archived separately and never mistaken for reusable response evidence.
    raw_hashes=[]
    for name in [x['job_id'] for x in states]:
        d=b/'raw'/name
        for p in sorted(d.rglob('*')):
            if p.is_file():
                with p.open('rb') as f:os.fsync(f.fileno())
                raw_hashes.append(dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p)))
        for prefix in ['live-acquisition-','live-request-progress-','live-']:
            p=b/(prefix+name+'.json')
            if p.exists():
                dest=out/p.name
                if not dest.exists():
                    with dest.open('xb') as f:f.write(p.read_bytes());f.flush();os.fsync(f.fileno())
    used_annotation=used_ontology=0
    for p in (b/'raw').rglob('*'):
        if p.is_file():
            if 'ontology-native-v1' in p.parts:used_ontology+=p.stat().st_size
            else:used_annotation+=p.stat().st_size
    used_ontology+=op['reuse_first_page']['bytes']
    for x in states:inspect_owned(x)
    checkpoint=dict(schema='mora3.QuickGO_frozen_checkpoint.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),root=str(b),proposal=str(pathlib.Path(proposal_path).resolve()),proposal_sha256=sha(proposal_path),owned_processes=states,annotation_snapshot='2026-07-28 10:08',ontology_timestamp=op['ontology_timestamp'],ontology_version=op['ontology_version'],annotation_previous_completed_manifests=completed,annotation_previous_completed_groups=completed_groups,remaining_annotation_tasks=tasks,annotation_conservation=proof,qualified_baseline_IDs=sorted(baseline_ids),qualified_baseline_path=str(baseline),qualified_baseline_sha256=sha(baseline),ontology_cached_pages=pages,ontology_remaining_pages=pending,ontology_expected_terms=48392,ontology_expected_pages=1936,old_raw_hashes=raw_hashes,unverified_response_quarantines=quarantines,initial_shared_raw_used=dict(annotation=used_annotation,ontology=used_ontology),source_bytes_preserved=True,no_requests_or_signals_by_builder=True,new_annotation_leases=[[t for i,t in enumerate(tasks) if i%6==k] for k in range(6)],new_ontology_page_leases=[[p for i,p in enumerate(pending) if i%2==k] for k in range(2)])
    write_new(out/'checkpoint.json',checkpoint);return checkpoint
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--proposal',required=True);p.add_argument('--output',required=True);a=p.parse_args();c=build(a.proposal,a.output);print('FROZEN_CHECKPOINT',c['annotation_conservation'],len(c['ontology_cached_pages']),len(c['ontology_remaining_pages']))
