import importlib.util,pathlib,json,hashlib,multiprocessing,time
import pytest
ROOT=pathlib.Path.cwd()
def module():
    p=ROOT/'scripts/data_sources_quickgo_resume.py';assert p.is_file(), 'strict immutable cache and shared dispatch gates not implemented'
    s=importlib.util.spec_from_file_location('data_sources_quickgo_resume',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def fixture(tmp_path):
    payload=tmp_path/'page.json';payload.write_text('{"results":[],"numberOfHits":0}')
    q=dict(base_url='https://www.ebi.ac.uk/QuickGO/services',path='annotation/search',params=dict(geneProductId='UniProtKB:P05067',limit=200,page=1),headers={'Accept':'application/json'})
    receipt=tmp_path/'page.request-0.json';r=dict(query=q,http={'status':200},skill_result={'ok':True},source_response_path=str(payload),source_bytes=payload.stat().st_size,source_sha256=hashlib.sha256(payload.read_bytes()).hexdigest());receipt.write_text(json.dumps(r));return payload,receipt,q,r

def test_cache_replay_requires_exact_query_accept_status_length_and_hash(tmp_path):
    m=module();payload,receipt,q,r=fixture(tmp_path);cert=m.certify_receipt(receipt)
    assert m.load_cached(cert,q['path'],q['params'],'application/json')=={'results':[],'numberOfHits':0}
    with pytest.raises(ValueError):m.load_cached(cert,q['path'],dict(q['params'],page=2),'application/json')
    with pytest.raises(ValueError):m.load_cached(cert,q['path'],q['params'],'text/gaf')
    payload.write_text('{"results":[1],"numberOfHits":1}')
    with pytest.raises(ValueError):m.load_cached(cert,q['path'],q['params'],'application/json')

def test_partial_response_missing_receipt_or_failed_http_is_never_certified(tmp_path):
    m=module();payload,receipt,q,r=fixture(tmp_path)
    with pytest.raises(FileNotFoundError):m.certify_receipt(tmp_path/'missing.request.json')
    for change in [dict(http={'status':500}),dict(skill_result={'ok':False}),dict(source_bytes=1),dict(query=dict(q,base_url='https://other.example'))]:
        receipt.write_text(json.dumps(dict(r,**change)))
        with pytest.raises(ValueError):m.certify_receipt(receipt)

def test_preserved_query_batches_are_disjoint_and_conserve_exact_baseline():
    m=module();completed=[['P00001'],['Q00002']];tasks=[dict(canonical_accessions=['O00003','P00004']),dict(canonical_accessions=['Q00005'])]
    result=m.conservation(completed,tasks,{'P00001','Q00002','O00003','P00004','Q00005'})
    assert result['completed']==2 and result['remaining']==3 and result['union']==5
    with pytest.raises(ValueError):m.conservation(completed,tasks+[tasks[0]],{'P00001','Q00002','O00003','P00004','Q00005'})
    with pytest.raises(ValueError):m.conservation(completed,tasks,{'P00001','Q00002','O00003','P00004'})

def test_shared_request_start_reservations_and_raw_caps_cover_every_worker():
    m=module();ctx=multiprocessing.get_context('fork');g=m.SharedLimits(ctx,3,{'annotation':(1000,100),'ontology':(100,10)})
    scheduled=[g.reserve_start(10) for _ in range(4)]
    assert scheduled==pytest.approx([10,10+1/3,10+2/3,11])
    g.reserve_bytes('annotation',800);g.consume('annotation',20,20);g.release_bytes('annotation',780)
    assert g.snapshot()['annotation']['used']==120 and g.snapshot()['annotation']['reserved']==0
    with pytest.raises(ValueError):g.reserve_bytes('annotation',881,wait=False)
    g.block('HTTP429')
    with pytest.raises(ValueError):g.reserve_start(20)

def test_cache_certificate_rejects_unknown_accept_or_non_read_only_endpoint(tmp_path):
    m=module();payload,receipt,q,r=fixture(tmp_path)
    for change in [dict(path='annotation/delete'),dict(headers={'Accept':'application/octet-stream'})]:
        receipt.write_text(json.dumps(dict(r,query=dict(q,**change))))
        with pytest.raises(ValueError):m.certify_receipt(receipt)
