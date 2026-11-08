import importlib.util,pathlib,json,sys
import pytest
ROOT=pathlib.Path.cwd();sys.path.insert(0,str(ROOT/'scripts'))
def module():
    p=ROOT/'scripts/data_sources_rnacentral_resume_run.py';s=importlib.util.spec_from_file_location('rnacentral_wrapper_guards',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_exact_execution_review_plan_and_all_helper_hashes_required(tmp_path):
    m=module();p=tmp_path/'plan.json';p.write_text('{"job_id":"fixture-only"}');a=tmp_path/'approval.json';r=dict(approved=True,action='RNAcentral28_immutable_copy_resume_v2',plan_sha256=m.sha(p),reviewed_code_sha256={x:m.sha(x) for x in m.REVIEWED_FILES});a.write_text(json.dumps(r));assert m.verify_review(p,a)['job_id']=='fixture-only'
    r['reviewed_code_sha256']['scripts/data_sources_originals.py']='changed';a.write_text(json.dumps(r))
    with pytest.raises(ValueError):m.verify_review(p,a)
    r['reviewed_code_sha256']={x:m.sha(x) for x in m.REVIEWED_FILES};a.write_text(json.dumps(r));p.write_text('{"job_id":"different"}')
    with pytest.raises(ValueError):m.verify_review(p,a)

def test_changed_source_HEAD_entity_rejected_before_range_or_copy(monkeypatch):
    m=module();calls=[]
    class Reply:
        status=200;headers={'Content-Length':'100','ETag':'"changed"'}
        def __enter__(self):return self
        def __exit__(self,*a):return False
    class Opener:
        def open(self,*a,**k):calls.append(1);return Reply()
    monkeypatch.setattr(m.urllib.request,'build_opener',lambda *a:Opener())
    with pytest.raises(ValueError,match='entity mismatch'):m.preflight([dict(url='https://ftp.ebi.ac.uk/fixture',id='fixture',expected_bytes=100,expected_etag='"original"')],{})
    assert len(calls)==1

def test_200_noRange_response_is_refused_even_if_first_bytes_match(monkeypatch,tmp_path):
    m=module();p=tmp_path/'original';p.write_bytes(b'x'*65536)
    class Reply:
        status=200;headers={'Content-Length':'65536','ETag':'"original"','Content-Range':'bytes 0-65535/65536'}
        def __enter__(self):return self
        def __exit__(self,*a):return False
        def read(self,n):return b'x'*65536
    class Opener:
        def open(self,*a,**k):return Reply()
    monkeypatch.setattr(m.urllib.request,'build_opener',lambda *a:Opener())
    with pytest.raises(ValueError,match='206'):m.preflight([dict(url='https://ftp.ebi.ac.uk/fixture',id='fixture',expected_bytes=65536,expected_etag='"original"')],dict(immutable_original_files=[dict(path=str(p))]))
    assert p.read_bytes()==b'x'*65536
