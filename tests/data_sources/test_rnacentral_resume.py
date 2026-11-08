import importlib.util,pathlib,json,hashlib
import pytest
ROOT=pathlib.Path.cwd()
def module():
    p=ROOT/'scripts/data_sources_rnacentral_resume.py';assert p.is_file(), 'immutable-original resume-only clone guard not implemented'
    s=importlib.util.spec_from_file_location('data_sources_rnacentral_resume',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def fixture(tmp_path):
    source=tmp_path/'original.fasta.gz';control=tmp_path/'original.fasta.gz.aria2';source.write_bytes(b'original sparse-like prefix'+b'\0'*100);control.write_bytes(b'original aria2 control')
    cp=dict(original_worker_pid=99999999,source_strong_ETag='"fixed-entity"',expected_full_active_bytes=250,reported_completed_bytes=100,immutable_original_files=[dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in [source,control]])
    asset=dict(expected_bytes=250,expected_etag='"fixed-entity"');return source,control,cp,asset

def test_clone_preserves_every_original_byte_and_control_and_has_independent_writes(tmp_path):
    m=module();source,control,cp,a=fixture(tmp_path);before=(source.read_bytes(),control.read_bytes());out=tmp_path/'exclusive-new';r=m.clone_partial(cp,a,out,'copied.fasta.gz');target=pathlib.Path(r['destination']);assert target.read_bytes()==before[0] and pathlib.Path(str(target)+'.aria2').read_bytes()==before[1]
    target.write_bytes(b'new resume bytes');assert (source.read_bytes(),control.read_bytes())==before

def test_changed_original_or_missing_control_fails_before_any_copy(tmp_path):
    m=module();source,control,cp,a=fixture(tmp_path);out=tmp_path/'must-not-create';source.write_bytes(b'tampered')
    with pytest.raises(ValueError):m.clone_partial(cp,a,out,'new.gz')
    assert not out.exists();source.write_bytes(b'original sparse-like prefix'+b'\0'*100);control.unlink()
    with pytest.raises(ValueError):m.clone_partial(cp,a,out,'new.gz')
    assert not out.exists()

def test_existing_new_destination_is_never_overwritten_or_reused(tmp_path):
    m=module();source,control,cp,a=fixture(tmp_path);out=tmp_path/'existing';out.mkdir();p=out/'copied.fasta.gz';p.write_bytes(b'protected previous attempt')
    with pytest.raises(FileExistsError):m.clone_partial(cp,a,out,p.name)
    assert p.read_bytes()==b'protected previous attempt'

def test_entity_length_or_ETag_change_refuses_progress_reuse(tmp_path):
    m=module();source,control,cp,a=fixture(tmp_path)
    for new in [dict(a,expected_bytes=251),dict(a,expected_etag='"changed"')]:
        with pytest.raises(ValueError):m.validate_checkpoint(cp,new)

def test_no_progress_reset_or_unbounded_fresh_download_fallback():
    m=module();assert m.checked_new_bytes(100,dict(downloaded_bytes=150),60)==50
    assert m.checked_new_bytes(100,None,60) is None
    with pytest.raises(ValueError):m.checked_new_bytes(100,dict(downloaded_bytes=0),60)
    with pytest.raises(ValueError):m.checked_new_bytes(100,dict(downloaded_bytes=161),60)
