import pathlib,sys,hashlib
import pytest
sys.path.insert(0,str(pathlib.Path.cwd()/'scripts'))
from data_sources_fetch_ranges import assemble_asset

def setup_asset(tmp_path,md5):
    prefix=tmp_path/'prefix';prefix.write_bytes(b'AB');piece=tmp_path/'piece';piece.write_bytes(b'CDEF')
    asset=dict(id='fixture',expected_bytes=6,expected_sha256=None,expected_md5=md5,destination_path='derived/assembled',prefix=dict(path=str(prefix),bytes=2,sha256=hashlib.sha256(b'AB').hexdigest()))
    pieces=[dict(path=str(piece),start=2,end=5,sha256=hashlib.sha256(b'CDEF').hexdigest())]
    return asset,pieces,prefix,piece

def test_assembled_source_rejects_wrong_upstream_md5_and_preserves_raw_pieces(tmp_path):
    asset,pieces,prefix,piece=setup_asset(tmp_path,'0'*32)
    with pytest.raises(ValueError,match='MD5'):assemble_asset(asset,tmp_path,pieces)
    assert prefix.read_bytes()==b'AB' and piece.read_bytes()==b'CDEF'
    assert (tmp_path/'derived/assembled').read_bytes()==b'ABCDEF'

def test_assembled_source_verifies_published_md5_before_availability(tmp_path):
    asset,pieces,prefix,piece=setup_asset(tmp_path,hashlib.md5(b'ABCDEF').hexdigest());r=assemble_asset(asset,tmp_path,pieces)
    assert r['upstream_md5_passed'] and r['md5']==asset['expected_md5'] and prefix.read_bytes()==b'AB'
