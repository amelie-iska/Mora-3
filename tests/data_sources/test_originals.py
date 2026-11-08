import pathlib,importlib.util,gzip,hashlib
import pytest
ROOT=pathlib.Path.cwd()
def module():
    import sys
    sys.path.insert(0,str(ROOT/'scripts'))
    p=ROOT/'scripts/data_sources_originals.py'
    assert p.is_file(), 'original source whole-file integrity helper not implemented'
    s=importlib.util.spec_from_file_location('data_sources_originals',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_whole_gzip_crc_and_decoded_bytes_pass_without_mutation(tmp_path):
    m=module();p=tmp_path/'native.fa.gz';raw=gzip.compress(b'>id\nACGU\n');p.write_bytes(raw)
    result=m.finalize(p,{'expected_bytes':len(raw),'expected_sha256':None,'expected_md5':hashlib.md5(raw).hexdigest(),'format':'gzip_fasta'})
    assert result['gzip_crc_full_stream_passed'] and result['uncompressed_bytes_streamed']==9 and result['upstream_md5_passed']
    assert p.read_bytes()==raw

def test_truncated_gzip_fails_and_preserves_original(tmp_path):
    m=module();p=tmp_path/'native.fa.gz';raw=gzip.compress(b'>id\nACGU\n')[:-4];p.write_bytes(raw)
    with pytest.raises((EOFError,OSError)):m.finalize(p,{'expected_bytes':len(raw),'expected_sha256':None,'expected_md5':None,'format':'gzip_fasta'})
    assert p.read_bytes()==raw

def test_md5_mismatch_does_not_overwrite_or_remove_source(tmp_path):
    m=module();p=tmp_path/'native.h5';raw=b'\x89HDF\r\n\x1a\nsource';p.write_bytes(raw)
    with pytest.raises(ValueError,match='MD5'):m.finalize(p,{'expected_bytes':len(raw),'expected_sha256':None,'expected_md5':'0'*32,'format':'hdf5'})
    assert p.read_bytes()==raw

def test_hdf5_signature_is_separate_from_full_structural_validation(tmp_path):
    m=module();p=tmp_path/'native.h5';raw=b'\x89HDF\r\n\x1a\nsource';p.write_bytes(raw)
    result=m.finalize(p,{'expected_bytes':len(raw),'expected_sha256':None,'expected_md5':hashlib.md5(raw).hexdigest(),'format':'hdf5'})
    assert result['HDF5_signature_passed'] and result['HDF5_structural_QC_performed'] is False
