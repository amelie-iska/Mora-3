from pathlib import Path
import io
import sys
import hashlib
import pytest
ROOT=next(p for p in Path(__file__).resolve().parents if (p/"AGENTS.md").exists())
sys.path.insert(0,str(ROOT/"scripts"))
from data_sources_fetch import validate_response, fetch_asset

class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data);self.status=status;self.headers=headers or {}

def asset(data=b"abcd"):
    return dict(id="fixture",url="https://example.invalid/data",relative_path="raw/pending.bin",expected_bytes=len(data),expected_sha256=hashlib.sha256(data).hexdigest(),expected_etag='"release"')

def test_resume_requires_exact_range_and_unchanged_validator():
    a=asset();validate_response(a,2,206,{"Content-Range":"bytes 2-3/4","ETag":'"release"'})
    for status,headers in [(200,{}),(206,{"Content-Range":"bytes 1-3/4"}),(206,{"Content-Range":"bytes 2-3/5"}),(206,{"Content-Range":"bytes 2-3/4","ETag":'"changed"'})]:
        with pytest.raises(ValueError):validate_response(a,2,status,headers)

def test_completed_corrupt_asset_is_never_overwritten(tmp_path):
    p=tmp_path/"raw/pending.bin";p.parent.mkdir();p.write_bytes(b"xxxx")
    with pytest.raises(ValueError,match="checksum"):
        fetch_asset(asset(),tmp_path,opener=lambda *_:pytest.fail("complete file must not redownload"))
    assert p.read_bytes()==b"xxxx"

def test_resumption_appends_only_verified_range(tmp_path):
    p=tmp_path/"raw/pending.bin";p.parent.mkdir();p.write_bytes(b"ab")
    def opener(req,timeout):
        assert req.get_header("Range")=="bytes=2-"
        return Response(b"cd",206,{"Content-Range":"bytes 2-3/4","ETag":'"release"'})
    r=fetch_asset(asset(),tmp_path,opener=opener)
    assert p.read_bytes()==b"abcd" and r["state"]=="verified_available"

def test_ignored_resume_request_cannot_append_full_file(tmp_path):
    p=tmp_path/"raw/pending.bin";p.parent.mkdir();p.write_bytes(b"ab")
    with pytest.raises(ValueError):fetch_asset(asset(),tmp_path,opener=lambda *a,**k:Response(b"abcd",200,{}))
    assert p.read_bytes()==b"ab"

def test_oversized_server_chunk_is_rejected_before_write(tmp_path):
    with pytest.raises(ValueError,match="cap"):
        fetch_asset(asset(),tmp_path,opener=lambda *a,**k:Response(b"abcde",200,{"ETag":'"release"'}))
    assert (tmp_path/"raw/pending.bin").read_bytes()==b""
