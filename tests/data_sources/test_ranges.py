from pathlib import Path
import hashlib
import sys
import pytest
ROOT=next(p for p in Path(__file__).resolve().parents if (p/"AGENTS.md").exists())
sys.path.insert(0,str(ROOT/"scripts"))
from data_sources_fetch_ranges import plan_ranges, validate_range_response, assemble_asset


def test_ranges_cover_remaining_suffix_once_without_overlap():
    assert plan_ranges(3,10,3)==[(3,5),(6,8),(9,9)]
    assert plan_ranges(10,10,3)==[]
    with pytest.raises(ValueError):plan_ranges(11,10,3)


def test_every_range_requires_exact_entity_and_bounds_before_write():
    a=dict(expected_bytes=10,expected_etag='"fixed"')
    validate_range_response(a,3,5,206,{"Content-Range":"bytes 3-5/10","Content-Length":"3","ETag":'"fixed"'})
    for status,headers in [(200,{}),(206,{"Content-Range":"bytes 2-5/10","ETag":'"fixed"'}),(206,{"Content-Range":"bytes 3-5/10","ETag":'"changed"'}),(206,{"Content-Range":"bytes 3-5/11","ETag":'"fixed"'})]:
        with pytest.raises(ValueError):validate_range_response(a,3,5,status,headers)


def test_assembly_reuses_preserved_prefix_and_ordered_pieces_without_overwrite(tmp_path):
    prefix=tmp_path/"original.partial";prefix.write_bytes(b"abc")
    p1=tmp_path/"piece3";p1.write_bytes(b"de");p2=tmp_path/"piece5";p2.write_bytes(b"f")
    asset=dict(expected_bytes=6,expected_sha256=hashlib.sha256(b"abcdef").hexdigest(),destination_path="new/final.bin",prefix=dict(path=str(prefix),bytes=3,sha256=hashlib.sha256(b"abc").hexdigest()))
    pieces=[dict(start=3,end=4,path=str(p1),sha256=hashlib.sha256(b"de").hexdigest()),dict(start=5,end=5,path=str(p2),sha256=hashlib.sha256(b"f").hexdigest())]
    result=assemble_asset(asset,tmp_path,list(reversed(pieces)))
    assert (tmp_path/"new/final.bin").read_bytes()==b"abcdef" and prefix.read_bytes()==b"abc"
    assert result["bytes"]==6
    with pytest.raises(FileExistsError):assemble_asset(asset,tmp_path,pieces)


def test_changed_preserved_prefix_blocks_assembly(tmp_path):
    p=tmp_path/"prefix";p.write_bytes(b"wrong")
    a=dict(expected_bytes=5,destination_path="new/final.bin",prefix=dict(path=str(p),bytes=5,sha256="0"*64))
    with pytest.raises(ValueError,match="prefix"):assemble_asset(a,tmp_path,[])
    assert not (tmp_path/"new/final.bin").exists()
