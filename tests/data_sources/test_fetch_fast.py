from pathlib import Path
import sys
import pytest
ROOT=next(p for p in Path(__file__).resolve().parents if (p/"AGENTS.md").exists())
sys.path.insert(0,str(ROOT/"scripts"))
from data_sources_fetch_fast import asset_lease, validate_plan, retry_delay


def plan():
    return dict(workers=8,per_host_connections=8,raw_budget_bytes=100,other_registered_raw_bytes=40,assets=[dict(id="x",relative_path="raw/new.bin",expected_bytes=50,url="https://example.invalid/x")])


def test_duplicate_asset_writers_cannot_take_the_same_kernel_lease(tmp_path):
    p=tmp_path/"lease"
    with asset_lease(p):
        with pytest.raises(BlockingIOError):
            with asset_lease(p): pass
    assert p.exists()
    with asset_lease(p): pass


def test_duplicate_paths_and_ids_are_rejected_before_acquisition():
    p=plan();validate_plan(p)
    p["assets"].append(dict(p["assets"][0]))
    with pytest.raises(ValueError):validate_plan(p)


def test_budget_and_path_escape_are_rejected():
    p=plan();p["other_registered_raw_bytes"]=99
    with pytest.raises(ValueError):validate_plan(p)
    p=plan();p["assets"][0]["relative_path"]="../protected.bin"
    with pytest.raises(ValueError):validate_plan(p)


def test_retry_after_is_honored_without_an_unbounded_retry_loop():
    assert retry_delay(0,{"Retry-After":"45"})==45
    assert retry_delay(1,{})==15
    with pytest.raises(ValueError):retry_delay(0,{"Retry-After":"9999"})
