"""Exact live source windows distinguish GOA form scope from corruption."""
from dataclasses import replace
from itertools import islice
from pathlib import Path

import pyarrow as pa
import pytest

from mora3.data.goa_streamed import GOAFormOutsideCanonicalScope
from mora3.research.goa_campaign_route import GOACampaignAdapter
from mora3.research.rfam_goa_registry import GOABudget,load_original_rfam_goa_assets


def test_verified_native_forms_do_not_stop_canonical_route(tmp_path):
    _,pool,_,_,_=load_original_rfam_goa_assets(Path.cwd(),goa_budget=GOABudget(3,1))
    adapter=GOACampaignAdapter(pool,'a'*64)
    route=adapter.train_binding().route
    windows=tuple(islice(route.iter_from(route.initial_cursor),4))
    assert len(windows)==4
    assert adapter.prepare(windows[0]).observed_positions>0
    for window in windows[1:3]:
        with pytest.raises(GOAFormOutsideCanonicalScope):
            pool.bind_native_rows(window.batch,split='train')
        prepared=adapter.prepare(window)
        assert prepared.observed_positions==0 and prepared.evaluate is None
        assert 'separate source policy' in prepared.unavailable_reason
    prepared=adapter.prepare(windows[3])
    assert prepared.observed_positions==0
    assert prepared.unavailable_reason=='original GOA packet has no observed signs'
    source=windows[1].batch
    col=source.native_rows.schema.get_field_index('native_payload_sha256')
    altered=source.native_rows.set_column(col,'native_payload_sha256',pa.array(['0'*64]))
    with pytest.raises(ValueError) as error:
        adapter.prepare(replace(windows[1],batch=replace(source,native_rows=altered)))
    assert type(error.value) is not GOAFormOutsideCanonicalScope
