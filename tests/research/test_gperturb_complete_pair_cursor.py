"""Pure test-only mutations; not campaign records, not a source qualification."""
from copy import deepcopy
import pytest

from mora3.research.gperturb_full_common_pair_cursor import checked_native_pair


def rows():
    context = dict(study="GSM4150377", cell_line="A549",
                   exposure_hours=24, baseline=None, desired_response=None)
    intervention = dict(identity="Dex", source_coordinate=.001,
                        units="source_scaled_coordinate",
                        physical_concentration=None, is_vehicle=False)
    common = dict(
        source_index=12, cell_id="test-only-cell", source_context=context,
        original_group="study:GSE139944:GSM4150377",
        source_shard_sha256="a"*64, expression_sha256="b"*64,
        axis_sha256="c"*64, intervention=intervention,
        original_source_revision="test-only-original",
    )
    response = common | dict(
        direction="response", condition_keys=("context","intervention"),
        target_keys=("expression",), native_payload_sha256="d"*64,
        target_positions=5000,
    )
    inverse = common | dict(
        direction="intervention", condition_keys=("context","expression"),
        target_keys=("intervention",), native_payload_sha256="e"*64,
        target_positions=2,
    )
    return response, inverse


def test_pure_exact_source_roles_and_distinct_record_ordinals():
    response, inverse = rows()
    result = checked_native_pair(12, response, inverse, 24, 25)
    assert result["source_index"] == 12
    assert result["cell_id_hash"] != response["cell_id"]
    assert result["response_ordinal"] == 24
    assert result["intervention_ordinal"] == 25
    assert result["expression_sha256"] == "b"*64
    assert result["native_target_positions"] == {"response":5000,"intervention":2}


@pytest.mark.parametrize("which,change",[
    ("response", {"source_index":13}),
    ("inverse", {"cell_id":"changed-cell"}),
    ("inverse", {"expression_sha256":"f"*64}),
    ("inverse", {"axis_sha256":"f"*64}),
    ("response", {"target_keys":("expression","intervention")}),
    ("inverse", {"condition_keys":("expression","context","intervention")}),
    ("inverse", {"source_shard_sha256":"f"*64}),
    ("inverse", {"native_payload_sha256":"d"*64}),
    ("response", {"original_source_revision":"different"}),
    ("inverse", {"intervention":dict(identity="Dex",source_coordinate=.001,
                                     units="source_scaled_coordinate",physical_concentration=.1,
                                     is_vehicle=False)}),
])
def test_pure_corruption_refused(which, change):
    r, i=rows()
    selected=r if which=="response" else i
    selected.update(change)
    with pytest.raises(ValueError):
        checked_native_pair(12,r,i,24,25)


@pytest.mark.parametrize("count,rord,iord",[(13,24,25),(12,24,24),
                                            (12,-1,25),(12,25,24)])
def test_pure_wrong_order_or_same_task_ordinal_refused(count,rord,iord):
    r,i=rows()
    with pytest.raises(ValueError):
        checked_native_pair(count,r,i,rord,iord)


def test_previous_chunk_requires_exact_raw_byte_sha_not_recomputed_from_json():
    import json
    from hashlib import sha256
    from mora3.research.gperturb_full_common_pair_cursor import verified_previous_input
    payload = {'schema':'test-only','source_index':511}
    raw = json.dumps(payload).encode()
    assert verified_previous_input(raw, sha256(raw).hexdigest()) == payload
    with pytest.raises(ValueError):
        verified_previous_input(raw, '0'*64)
    with pytest.raises(ValueError):
        verified_previous_input(raw + b' ', sha256(raw).hexdigest())
