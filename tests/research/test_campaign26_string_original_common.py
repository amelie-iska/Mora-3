"""Frozen original STRING score axes and two-window common-source readback tests."""
from pathlib import Path
import pytest

from mora3.research.campaign26_string_original_common import (
    audit_string_pair, verified_score_packet,
)

ACTIVE=Path("/home/iska/Documents/amelie/bio/Mora-3")

def test_score_must_remain_native_four_channel_integer_not_experimental_binding():
    valid={"experimental_channel":0,"database_channel":200,
           "textmining_channel":700,"combined_score":730}
    assert verified_score_packet(valid,(0,200,700,730)) == 4
    with pytest.raises(ValueError,match="native"):
        verified_score_packet({**valid,"combined_score":731},(0,200,700,730))
    with pytest.raises(ValueError,match="native"):
        verified_score_packet({**valid,"combined_score":0.73},(0,200,700,730))

def test_actual_string_two_original_windows_candidate_common():
    result=audit_string_pair(ACTIVE)
    assert result["schema"]=="Mora3.Campaign26StringTwoOriginalCommon/v1"
    assert result["task_key"]=="string_score_profile"
    assert result["windows_per_binding"]==2
    assert result["candidate_common_native_equal"] is True
    assert result["native_targets_per_binding"]==8
    assert result["source_snapshot"]=="79d328f542d1158581875ef325e56f213b407992f95db6ee0c7352a3459669d4"
    assert result["source_view_policy"]=="c3812ecfcfea8a01f81769bf5ef6fb3880e04a0ac054a6349e9272db14765393"
    assert result["training_admission"] is False
    assert result["actual_counted_trials"] == result["optimizer_updates"] == 0
