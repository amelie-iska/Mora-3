import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path.cwd()/"scripts"))
from data_sources_bernett import source_pair,protein_source_record,qualified_sequence_disposition

def test_sampled_negative_role_split_and_pair_parent_are_explicit():
    r=source_pair("Q9NTX9 Q9NWT6", "Intra0_neg_rr.txt", 7)
    assert r["benchmark_label"]==0 and r["source_native_split"]=="validation"
    assert r["evidence_class"]=="algorithmically_sampled_negative_pair"
    assert r["experimental_noninteraction"] is False and r["direct_binding_proven"] is False
    assert not r["task_admitted"] and not r["model_adapter_supported"]
    rev=source_pair("Q9NWT6 Q9NTX9", "Intra0_neg_rr.txt", 8)
    assert r["undirected_accession_pair_key"]==rev["undirected_accession_pair_key"]
    assert r["source_record_id"]!=rev["source_record_id"]

@pytest.mark.parametrize("line,name",[("P12345","Intra1_pos_rr.txt"),("P12345 Q12345 1","Intra1_pos_rr.txt"),("BAD Q12345","Intra1_pos_rr.txt"),("P12345 Q12345","train.txt")])
def test_bad_native_pair_or_renamed_split_cannot_be_silently_repaired(line,name):
    with pytest.raises(ValueError):source_pair(line,name,1)

def test_exact_qualified_current_entry_and_sequence_both_required():
    base={"P12345":{"sequence":"ACD","annotation_score":4,"reviewed":True},"Q12345":{"sequence":"ACD","annotation_score":3,"reviewed":True}}
    assert qualified_sequence_disposition("P12345","ACD",base)["exact_qualified_current_entry_sequence_candidate"]
    for accession,seq in [("P12345","ACE"),("Q12345","ACD"),("O12345","ACD"),("P12345-2","ACD")]:
        assert not qualified_sequence_disposition(accession,seq,base)["exact_qualified_current_entry_sequence_candidate"]
    base["P12345"]["reviewed"]=False
    assert not qualified_sequence_disposition("P12345","ACD",base)["exact_qualified_current_entry_sequence_candidate"]

def test_form_and_full_source_axis_are_retained_without_canonical_transfer():
    r=protein_source_record("P12345-2 original source form", "W"*9000)
    assert r["native_source_id"]=="P12345-2" and not r["canonical_accession_syntax"]
    assert len(r["native_sequence"])==r["native_length"]==9000
    assert r["native_header"]=="P12345-2 original source form" and not r["task_admitted"]
