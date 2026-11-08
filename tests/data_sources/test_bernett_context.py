import sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,str(Path.cwd()/"scripts"))
from data_sources_bernett import unique_source_sequence

def test_unknown_source_subject_does_not_invent_FASTA_ID_and_ambiguous_is_not_guessed():
    mapping=defaultdict(list,{"P12345":[{"seq":"ACD"},{"seq":"ACE"}]})
    assert unique_source_sequence("Q12345",mapping) is None
    assert list(mapping)==["P12345"]
    assert unique_source_sequence("P12345",mapping) is None

def test_known_unique_source_record_keeps_exact_record_identity():
    row={"native_source_id":"P12345","seq":"ACD"}
    assert unique_source_sequence("P12345",{"P12345":[row]}) is row
