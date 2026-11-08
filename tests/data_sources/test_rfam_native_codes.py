import pathlib,sys,importlib.util
import pytest
ROOT=pathlib.Path.cwd();sys.path.insert(0,str(ROOT/'scripts'))
def module():
    p=ROOT/'scripts/data_sources_rfam_v2.py'
    assert p.is_file(), 'release-native nonBoolean Rfam truncation-code parser not implemented'
    s=importlib.util.spec_from_file_location('data_sources_rfam_v2',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_actual_release_code5_is_preserved_without_binary_interpretation():
    m=module();r=m.parse_region('RF00001 JABFAC010159342.1 206 140 38.10 1.5 54 120 5 full')
    assert r['native_truncation_code_text']=='5' and type(r['native_truncation_code_integer']) is int
    assert r['native_truncation_code_integer']==5 and r['truncation_boolean_interpretation_inferred'] is False
    assert r['native_fasta_identifier_key']=='JABFAC010159342.1/206-140'

def test_all_four_observed_release_codes_remain_exact_source_categories():
    m=module()
    for code in ['0','3','5','53']:
        r=m.parse_region('RF00001 A.1 1 2 38 1e-9999 1 2 '+code+' full')
        assert r['native_truncation_code_text']==code and r['native_evalue_text']=='1e-9999'
        assert not r['source_quality_fields_are_inference_conditions']

def test_unknown_future_encoding_or_invalid_numerical_evidence_cannot_silently_pass():
    m=module()
    for line in ['RF00001 A.1 1 2 38 1 1 2 1 full','RF00001 A.1 1 2 38 -1 1 2 0 full']:
        with pytest.raises(ValueError):m.parse_region(line)
