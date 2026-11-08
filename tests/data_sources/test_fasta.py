import importlib.util,pathlib,io
import pytest
ROOT=pathlib.Path.cwd()
def module():
    p=ROOT/'scripts/data_sources_fasta.py'
    assert p.is_file(), 'immutable source FASTA parser and full-axis QC not implemented'
    s=importlib.util.spec_from_file_location('data_sources_fasta',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_multiline_sequence_case_and_complete_header_are_preserved():
    m=module();records=list(m.fasta_records(io.StringIO('>ACCESSION.1/120-1 family=RF00001\nAcGU\nNn\n\n>URS0000000001 example\nTTAC\n')))
    assert records==[('ACCESSION.1/120-1 family=RF00001','AcGUNn'),('URS0000000001 example','TTAC')]

def test_sequence_before_header_and_missing_identifier_fail_closed():
    m=module()
    for text in ['ACGU\n>id\nACGU\n','>\nACGU\n']:
        with pytest.raises(ValueError):list(m.fasta_records(io.StringIO(text)))

def test_long_native_sequence_axis_is_not_cut_to_legacy8192_window():
    m=module();sequence='ACGUTRYSWKMBDHVN'*600;qc=m.sequence_qc('URS0000000001 header',sequence)
    assert qc['native_length']==9600 and qc['sequence_alphabet_pass'] and qc['source_identifier']=='URS0000000001'
    assert qc['source_sequence_preserved'] and qc['native_T_count']==600 and qc['native_U_count']==600

def test_empty_or_gapped_fasta_sequence_is_quarantined_and_not_rewritten():
    m=module()
    for sequence in ['', 'AC-GU','AC GU']:
        q=m.sequence_qc('id',sequence);assert not q['sequence_alphabet_pass']
    assert m.sequence_qc('id','acgu')['sequence_alphabet_pass']
