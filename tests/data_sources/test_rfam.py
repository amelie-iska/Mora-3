import pathlib,sys,importlib.util,io
import pytest
ROOT=pathlib.Path.cwd()
def module():
    p=ROOT/'scripts/data_sources_rfam.py'
    assert p.is_file(), 'native Rfam family/confidence parser not implemented'
    s=importlib.util.spec_from_file_location('data_sources_rfam',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_native_reverse_region_coordinate_and_decimal_evalue_are_not_normalized():
    m=module();r=m.parse_region('RF00001\tFR821777.2\t2657646\t2657543\t38.10\t1e-9999\t1\t120\t0\tfull')
    assert r['family_accession']=='RF00001' and r['native_start']==2657646 and r['native_end']==2657543
    assert r['native_fasta_identifier_key']=='FR821777.2/2657646-2657543' and r['native_evalue_text']=='1e-9999'
    assert r['native_bitscore_text']=='38.10' and r['source_quality_fields_are_inference_conditions'] is False

def test_region_unsupported_schema_or_invalid_evidence_is_rejected():
    m=module()
    for line in ['RF00001 A.1 1 2 38 1e-5 1 2 0', 'RF00001 A.1 1 2 38 -1 1 2 0 full', 'RF00001 A.1 1 2 38 1 1 2 5 full']:
        with pytest.raises(ValueError):m.parse_region(line)

def test_seed_metadata_preserves_repeated_provenance_and_consensus_alignment_axis():
    m=module();text='# STOCKHOLM 1.0\n#=GF AC RF00001\n#=GF ID tRNA\n#=GF SS Predicted; CMfinder;\n#=GF GA 100.00\n#=GF RT first line\n#=GF RT continuation\na/1-5 AC-GU\nb/5-1 ACUGU\n#=GC SS_cons ((.))\n//\n';r=list(m.seed_families(io.StringIO(text)))[0]
    assert r['family_accession']=='RF00001' and r['family_name']=='tRNA' and r['SS_cons_native']=='((.))'
    assert r['source_GF_fields']['RT']==['first line','continuation'] and r['source_GF_fields']['SS']==['Predicted; CMfinder;']
    assert r['seed_sequences']==2 and r['alignment_columns']==5 and r['alignment_axis_quality_pass']

def test_clanin_names_bind_through_exact_seed_ID_and_never_assumed_RFaccessions():
    m=module();rows=m.clan_bindings(['CL00001\ttRNA\tunknown_name'],{'tRNA':['RF00001']})
    assert rows[0]['family_accession']=='RF00001' and rows[0]['exact_seed_family_name_bound']
    assert rows[1]['family_accession'] is None and not rows[1]['exact_seed_family_name_bound']
