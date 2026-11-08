import importlib.util,pathlib,hashlib
import pytest
ROOT=pathlib.Path.cwd()
def module():
    p=ROOT/'scripts/data_sources_uniref.py'
    assert p.is_file(), 'UniRef native record QC is not implemented yet'
    s=importlib.util.spec_from_file_location('data_sources_uniref',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def record(seq='MABZXJUO'):
    import xxhash
    return dict(id='UniRef50_Q12345',seed_id='UPI0000000001',rep_member_id='ENTRY_HUMAN',rep_member_id_type='UniProtKB ID',rep_accessions=['Q12345'],rep_is_seed=False,member_ids=['ENTRY_HUMAN','UPI0000000001'],member_count=2,sequence=seq,sequence_length=len(seq),sequence_crc64='0123456789ABCDEF',sequence_xxh128=xxhash.xxh128_hexdigest(seq.encode()),go_mf=['GO:0003674'],go_bp=[],go_cc=[],updated='2026-01-28')

def test_native_long_and_ambiguous_sequences_retain_full_axis_checks():
    m=module();r=record('M'*49499);qc=m.qc_record(r,{})
    assert qc['native_sequence_length']==49499 and qc['sequence_integrity_pass'] and qc['cluster_metadata_integrity_pass']
    assert qc['native_sequence_sha256']==hashlib.sha256(r['sequence'].encode()).hexdigest()
    q=m.qc_record(record(),{});assert q['sequence_integrity_pass'] and q['sequence_crc64_recomputed'] is False

def test_member_seed_and_representative_roles_are_not_collapsed():
    m=module();r=record();q=m.qc_record(r,{'Q12345':hashlib.sha256(r['sequence'].encode()).hexdigest()})
    assert q['cluster_id']=='UniRef50_Q12345' and q['seed_id']=='UPI0000000001' and q['representative_member_id']=='ENTRY_HUMAN'
    assert q['qualified_representative_exact_sequence_accessions']==['Q12345']
    assert q['member_annotation_transfer_allowed'] is False and q['new_uniprot_admissions']==0

def test_accession_match_without_exact_sequence_cannot_qualify():
    m=module();q=m.qc_record(record(),{'Q12345':hashlib.sha256(b'OTHER').hexdigest()})
    assert not q['qualified_representative_exact_sequence_accessions']
    assert q['qualified_representative_sequence_mismatch_accessions']==['Q12345']

def test_sequence_checksum_and_membership_inconsistency_are_separate_components():
    m=module();r=record();r['member_count']=3;q=m.qc_record(r,{})
    assert q['sequence_integrity_pass'] and not q['cluster_metadata_integrity_pass']
    r=record();r['sequence']='MA*';q=m.qc_record(r,{})
    assert not q['sequence_integrity_pass'] and q['cluster_metadata_integrity_pass']
