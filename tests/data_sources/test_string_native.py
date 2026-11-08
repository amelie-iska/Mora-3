import hashlib,importlib.util,pathlib
import pytest
import pyarrow as pa
ROOT=pathlib.Path.cwd()
def module():
    p=ROOT/'scripts/data_sources_string.py'
    assert p.is_file(), 'native STRING source validation is not implemented'
    s=importlib.util.spec_from_file_location('data_sources_string',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def batch(m,rows):
    return pa.RecordBatch.from_pylist(rows,schema=m.PHYSICAL_SCHEMA)

def row(m,**updates):
    r=dict(zip(m.PHYSICAL_COLUMNS,['9606.ENSP0001','9606.ENSP0002',0,200,300,400,500,600,700,800]));r.update(updates);return r

def test_native_physical_score_profile_and_oriented_occurrences_are_preserved():
    m=module();r=row(m);b=batch(m,[r,r]);q=m.physical_batch_qc(b)
    assert q['native_rows']==2 and q['self_pair_rows']==0 and q['cross_taxon_rows']==0
    assert q['channel_nonzero']['experiments']==2 and q['channel_nonzero']['experiments_transferred']==2
    assert b.to_pylist()==[r,r]

def test_invalid_provider_namespace_nulls_and_out_of_range_scores_fail_closed():
    m=module()
    for u in [dict(protein1='ENSP1'),dict(protein2='9606.'),dict(protein1=None),dict(database=1001)]:
        with pytest.raises(ValueError):m.physical_batch_qc(batch(m,[row(m,**u)]))
    with pytest.raises(ValueError):m.physical_batch_qc(pa.record_batch([['x']],names=['different_schema']))

def test_cross_taxon_and_self_pairs_are_counted_without_fabricated_labels():
    m=module();q=m.physical_batch_qc(batch(m,[row(m,protein2='10090.ENSMUSP1'),row(m,protein2='9606.ENSP0001')]))
    assert q['cross_taxon_rows']==1 and q['self_pair_rows']==1

def test_qualified_sequence_candidate_requires_exact_native_sequence_and_no_alias_transfer():
    m=module();seq='ACDEFGHIK';h=hashlib.sha256(seq.encode()).hexdigest();qualified={h:['P00001','Q00002']}
    q=m.provider_sequence_qc('9606.ENSP0001 complete header',seq,qualified)
    assert q['qualified_exact_sequence_accessions']==['P00001','Q00002'] and q['native_sequence_sha256']==h
    assert q['provider_id']=='9606.ENSP0001' and q['taxon_id']==9606 and q['native_header'].endswith('complete header')
    assert not q['task_admitted'] and m.provider_sequence_qc('9606.ENSP0001',seq.lower(),qualified)['qualified_exact_sequence_accessions']==[]
    assert not m.provider_sequence_qc('9606.ENSP0001','AC-D',qualified)['sequence_alphabet_pass']

def test_alias_provenance_remains_literal_and_is_not_sequence_or_isoform_proof():
    m=module();q=m.alias_record('9606.ENSP0001\tP00001\tEnsembl_UniProt_AC Ensembl_HGNC\n')
    assert q==dict(provider_id='9606.ENSP0001',alias='P00001',native_source='Ensembl_UniProt_AC Ensembl_HGNC',source_tokens=['Ensembl_UniProt_AC','Ensembl_HGNC'])
    for s in ['9606.ENSP1\tx\n','ENSP1\tx\tUniProt_AC\n','9606.ENSP1\t\tUniProt_AC\n']:
        with pytest.raises(ValueError):m.alias_record(s)
