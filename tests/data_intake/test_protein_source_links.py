import importlib.util,hashlib
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists())
s=importlib.util.spec_from_file_location('protein_source_links',ROOT/'scripts/data_intake_protein_source_links.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def baseline():return {'P00001':dict(Entry='P00001',Sequence='ACD',Reviewed='reviewed',Annotation='4.0'),'P00002':dict(Entry='P00002',Sequence='ACD',Reviewed='reviewed',Annotation='5.0')}
def row():return dict(id='UniRef50_NOT_A_CANONICAL_ACCESSION',rep_accessions=['P00001'],sequence='ACD',seed_id='P00002',rep_member_id='EXAMPLE_HUMAN',rep_is_seed=False)
def test_exact_declared_representative_only_never_seed_cluster_or_sequence_neighbor():
 a,q=m.representative_links(row(),baseline());assert len(a)==1 and a[0]['qualified_accession']=='P00001' and not q
 assert a[0]['annotation_transfer_allowed'] is False and a[0]['homology_split_safe'] is False
 r=row();r['rep_accessions']=[];a,q=m.representative_links(r,baseline());assert not a and not q

def test_explicit_accession_sequence_mismatch_is_quarantined_not_repaired():
 r=row();r['sequence']='ACE';a,q=m.representative_links(r,baseline());assert not a and q[0]['reason']=='representative_sequence_differs_from_current_qualified_accession'
 assert q[0]['source_sequence_sha256']!=q[0]['current_qualified_sequence_sha256']

def test_reviewed_and_annotation_threshold_remain_specific_to_current_baseline_entry():
 b=baseline();b['P00001']['Annotation']='3.0';a,q=m.representative_links(row(),b);assert not a and q[0]['reason']=='current_baseline_reviewed_score_gate_failed'
 b=baseline();b['P00001']['Reviewed']='unreviewed';a,q=m.representative_links(row(),b);assert not a
