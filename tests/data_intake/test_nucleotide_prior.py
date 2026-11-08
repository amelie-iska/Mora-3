import importlib.util,copy,json
from pathlib import Path
root=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists());s=importlib.util.spec_from_file_location('np',root/'scripts/data_intake_nucleotide_prior.py');m=importlib.util.module_from_spec(s)
try:s.loader.exec_module(m)
except FileNotFoundError:m=None

def module():assert m is not None,'native-symbol nucleotide prior projection missing';return m

def test_long_native_case_and_TU_are_preserved_without_clipping():
 seq='UaCgTN'*2000;t=module().prior_task(seq,domain='RNA_family_reference_native_symbols',organism=None,source_ref={'id':'a'})
 assert json.loads(t.canonical_bytes())['targets']['source_native_sequence']==seq
 assert len(json.loads(t.canonical_bytes())['targets']['source_native_sequence'])==12000
 assert module().base_known_mask('aCNTu')==[True,True,False,True,True]

def test_target_headers_coordinates_proteins_and_length_never_enter_predictors():
 a=module().prior_task('ACGTN',domain='source_DNA_sequence',organism='source organism',source_ref={'header':'a','protein':'hidden','length':5})
 b=module().prior_task('TTT',domain='source_DNA_sequence',organism='source organism',source_ref={'header':'other','protein':'other','length':3})
 assert a.input_bytes()==b.input_bytes();assert json.loads(a.canonical_bytes())['conditions']==dict(prior_domain='source_DNA_sequence',source_organism='source organism')

def test_normalized_and_RC_groups_are_only_derived_groups_native_targets_stay_distinct():
 a=module().sequence_groups('ACGTT');b=module().sequence_groups('AACGT');assert a['reverse_complement_base_order_group']==b['reverse_complement_base_order_group']
 a=module().sequence_groups('acgU');b=module().sequence_groups('ACGT');assert a['native_sha256']!=b['native_sha256'] and a['uppercase_sha256']!=b['uppercase_sha256'];assert a['reverse_complement_base_order_group']==b['reverse_complement_base_order_group']

def test_parent_coordinate_order_is_retained_but_no_family_or_strand_measurement_invented():
 p=module().rfam_parent('FR821777.2/2657646-2657543');assert p['reference_id']=='FR821777.2' and p['source_interval_1based_closed']==[2657646,2657543] and p['observed_strand'] is None
 assert 'genbank:FR821777.2' in p['groups'] and 'genomic_accession_unversioned:FR821777' in p['groups']
 p=module().rfam_parent('unparsed header');assert p['reference_id'] is None and p['source_interval_1based_closed'] is None and p['Rfam_family_id'] is None

def test_dedup_keeps_native_case_and_desired_organism_distinct():
 f=module().prior_identity
 assert f('ACGT','source_DNA_sequence','human')!=f('ACGT','source_DNA_sequence','mouse')
 assert f('ACGT','source_DNA_sequence','human')!=f('acgt','source_DNA_sequence','human')
