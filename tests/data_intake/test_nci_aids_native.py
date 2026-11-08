"""Original NCI schema and source-native legacy CTAB semantics, not synthetic coverage."""
from pathlib import Path
import importlib.util
import hashlib
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists())
spec=importlib.util.spec_from_file_location('nci_native',ROOT/'scripts/data_intake_NCI_AIDS_native.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
FIXTURE=Path(__file__).parent/'fixtures/nci_aids_first_native.sdf'

def endpoint(flag='=',unit='M',high='-4',extra=None):
 r={'NSC':'6','Log10HiConc':high,'ConcUnit':unit,'Flag':flag,'Log10IC50':'-5.7','NumExp':'2','StdDev':'0'}
 if extra:r.update(extra)
 return m.parse_endpoint(r,'IC50')

def test_native_legacy_counts_and_syntax_copy_are_explicit():
 raw=FIXTURE.read_text();r=m.parse_structure_record(raw)
 assert r['native_atom_count']==43 and r['native_bond_count']==44
 assert r['derived_syntax_added_M_END'] and r['native_record_sha256']==hashlib.sha256(raw.encode()).hexdigest()
 assert r['source_stereochemistry_known'] is False
 assert r['source_geometry_role']=='source_computed_Corina_no_optimization_not_observed'
 assert FIXTURE.read_text()==raw

def test_computed_coordinates_do_not_change_derived_connectivity():
 raw=FIXTURE.read_text();lines=raw.splitlines()
 for i in range(4,47):lines[i]=f'{i*1.25:10.4f}{-i*2.5:10.4f}{i*3.75:10.4f}'+lines[i][30:]
 assert m.parse_structure_record(raw)['derived_nonstereo_connectivity_smiles']==m.parse_structure_record('\n'.join(lines)+'\n')['derived_nonstereo_connectivity_smiles']

def test_documentation_gaps_and_aggregate_censoring_are_not_point_truth():
 lower=endpoint(flag='<');assert lower['source_documentation_gap'] and not lower['point_target_available']
 volume=endpoint(unit='V');assert not volume['physical_unit_mapping_available'] and not volume['point_target_available']
 assert endpoint(flag='>')['censoring_role']=='source_aggregate_flag_at_least_one_experiment;not_bound_of_aggregate_mean'

def test_zero_reported_sd_does_not_assert_perfect_precision():
 r=endpoint();assert r['point_target_available'] and r['source_stddev']==0 and not r['positive_uncertainty_target_available']
 assert r['source_native_fields']['StdDev']=='0'

def test_repeated_NSC_different_high_dose_contexts_do_not_collapse():
 assert endpoint(high='-4')['context_key']!=endpoint(high='-3.7')['context_key']

def test_unknown_native_columns_are_retained():
 assert endpoint(extra={'rare_nested_context':'future-value'})['source_native_fields']['rare_nested_context']=='future-value'
