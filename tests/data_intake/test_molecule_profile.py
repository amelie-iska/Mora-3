import importlib.util,copy,json
from pathlib import Path
import pytest
root=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists())
spec=importlib.util.spec_from_file_location('molecule_profile',root/'scripts/data_intake_molecule_profile.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def array(n=977):return dict(name='profile',axis_identity='fixture',axis_ids=[str(i) for i in range(n)],values=[.25]*n,status='observed',units='source_normalized_or_image_computed_feature_units')
def test_forward_profile_and_postassay_metadata_mutations_do_not_change_inputs():
    a=array();t=m.molecule_profile_task(a,'CCO','L1000','molecule_to_profile',audit={'MOA':'name','QC':4})
    a['values'][0]=999;u=m.molecule_profile_task(a,'CCO','L1000','molecule_to_profile',audit={'MOA':'changed','QC':999})
    assert t.input_bytes()==u.input_bytes() and t.canonical_bytes()!=u.canonical_bytes()
    d=json.loads(t.canonical_bytes());assert len(d['targets']['source_profile_array']['values'])==977
    assert len(d['conditions']['known_profile_axis'])==977
    assert d['conditions']['cell_line'] is None and 'cell_line' in d['missing']
def test_inverse_target_chemical_identity_mutation_cannot_change_profile_inputs():
    a=m.molecule_profile_task(array(),'CCO','L1000','profile_to_molecule',audit={'source_sample':'one'})
    b=m.molecule_profile_task(array(),'CCN','L1000','profile_to_molecule',audit={'source_sample':'two'})
    assert a.input_bytes()==b.input_bytes() and a.canonical_bytes()!=b.canonical_bytes()
    assert 'canonical_isomeric_smiles' not in json.loads(a.input_bytes())['conditions']
def test_complete_morphology_axis_and_explicit_cell_line_are_preserved():
    a=array(1783);t=m.molecule_profile_task(a,'CCO','CellPainting','molecule_to_profile',cell_line='U2OS')
    d=json.loads(t.canonical_bytes());assert d['targets']['source_profile_array']==a
    assert d['conditions']['cell_line']=='U2OS' and 'cell_line' not in d['missing']
    assert 'physical_dose' in d['missing'] and 'exposure_time' in d['missing']
def test_truncated_axes_nonfinite_or_unsupported_direction_are_rejected():
    with pytest.raises(ValueError):m.molecule_profile_task(array(10),'CCO','L1000','molecule_to_profile')
    a=array();a['values'][3]=float('nan')
    with pytest.raises(ValueError):m.molecule_profile_task(a,'CCO','L1000','molecule_to_profile')
    with pytest.raises(ValueError):m.molecule_profile_task(array(),'CCO','L1000','matched_assay_causal_response')
