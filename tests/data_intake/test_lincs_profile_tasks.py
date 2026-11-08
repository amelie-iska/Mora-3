import copy,sys,json
from pathlib import Path
import pytest
sys.path.insert(0,str(Path('/home/iska/Documents/amelie/bio/Mora-3/scripts')))
from data_intake_lincs_profile_tasks import source_native_profile_task


def doc(t):return json.loads(t.canonical_bytes())


def array():
    return dict(axis_ids=['probe_'+str(i) for i in range(978)],native_types=['double']*978,
                values=[0.0]*977+[None],observed_mask=[True]*977+[False],feature_units='source preprocessed; per-feature physical units unresolved')


def task(a=None,direction='molecule_to_profile',smiles='CCO',audit=None,dose=0.0411523):
    return source_native_profile_task(a or array(),smiles,'L1000',direction,nominal_dose_micromolar=dose,time_hours=24,cell_line='A549',audit=audit)


def test_forward_preserves_null_and_exact_observed_mask_outside_inputs():
    t=task();assert doc(t)['targets']['source_profile_native_array']['values'][-1] is None
    assert doc(t)['targets']['source_profile_native_array']['observed_mask'][-1] is False
    assert 'observed_mask' not in doc(t)['conditions']
    assert doc(t)['conditions']['source_nominal_dose']['value']==0.0411523


def test_hidden_profile_values_null_mask_and_source_identifiers_do_not_change_forward_inputs():
    a=array();b=copy.deepcopy(a);b['values'][-1]=2.5;b['observed_mask'][-1]=True
    assert task(a,audit={'source_id':'A','names':'x'}).input_bytes()==task(b,audit={'source_id':'B','names':'target name'}).input_bytes()


def test_inverse_hidden_molecule_name_and_audit_do_not_change_inputs():
    assert task(direction='profile_to_molecule',smiles='CCO',audit={'Metadata_SMILES':'CCO'}).input_bytes()==task(direction='profile_to_molecule',smiles='CCN',audit={'Metadata_SMILES':'CCN'}).input_bytes()


def test_native_integer_feature_is_not_silently_cast_to_double():
    a=array();a['native_types'][0]='int64';a['values'][0]=2**53+1
    assert doc(task(a))['targets']['source_profile_native_array']['values'][0]==2**53+1
    a['values'][0]=float(2**53+1)
    with pytest.raises(ValueError,match='int64'):task(a)


def test_mask_cannot_invent_observations_or_zero_fill_missing_values():
    a=array();a['observed_mask'][-1]=True
    with pytest.raises(ValueError,match='validity'):task(a)


def test_control_sentinel_cannot_become_physical_dose_condition():
    with pytest.raises(ValueError,match='positive'):task(dose=-666.0)


def test_cp_and_expression_times_cannot_be_swapped():
    with pytest.raises(ValueError,match='source time'):
        source_native_profile_task(array(),'CCO','L1000','molecule_to_profile',nominal_dose_micromolar=1.0,time_hours=48,cell_line='A549')


def test_profile_is_not_image_segmentation_or_causal_supervision():
    t=task();assert 'raw_image_mask' in doc(t)['missing']
    assert doc(t)['audit']['same_cell_crossassay_pair'] is False
    assert doc(t)['audit']['causal_or_independent_biological_validation'] is False
