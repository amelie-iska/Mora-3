"""RNAcentral source symbols and metadata must not become invented biological labels."""
import importlib.util
import json
from pathlib import Path
import pytest
ROOT=Path('/home/iska/Documents/amelie/bio/Mora-3')
spec=importlib.util.spec_from_file_location('rnacentral_prior',ROOT/'scripts/data_intake_rnacentral_prior.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_native_axis_and_metadata_target_exclusion():
    seq='aUrNacgtRYSWKMBDHV'
    task=m.prior_task(seq,source_ref={'identifier':'URS0000000001','status':'active','header':'rRNA found in 1 taxa'})
    changed=m.prior_task('TTT',source_ref={'identifier':'URS0000000002','status':'inactive','header':'MUTATED'})
    assert json.loads(task.canonical_bytes())['targets']=={'source_native_sequence':seq}
    assert task.input_bytes()==changed.input_bytes()
    assert json.loads(task.canonical_bytes())['conditions']=={'prior_domain':'RNA_reference_native_symbols','source_organism':None}

def test_nonstandard_source_characters_are_quarantined_not_repaired():
    for seq in ['ACGX','A1C','', 'ACG-']:
        with pytest.raises(ValueError,match='source'):
            m.prior_task(seq,source_ref={})

def test_ambiguity_is_observed_but_not_imputed_as_known_base():
    assert m.base_known_mask('AaTtUuCGnR')==[True]*8+[False,False]
