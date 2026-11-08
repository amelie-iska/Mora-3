import json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path('/home/iska/Documents/amelie/bio/Mora-3/scripts')))
from data_intake_corpus import CorpusTask,canonical
from data_intake_lincs_dose_qualification import qualify_source_dose


def old(target=1.0):
    return CorpusTask('observational',dict(assay_modality='CellPainting',source_nominal_dose=dict(value=3.3333,unit='micromolar',evidence='author field',rare_nested={'unknown':None}),source_exposure_time={'value':48,'unit':'hours'}),{'profile':[target,None,2**53+1]},{'source_metadata':{'name':'preserved'}},['calibrated_assay_concentration'])


def test_coordinate_preserves_numeric_value_and_removes_physical_unit_assertion():
    d=json.loads(qualify_source_dose(old(),{'sha':'primary'}).canonical_bytes())
    assert 'source_nominal_dose' not in d['conditions']
    q=d['conditions']['source_export_dose_coordinate']
    assert q['value']==3.3333 and q['unit'] is None
    assert q['physical_unit_confirmed'] is False and q['cross_assay_conversion_allowed'] is False
    assert d['conditions']['source_exposure_time']=={'value':48,'unit':'hours'}


def test_full_unknown_source_unit_fields_are_retained_as_audit():
    d=json.loads(qualify_source_dose(old(),{'sha':'primary'}).canonical_bytes())
    assert d['audit']['source_unit_semantic_qualification']['original_source_dose_condition']['rare_nested']=={'unknown':None}
    assert d['audit']['source_metadata']=={'name':'preserved'}


def test_targets_integer_precision_nulls_and_native_masks_are_unchanged():
    a=json.loads(old().canonical_bytes());b=json.loads(qualify_source_dose(old(),{'sha':'primary'}).canonical_bytes())
    assert canonical(a['targets'])==canonical(b['targets'])
    assert b['targets']['profile'][2]==2**53+1 and b['targets']['profile'][1] is None


def test_hidden_target_and_unit_claim_evidence_never_enter_effective_inputs():
    assert qualify_source_dose(old(1),{'sha':'one'}).input_bytes()==qualify_source_dose(old(99),{'sha':'two'}).input_bytes()


def test_already_qualified_payload_cannot_be_double_projected_or_repaired():
    with pytest.raises(ValueError,match='original source dose'):
        qualify_source_dose(qualify_source_dose(old(),{'sha':'primary'}),{'sha':'again'})
