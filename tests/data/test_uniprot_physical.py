import copy
import json
import pytest
from mora3.data import uniprot_physical as physical
from mora3.data.uniprot_annotations import AnnotationPredictionView, _canonical


def view(comments):
    return AnnotationPredictionView('TEST', 'ACD', _canonical(dict(annotation_packet=dict(comments=comments), provider_computed_sequence_properties=[])), _canonical({'split':'train'}))


def test_scoped_kinetic_target_preserves_parent_without_assay_inference():
    parent={'commentType':'BIOPHYSICOCHEMICAL PROPERTIES','molecule':'Isoform 2',
            'kineticParameters':{'michaelisConstants':[{'constant':2.1,'unit':'mM','substrate':'ATP','evidences':[{'evidenceCode':'ECO:0000269'}]}]},
            'phDependence':{'texts':[{'value':'Optimal pH 7'}]}}
    v=view([parent]); rows=physical.physical_targets(v)
    k=next(r for r in rows if r.parameter_type=='MichaelisConstant')
    assert k.source()==parent['kineticParameters']['michaelisConstants'][0]
    assert k.parent()==parent
    assert k.molecule_scope=='Isoform 2'
    assert k.unit_status=='explicit_source_unit'
    assert k.unit=='mM'
    assert k.assay_pH is None and k.assay_temperature is None
    assert not k.canonical_property_supervision
    assert not k.numeric_objective_eligible
    assert k.source_pointer=='/comments/0/kineticParameters/michaelisConstants/0'
    assert v.inputs()=={'canonical_sequence':'ACD'}
    with pytest.raises(ValueError,match='not admitted'):
        k.numeric_target()
    parent['molecule']='changed'
    assert k.molecule_scope=='Isoform 2'
    k.parent()['molecule']='changed'
    assert k.parent()['molecule']=='Isoform 2'


def test_mass_method_error_and_unknown_units_and_missingness():
    v=view([{'commentType':'MASS SPECTROMETRY','molWeight':3100,'molWeightError':2,'method':'MALDI','molecule':'PRO_123'}])
    r,=physical.physical_targets(v)
    assert r.value==3100 and r.method=='MALDI' and r.error==2
    assert r.unit is None and r.unit_status=='unknown'
    assert r.molecule_scope=='PRO_123'
    assert physical.physical_targets(view([]))==()


def test_vmax_never_kcat_and_targets_excluded():
    c={'commentType':'BIOPHYSICOCHEMICAL PROPERTIES','kineticParameters':{'maximumVelocities':[{'velocity':5,'unit':'umol/min/mg'}]}}
    a=view([c]); b=copy.deepcopy(c);b['kineticParameters']['maximumVelocities'][0]['velocity']=999
    assert a.inputs()==view([b]).inputs()
    r=next(r for r in physical.physical_targets(a) if r.parameter_type=='maximumVelocity')
    assert r.value==5 and not r.converted_to_kcat
    assert r.canonical_property_supervision
    assert not r.numeric_objective_eligible
