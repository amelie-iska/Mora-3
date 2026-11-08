import json
from dataclasses import replace
import pytest
from mora3.data.uniprot_annotations import AnnotationPredictionView, _canonical
from mora3.data import uniprot_residue as r


def view(features):
    return AnnotationPredictionView('P12345', 'MAGK', _canonical({'annotation_packet': {'features': features}}), b'{}')


def feature(kind='Region', start=1, end=3, **extra):
    return dict(type=kind, location={'start': {'value': start, 'modifier': 'EXACT'},
                                   'end': {'value': end, 'modifier': 'EXACT'}}, **extra)


def test_exact_interval_evidence_and_no_target_input():
    f=feature(evidences=[{'evidenceCode': 'ECO:0000250', 'source': 'UniProtKB', 'id':'Q9'}])
    v=view([f]); t=r.residue_targets(v)[0]
    assert t.interval0 == (0,3) and t.endpoint_indices0 is None
    assert t.coordinate_status == 'exact_canonical' and t.source()==f
    assert t.source_pointer=='/features/0'
    x=t.source();x['evidences'].clear()
    assert t.source()==f
    b=r.residue_batch([v], feature_axis=['Region','Binding site'])
    assert b.inputs()==({'canonical_sequence':'MAGK'},)
    assert b.observed==((True,False),) and b.missing_is_negative is False
    changed=view([feature(description='hidden target')])
    assert r.residue_batch([changed],feature_axis=['Region','Binding site']).inputs()==b.inputs()
    assert r.residue_batch([changed],feature_axis=['Region','Binding site']).binding_sha256!=b.binding_sha256


@pytest.mark.parametrize('mode,status', [('fuzzy','fuzzy_or_unknown'),('isoform','unmapped_sequence_scope'),('outside','outside_canonical'),('original','reference_sequence_mismatch')])
def test_ineligible_locations_preserved_without_coordinate_transfer(mode,status):
    f=feature()
    if mode=='fuzzy': f['location']['start']['modifier']='OUTSIDE'
    if mode=='isoform': f['location']['sequence']='P12345-2';f['location']['end']['value']=99
    if mode=='outside': f['location']['end']['value']=5
    if mode=='original': f['alternativeSequence']={'originalSequence':'XXX','alternativeSequences':['QQ']}
    t=r.residue_targets(view([f]))[0]
    assert t.source()==f and t.coordinate_status==status
    assert t.interval0 is None and t.endpoint_indices0 is None


def test_bonds_are_endpoints_never_dense_intervals():
    t=r.residue_targets(view([feature('Disulfide bond',1,4)]))[0]
    assert t.endpoint_indices0==(0,3) and t.interval0 is None


def test_missing_feature_packet_is_not_negative_and_axis_cannot_drop_types():
    b=r.residue_batch([view([])],feature_axis=['Region'])
    assert b.observed==((False,),) and b.targets()==[[]]
    with pytest.raises(ValueError,match='omits'):r.residue_batch([view([feature()])],feature_axis=['Binding site'])
    with pytest.raises(ValueError):r.residue_batch([view([])],feature_axis=['Region','Region'])
    with pytest.raises(TypeError):r.residue_targets({})


def test_cached_targets_cannot_reassign_source_or_scope():
    v=view([feature()]);t=r.residue_targets(v)[0]
    with pytest.raises(ValueError,match='source-bound'):
        r.residue_batch([v],feature_axis=['Region'],assertions=[(replace(t,interval0=(1,4)),)])
    b=r.residue_batch([v],feature_axis=['Region'])
    assert b.binding_sha256!=r.residue_batch([replace(v,provenance_bytes=b'{"new":1}')],feature_axis=['Region']).binding_sha256
