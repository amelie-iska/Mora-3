from dataclasses import replace
import pytest
from mora3.data import uniprot_physical as p
from test_uniprot_physical import view


def example():
    return view([{'commentType':'MASS SPECTROMETRY','molWeight':3100,'molecule':'PRO_123','method':'MALDI'}])


def test_batch_scope_missingness_and_target_exclusion():
    a=example(); empty=view([])
    b=p.physical_batch((a,empty), parameter_axis=('mass_spectrometry_source_mass','MichaelisConstant'))
    assert b.observed==((True,False),(False,False))
    assert b.inputs()==(a.inputs(),empty.inputs())
    assert b.targets()[0][0]['molecule_scope']=='PRO_123'
    assert b.targets()[0][0]['canonical_property_supervision'] is False
    assert b.targets()[0][0]['numeric_objective_eligible'] is False
    assert b.targets()[1]==[]
    changed=replace(a,target_bytes=empty.target_bytes)
    assert p.physical_batch((changed,empty),parameter_axis=b.parameter_axis).inputs()==b.inputs()
    assert p.physical_batch((changed,empty),parameter_axis=b.parameter_axis).binding_sha256!=b.binding_sha256
    b.targets()[0][0]['molecule_scope']='canonical'
    assert b.targets()[0][0]['molecule_scope']=='PRO_123'
    for modified in (replace(a,sequence='AAA'),replace(a,provenance_bytes=b'{}'),replace(a,accession='OTHER')):
        assert p.physical_batch((modified,empty),parameter_axis=b.parameter_axis).binding_sha256!=b.binding_sha256


@pytest.mark.parametrize('field,value',[('unit','Da'),('molecule_scope',None),('assay_pH',7),('numeric_objective_eligible',True),('value',9000),('parent_bytes',b'{}')])
def test_assertion_rebinding_rejects_fabricated_context(field,value):
    a=example(); r,=p.physical_targets(a)
    with pytest.raises(ValueError,match='source-bound'):
        p.physical_batch((a,),parameter_axis=(r.parameter_type,),assertions=((replace(r,**{field:value}),),))


def test_axis_must_cover_observed_targets_without_duplicates():
    a=example()
    for axis in ((),('MichaelisConstant',),('mass_spectrometry_source_mass',)*2):
        with pytest.raises(ValueError,match='axis'):
            p.physical_batch((a,),parameter_axis=axis)
    with pytest.raises(ValueError,match='nonempty'):
        p.physical_batch((),parameter_axis=('mass_spectrometry_source_mass',))
