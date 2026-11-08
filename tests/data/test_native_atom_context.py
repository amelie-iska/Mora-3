"""Native identity/missingness boundaries; no coordinate-derived predictors."""
import importlib
import copy
import pytest


def api():
    spec = importlib.util.find_spec('mora3.data.native_atom_context')
    assert spec is not None, 'native atom context contract missing'
    return importlib.import_module('mora3.data.native_atom_context')


def metadata():
    return dict(source_record_id='source:v4:reaction', source_reaction_group='source:v4:group',
                source_native_split='val', atomic_numbers=[8,6,1],
                electronic_charge=None, spin_multiplicity=None,
                electronic_charge_known=False, spin_multiplicity_known=False,
                task_admitted=False)


def build(row=None, **kw):
    return api().native_atom_context(metadata() if row is None else row,
        ordered_atom_ids=('a','b','c'), source_payload_sha256='a'*64, **kw)


def test_preserves_order_unknowns_and_no_admission():
    c=build()
    assert c.atomic_numbers == (8,6,1)
    assert c.ordered_atom_ids == ('a','b','c')
    assert c.electronic_charge is None and c.spin_multiplicity is None
    assert c.missing_context == ('electronic_charge','spin_multiplicity','enzyme','cellular_assay','solvent')
    assert c.source_native_split == 'val'
    with pytest.raises(ValueError, match='admission'): c.require_model_use()


def test_coordinates_are_opaque_and_input_unchanged():
    row=metadata(); row['roles']={'transition_state':object()}
    before=copy.copy(row)
    assert build(row)==build()
    assert row==before


@pytest.mark.parametrize('field,value',[
 ('atomic_numbers',[]),('atomic_numbers',[8,True,1]),('atomic_numbers',[8,119,1]),
 ('atomic_numbers',[8,6]),('source_record_id',''),('source_reaction_group',''),
 ('source_native_split',''),('electronic_charge',0),('spin_multiplicity',1),
 ('electronic_charge_known',True),('spin_multiplicity_known',True),
 ('task_admitted',True),('task_admitted',0)])
def test_bad_or_unqualified_metadata_refused(field,value):
    row=metadata();row[field]=value
    with pytest.raises(ValueError):build(row)


@pytest.mark.parametrize('key',['electronic_charge','spin_multiplicity','electronic_charge_known','spin_multiplicity_known','task_admitted'])
def test_missing_declarations_refused(key):
    row=metadata();del row[key]
    with pytest.raises(ValueError):build(row)


@pytest.mark.parametrize('ids',[('a','a','b'),('a','b'),('a','b',''),('a','b',3)])
def test_bad_axis_refused(ids):
    with pytest.raises(ValueError):api().native_atom_context(metadata(),ordered_atom_ids=ids,source_payload_sha256='a'*64)


@pytest.mark.parametrize('digest',['','g'*64,'A'*64,True])
def test_bad_source_digest_refused(digest):
    with pytest.raises(ValueError):api().native_atom_context(metadata(),ordered_atom_ids=('a','b','c'),source_payload_sha256=digest)


def test_frozen_value_and_order_sensitive_binding():
    c=build()
    with pytest.raises(AttributeError):c.atomic_numbers=(1,6,8)
    row=metadata();row['atomic_numbers']=[1,6,8]
    assert build(row)!=c
