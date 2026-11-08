"""Source-policy identity is provenance, not arbitrary biological text."""
import json
import pytest
from mora3.data import rgd1_consumption as gate


@pytest.mark.parametrize('purpose', ['conditioning', 'objective', 'evaluation'])
@pytest.mark.parametrize('field', ['gene_primary', 'gene_name', 'protein_name', 'description'])
def test_biological_annotation_is_not_dataset_identity(field, purpose):
    packet = dict(task='protein_function', conditions={'sequence':'MAGK'},
                  targets={'annotation':{field:'RGD1'}}, audit={field:'RGD1'})
    row = dict(source_family='UniProt-full-record-annotations',
               native_payload=json.dumps(packet).encode())
    before = dict(row)
    assert gate.require_corpus_consumption(row, purpose=purpose) is None
    assert row == before


@pytest.mark.parametrize('field', ['source_family', 'source_release', 'source_revision',
    'source_record_id', 'record_id', 'task', 'task_name', 'source_reaction_family'])
@pytest.mark.parametrize('location', ['outer', 'audit', 'native_payload'])
def test_identity_fields_refuse_even_with_conflicting_outer_family(field, location):
    identity = {field:'RGD1:v6:MR_100001_2'}
    row = {'source_family':'UniProt-full-record-annotations', 'loss_mask':0,
           'admitted_for_task':True, 'model_adapter_supported':True}
    if location == 'outer':
        row.update(identity)
    elif location == 'audit':
        row['audit'] = {'source_ref': [identity]}
    else:
        row['native_payload'] = json.dumps({'audit':{'source_ref':[identity]}}).encode()
    with pytest.raises(ValueError, match='RGD1.*quarantine'):
        gate.require_corpus_consumption(row)


def test_biological_text_never_suppresses_nested_source_identity():
    row = {'source_family':'UniProt', 'audit':{'gene_primary':'RGD1',
           'source_ref':{'source_family':'RGD1-computed-energy'}}}
    with pytest.raises(ValueError, match='RGD1.*quarantine'):
        gate.require_corpus_consumption(row)
