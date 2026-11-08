from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import sys
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
sys.path.insert(0, '/home/iska/Documents/amelie/bio/Mora-3/scripts')
from data_intake_admet_observations import native_cell, observation, task_view


def record(value='< 1.200E-3', **changes):
    data=dict(entity=dict(molecule_class='peptide',representation_type='native_modified_sequence',
        representation='Ac-D-Ala-Lys[PEG2]-Cys-NH2',native_modifications=[{'position':2,'kind':'D','source':'literal'}]),
        endpoint=dict(source_endpoint='source serum half-life',endpoint_key='native-serum-t1/2',
            readout='half-life',unit='min',unit_basis='source_literal'),
        cell=native_cell(value),known_context={'matrix':'human serum'},
        source_ref={'file':'raw.csv','sha256':'x','row':0},
        native_source_record={'name':'source-name','citation':'original','rare':{'features':[{'fuzzy':'<5','unknown':[None,{'a':True}]}]}},
        evidence_role='experimental_measurement',source_split={'native':'train'},
        admission=dict(source_rights=True,exact_identity=True,endpoint_definition=True,
            task_units_or_declared_native_coordinate=True,split_group_review=True))
    data.update(changes)
    return observation(**data)


def test_censor_and_decimal_literals_preserve_precision_and_bound():
    cell=native_cell(' <= 1.200E-3 ')
    assert cell['native_literal']==' <= 1.200E-3 '
    assert cell['status']=='censored_bound' and cell['relation']=='<='
    assert cell['decimal_literal']=='1.200E-3'
    assert Decimal(cell['decimal_literal'])==Decimal('0.0012')
    assert native_cell('-0.000')['status']=='exact'


def test_missing_not_negative_and_source_missing_lexicon_is_explicit():
    assert not native_cell(None)['observation_present']
    assert native_cell('NA')['status']=='unparsed'
    assert native_cell('0')['observation_present']
    with pytest.raises(ValueError, match='missing or unparsed'):
        task_view(record(''),direction='entity_to_measurement')
    with pytest.raises(TypeError):native_cell(float('nan'))


def test_prediction_excludes_hidden_target_names_and_citations():
    a=record();b=deepcopy(a)
    b['cell']=native_cell('>10000');b['native_source_record']['name']='TARGET HINT'
    b['native_source_record']['citation']='answer label';b['source_ref']['row']=999
    x=task_view(a,direction='entity_to_measurement');y=task_view(b,direction='entity_to_measurement')
    assert x.input_bytes()==y.input_bytes()
    assert x.canonical_bytes()!=y.canonical_bytes()
    b['entity']['representation']='Ac-L-Ala-Lys[PEG2]-Cys-NH2'
    assert x.input_bytes()!=task_view(b,direction='entity_to_measurement').input_bytes()


def test_inverse_excludes_hidden_identity_and_preserves_observed_bound():
    a=record();b=deepcopy(a);b['entity']['representation']='DIFFERENT TARGET'
    b['native_source_record']['citation']='DIFFERENT';b['source_ref']['row']=99
    x=task_view(a,direction='measurement_to_entity')
    assert x.input_bytes()==task_view(b,direction='measurement_to_entity').input_bytes()
    b['cell']=native_cell('>1.200E-3')
    assert x.input_bytes()!=task_view(b,direction='measurement_to_entity').input_bytes()
    assert json.loads(x.canonical_bytes())['targets']['native_entity']==a['entity']


def test_all_original_nested_fields_survive_arrow_even_when_first_row_lacks_them(tmp_path):
    a=record();b=deepcopy(a);a['native_source_record']={}
    payloads=[json.dumps(x,sort_keys=True).encode() for x in (a,b)]
    p=tmp_path/'native.parquet';pq.write_table(pa.table({'native_observation':payloads}),p)
    back=pq.read_table(p).column('native_observation').to_pylist()
    assert back==payloads and json.loads(back[1])['native_source_record']==b['native_source_record']


def test_units_schema_and_literal_are_not_interchangeable_or_guessed():
    a=record();a['endpoint']['unit_basis']='official_source_schema'
    assert task_view(a,direction='entity_to_measurement').input_bytes()!=task_view(record(),direction='entity_to_measurement').input_bytes()
    ep=dict(source_endpoint='native-coordinate',unit_basis='unresolved',unit='uM')
    with pytest.raises(ValueError, match='unresolved physical unit'):record(endpoint=ep)
    with pytest.raises(ValueError, match='not assay predictors'):record(known_context={'name':'leak'})


def test_computed_labels_and_unreviewed_gates_cannot_become_experimental_targets():
    a=record();a['admission']['source_rights']=False
    with pytest.raises(ValueError, match='gates required'):task_view(a,direction='entity_to_measurement')
    a=record(evidence_role='computed')
    with pytest.raises(ValueError, match='separate explicit objective'):task_view(a,direction='entity_to_measurement')
    a['admission']['explicit_nonexperimental_objective']=True
    assert json.loads(task_view(a,direction='entity_to_measurement').canonical_bytes())['conditions']['evidence_role']=='computed'


def test_identity_modifications_never_collapse_to_plain_sequence():
    a=record();b=record();b['entity']['native_modifications'][0]['kind']='L'
    assert task_view(a,direction='entity_to_measurement').input_bytes()!=task_view(b,direction='entity_to_measurement').input_bytes()
    assert task_view(a,direction='measurement_to_entity').canonical_bytes()!=task_view(b,direction='measurement_to_entity').canonical_bytes()


def test_source_axis_cell_count_does_not_assert_independent_experiment():
    r=record()
    assert r["source_native_axis_cell_multiplicity"]==1
    assert r["independent_experimental_measurement_count"] is None
    assert "independent_observation_multiplicity" not in r
